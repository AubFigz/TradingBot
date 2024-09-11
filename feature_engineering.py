import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sqlalchemy import create_engine
from utils import load_config, setup_logging, append_to_csv, load_data_from_db, save_data_to_db, validate_data, \
    detect_anomalies, extract_datetime_features, setup_prometheus, send_alert, analyze_price_sentiment
from risk_management import calculate_risk_metrics, perform_correlation_analysis, dynamic_stop_loss_take_profit, \
    perform_scenario_analysis, generate_risk_report, save_risk_report, evaluate_basic_risks
from prometheus_client import Gauge
from dask.distributed import Client
from featuretools import dfs, EntitySet
import asyncio

# Load configuration
config = load_config()

# Setup logging
logger = setup_logging(config)

# Setup Prometheus metrics server
setup_prometheus(config['prometheus']['port'])

# Prometheus metrics
data_rows_processed_gauge = Gauge('data_rows_processed', 'Number of Data Rows Processed')
num_anomalies_gauge = Gauge('num_anomalies', 'Number of Anomalies Detected')
num_features_selected_gauge = Gauge('num_features_selected', 'Number of Features Selected')
signals_generated_gauge = Gauge('signals_generated', 'Number of Trading Signals Generated')
risk_metrics_gauges = {
    'var': Gauge('var', 'Value at Risk'),
    'cvar': Gauge('cvar', 'Conditional Value at Risk'),
    'sharpe_ratio': Gauge('sharpe_ratio', 'Sharpe Ratio'),
    'sortino_ratio': Gauge('sortino_ratio', 'Sortino Ratio'),
    'max_drawdown': Gauge('max_drawdown', 'Max Drawdown'),
    'beta': Gauge('beta', 'Beta')
}
correlation_matrix_gauge = Gauge('correlation_matrix', 'Correlation Matrix')
dynamic_stop_loss_gauge = Gauge('dynamic_stop_loss', 'Dynamic Stop Loss')
dynamic_take_profit_gauge = Gauge('dynamic_take_profit', 'Dynamic Take Profit')
feature_importance_gauge = Gauge('feature_importance', 'Feature Importance')

# Initialize Dask client for distributed computing
client = Client()

def compute_technical_indicators(data):
    try:
        indicators = {
            'sma_10': ta.trend.sma_indicator(data['close'], window=10),
            'sma_50': ta.trend.sma_indicator(data['close'], window=50),
            'ema_10': ta.trend.ema_indicator(data['close'], window=10),
            'ema_50': ta.trend.ema_indicator(data['close'], window=50),
            'momentum': ta.momentum.roc(data['close'], window=4),
            'volatility': ta.volatility.bollinger_hband(data['close'], window=10),
            'rsi': ta.momentum.rsi(data['close']),
            'macd': ta.trend.macd(data['close']),
            'macd_diff': ta.trend.macd_diff(data['close']),
            'macd_signal': ta.trend.macd_signal(data['close']),
            'bollinger_hband': ta.volatility.bollinger_hband(data['close']),
            'bollinger_lband': ta.volatility.bollinger_lband(data['close']),
            'stoch': ta.momentum.stoch(data['high'], data['low'], data['close']),
            'williams_r': ta.momentum.williams_r(data['high'], data['low']),
            'ichimoku_a': ta.trend.ichimoku_a(data['high'], data['low']),
            'ichimoku_b': ta.trend.ichimoku_b(data['high'], data['low']),
            'parabolic_sar': ta.trend.psar(data['high'], data['low'], data['close']),
            'fibonacci_retracement': ta.trend.fibonacci_retracement(data['high'], data['low'], data['close']),
            'atr': ta.volatility.average_true_range(data['high'], data['low'], data['close']),
            'cmo': ta.momentum.cmo(data['close']),
            'cci': ta.trend.cci(data['high'], data['low']),
            'mfi': ta.volume.money_flow_index(data['high'], data['low'], data['close'], data['volume']),
            'obv': ta.volume.on_balance_volume(data['close'], data['volume']),
            'proc': ta.momentum.roc(data['close']),
            'uo': ta.momentum.ultimate_oscillator(data['high'], data['low'], data['close']),
            'vpt': ta.volume.volume_price_trend(data['close'], data['volume']),
            'keltner_hband': ta.volatility.keltner_channel_hband(data['high'], data['low'], data['close']),
            'keltner_lband': ta.volatility.keltner_channel_lband(data['high'], data['low'], data['close']),
            'cmf': ta.volume.chaikin_money_flow(data['high'], data['low'], data['close'], data['volume'])
        }
        logger.info("Technical indicators computed successfully.")
        return pd.DataFrame(indicators)
    except Exception as e:
        logger.error(f"Error computing technical indicators: {e}")
        send_alert(f"Error computing technical indicators: {e}")
        return pd.DataFrame()

def add_technical_indicators(data):
    try:
        indicators_df = compute_technical_indicators(data)
        data_before = data.shape[0]
        data = pd.concat([data, indicators_df], axis=1)
        data = data.dropna()  # Dropping rows with missing values after adding indicators
        logger.info(f"Technical indicators added to data. Dropped {data_before - data.shape[0]} rows due to missing values.")
        return data
    except Exception as e:
        logger.error(f"Error adding technical indicators: {e}")
        send_alert(f"Error adding technical indicators: {e}")
        return data

def generate_lagged_features(data, columns, lags):
    try:
        for column in columns:
            for lag in lags:
                data[f'{column}_lag_{lag}'] = data[column].shift(lag)
        logger.info("Lagged features generated successfully.")
        return data
    except Exception as e:
        logger.error(f"Error generating lagged features: {e}")
        send_alert(f"Error generating lagged features: {e}")
        return data

def generate_rolling_statistics(data, columns, windows):
    try:
        for column in columns:
            for window in windows:
                data[f'{column}_rolling_mean_{window}'] = data[column].rolling(window=window).mean()
                data[f'{column}_rolling_std_{window}'] = data[column].rolling(window=window).std()
        logger.info("Rolling statistics generated successfully.")
        return data
    except Exception as e:
        logger.error(f"Error generating rolling statistics: {e}")
        send_alert(f"Error generating rolling statistics: {e}")
        return data

def generate_sentiment_features(data):
    try:
        data['sentiment_momentum'] = data['sentiment_score'].diff()
        data['sentiment_volatility'] = data['sentiment_score'].rolling(window=10).std()
        logger.info("Sentiment features generated successfully.")
        return data
    except Exception as e:
        logger.error(f"Error generating sentiment features: {e}")
        send_alert(f"Error generating sentiment features: {e}")
        return data

def perform_feature_selection(X, y):
    try:
        # RFE with RandomForest
        rf_model = RandomForestRegressor()
        rfe_selector = RFE(rf_model, n_features_to_select=config['feature_selection']['rfe_n_features'])
        X_rfe = rfe_selector.fit_transform(X, y)

        # LASSO with cross-validation
        lasso_model = LassoCV(cv=5)
        lasso_selector = SelectFromModel(lasso_model)
        X_lasso = lasso_selector.fit_transform(X, y)

        # Decision Trees with SelectFromModel
        dt_model = GradientBoostingClassifier()
        dt_selector = SelectFromModel(dt_model)
        X_dt = dt_selector.fit_transform(X, y)

        # Combine features selected by all methods
        X_combined = np.hstack((X_rfe, X_lasso, X_dt))
        num_features_selected_gauge.set(X_combined.shape[1])

        # Cross-validation to evaluate combined feature selection
        cv_scores = cross_val_score(rf_model, X_combined, y, cv=StratifiedKFold(5))
        logger.info(f"Combined feature selection cross-validation scores: {cv_scores.mean()}")

        return X_combined
    except Exception as e:
        logger.error(f"Error in advanced feature selection: {e}")
        send_alert(f"Error in advanced feature selection: {e}")
        return X

def apply_feature_transformations(X):
    try:
        poly = PolynomialFeatures(degree=2, interaction_only=True)
        X_poly = poly.fit_transform(X)

        X_log = np.log1p(X)
        X_transformed = np.hstack((X, X_poly, X_log))

        logger.info("Feature transformations applied successfully.")
        return X_transformed
    except Exception as e:
        logger.error(f"Error applying feature transformations: {e}")
        send_alert(f"Error applying feature transformations: {e}")
        return X

def automated_feature_engineering(data):
    try:
        es = EntitySet(id="market_data")
        es = es.entity_from_dataframe(entity_id="data", dataframe=data, index="index")

        feature_matrix, feature_defs = dfs(entityset=es, target_entity="data")
        logger.info("Automated feature engineering completed successfully.")
        return feature_matrix
    except Exception as e:
        logger.error(f"Error in automated feature engineering: {e}")
        send_alert(f"Error in automated feature engineering: {e}")
        return data

def monitor_feature_importance(model, X, y):
    try:
        model.fit(X, y)
        importance = model.feature_importances_

        for i, importance_value in enumerate(importance):
            feature_importance_gauge.set(importance_value)
            logger.info(f"Feature {i} importance: {importance_value}")

        logger.info("Feature importance monitored successfully.")
    except Exception as e:
        logger.error(f"Error monitoring feature importance: {e}")
        send_alert(f"Error monitoring feature importance: {e}")

def preprocess_data(data):
    try:
        # Capture the original columns before adding new features
        original_columns = data.columns

        data = validate_data(data, config['data_validation']['required_columns'])
        data_rows_processed_gauge.set(len(data))

        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')

        # Detect and remove anomalies
        data = detect_anomalies(data, ['close'], config['anomaly_detection']['contamination'])
        num_anomalies_gauge.set(data['anomaly'].sum())

        # Extract datetime features
        data = extract_datetime_features(data)

        # Add technical indicators (excluding those computed in data_preparation.py)
        data = add_technical_indicators(data)

        # Generate lagged features
        lagged_columns = ['close', 'volume']
        lags = [1, 2, 3, 5, 10]
        data = generate_lagged_features(data, lagged_columns, lags)

        # Generate rolling statistics
        rolling_columns = ['close', 'volume']
        windows = [5, 10, 20]
        data = generate_rolling_statistics(data, rolling_columns, windows)

        # Generate sentiment features
        data = generate_sentiment_features(data)

        # Analyze price sentiment
        data = analyze_price_sentiment(data)

        # Identify new numeric columns
        new_numeric_cols = data.select_dtypes(include=np.number).columns.difference(original_columns)

        # Normalize only the newly created features
        if not new_numeric_cols.empty:
            scaler = StandardScaler()
            data[new_numeric_cols] = scaler.fit_transform(data[new_numeric_cols])

        # Automated feature engineering
        data = automated_feature_engineering(data)

        logger.info("Data preprocessing completed.")
        return data
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        send_alert(f"Error preprocessing data: {e}")
        return data

async def main():
    try:
        # Database connections
        feature_engineering_output_engine = create_engine(config['database']['feature_engineering_output'])
        backtesting_db_engine = create_engine(config['database']['backtesting_data'])
        risk_management_db_engine = create_engine(config['database']['risk_management_data'])

        # Load data from database
        data = load_data_from_db(feature_engineering_output_engine, 'final_data')

        # Preprocess data
        data = preprocess_data(data)

        # Split data into training and unseen test data
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        # Save the unseen test data for later model validation
        save_data_to_db(test_data, feature_engineering_output_engine, 'test_data_with_features')
        append_to_csv(test_data, config['data_fetching']['test_data_with_features_file'])

        # Calculate risk metrics after preprocessing
        risk_metrics = calculate_risk_metrics(train_data)

        # Evaluate basic risks using thresholds from the config file
        risk_evaluation_passed = evaluate_basic_risks(risk_metrics, config['risk_management']['alert_threshold'])

        if not risk_evaluation_passed:
            logger.warning("Risk evaluation did not pass. Adjusting strategy or alerting user.")
            return  # Exit the script if the strategy does not pass basic risk evaluation

        # Save preprocessed training data for backtesting
        save_data_to_db(train_data, feature_engineering_output_engine, 'preprocessed_data_for_backtesting')

        # Load backtesting results from the backtesting database
        backtesting_results = load_data_from_db(backtesting_db_engine, 'backtesting_results')

        # Incorporate backtesting results
        train_data = pd.concat([train_data, backtesting_results], axis=1)

        # Feature selection
        X = train_data.drop(columns=['price'])
        y = train_data['price']
        X_selected = perform_feature_selection(X, y)

        # Apply feature transformations
        X_transformed = apply_feature_transformations(X_selected)

        # Monitor feature importance
        model = RandomForestRegressor()
        monitor_feature_importance(model, X_transformed, y)

        # Comprehensive Risk Management (recalculate risk metrics with updated data)
        risk_metrics = calculate_risk_metrics(train_data)
        for metric, value in risk_metrics.items():
            risk_metrics_gauges[metric].set(value)
        logger.info(f"Risk Metrics: {risk_metrics}")

        # Correlation analysis for diversification
        correlation_analysis = perform_correlation_analysis(train_data)
        correlation_matrix_gauge.set(correlation_analysis.values.flatten().mean())
        logger.info(f"Correlation Analysis: {correlation_analysis}")

        # Dynamic stop loss and take profit
        train_data = dynamic_stop_loss_take_profit(train_data)
        dynamic_stop_loss_gauge.set(train_data['stop_loss'].mean())
        dynamic_take_profit_gauge.set(train_data['take_profit'].mean())

        # Perform scenario analysis
        scenario_results = perform_scenario_analysis(train_data, config['risk_management']['scenario_analysis'])
        logger.info(f"Scenario Analysis Results: {scenario_results}")

        # Generate and save risk report
        report = generate_risk_report(train_data, risk_metrics, correlation_analysis)
        save_risk_report(report, config['data_fetching']['risk_management_results_file'])

        # Save risk management results to the risk management database
        save_data_to_db(report, risk_management_db_engine, 'risk_management_results')

        # Save processed training data with features to database and CSV
        save_data_to_db(train_data, feature_engineering_output_engine, 'final_data_with_features')
        append_to_csv(train_data, config['data_fetching']['final_data_with_features_file'])

        # Save backtesting results to database and CSV
        save_data_to_db(backtesting_results, backtesting_db_engine, 'backtesting_results')
        append_to_csv(backtesting_results, config['data_fetching']['backtesting_results_file'])

        logger.info("Feature engineering and risk management completed, and data saved successfully.")
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        send_alert(f"Error in feature engineering: {e}")

if __name__ == "__main__":
    asyncio.run(main())

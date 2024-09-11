import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, explained_variance_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestRegressor
import optuna
import joblib
import asyncio
import aiohttp
from prometheus_client import start_http_server, Gauge
from sqlalchemy import create_engine
from utils import load_config, setup_logging, validate_data, load_data_from_db, save_data_to_db, setup_prometheus, \
    analyze_price_sentiment
from risk_management import (
    calculate_risk_metrics, perform_correlation_analysis, dynamic_stop_loss_take_profit,
    perform_scenario_analysis, generate_risk_report, save_risk_report, evaluate_basic_risks
)

# Load configuration
config = load_config()

# Configure logging
logger = setup_logging(config)

# Setup Prometheus
setup_prometheus(config['prometheus']['port'])

# Prometheus metrics
cumulative_returns_gauge = Gauge('cumulative_returns', 'Cumulative Returns')
annualized_return_gauge = Gauge('annualized_return', 'Annualized Return')
annualized_volatility_gauge = Gauge('annualized_volatility', 'Annualized Volatility')
sharpe_ratio_gauge = Gauge('sharpe_ratio', 'Sharpe Ratio')
max_drawdown_gauge = Gauge('max_drawdown', 'Max Drawdown')
sortino_ratio_gauge = Gauge('sortino_ratio', 'Sortino Ratio')
calmar_ratio_gauge = Gauge('calmar_ratio', 'Calmar Ratio')
feature_importance_gauge = Gauge('feature_importance', 'Feature Importance')
beta_gauge = Gauge('beta', 'Beta')
alpha_gauge = Gauge('alpha', 'Alpha')
mse_gauge = Gauge('mse', 'Mean Squared Error')
mae_gauge = Gauge('mae', 'Mean Absolute Error')
r2_gauge = Gauge('r2', 'R2 Score')
explained_variance_gauge = Gauge('explained_variance', 'Explained Variance Score')

# Function to apply PCA for dimensionality reduction
def apply_pca(data, n_components):
    try:
        pca = PCA(n_components=n_components)
        data_pca = pca.fit_transform(data)
        logger.info("PCA applied successfully.")
        return data_pca
    except Exception as e:
        logger.error(f"Error applying PCA: {e}")
        raise

# Function to calculate returns
def calculate_returns(data):
    try:
        data['returns'] = data['price'].pct_change()
        return data
    except Exception as e:
        logger.error(f"Error calculating returns: {e}")
        raise

# Function to generate signals using LSTM
def generate_signals_lstm(data, model, scaler, look_back):
    try:
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['price']

        scaled_data = scaler.transform(data[['price']])
        for i in range(look_back, len(data)):
            X_test = np.array(scaled_data[i - look_back:i])
            X_test = np.reshape(X_test, (1, X_test.shape[0], 1))
            signals.loc[signals.index[i], 'signal'] = model.predict(X_test)

        signals['signal'] = signals['signal'].shift(1)  # Shift signals to align with the next day's price change
        return signals
    except Exception as e:
        logger.error(f"Error generating LSTM signals: {e}")
        raise

# Function to generate signals using RandomForestRegressor
def generate_signals_rf(data, model):
    try:
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['price']

        features = data.drop(columns=['price'])
        signals['signal'] = model.predict(features)
        signals['signal'] = signals['signal'].shift(1)  # Shift signals to align with the next day's price change
        return signals
    except Exception as e:
        logger.error(f"Error generating RandomForest signals: {e}")
        raise

# Function to determine dataset structure (e.g., imbalanced classes) and apply necessary techniques
def determine_dataset_structure(data):
    try:
        class_distribution = data['price'].value_counts()
        imbalance_ratio = class_distribution.max() / class_distribution.min()

        if imbalance_ratio > config['data_validation'].get('imbalance_threshold', 1.5):
            logger.info(f"Imbalanced dataset detected with ratio {imbalance_ratio}. Applying StratifiedKFold.")
            return TimeSeriesSplit(n_splits=5)
        else:
            logger.info("Balanced dataset detected. Using TimeSeriesSplit.")
            return TimeSeriesSplit(n_splits=5)
    except Exception as e:
        logger.error(f"Error determining dataset structure: {e}")
        raise

# Function to backtest strategy
def backtest_strategy(signals, initial_balance, transaction_cost):
    try:
        balance = initial_balance
        holdings = 0
        signals['balance'] = 0
        signals['holdings'] = 0
        signals['transactions'] = []

        for i in range(1, len(signals)):
            if signals['signal'][i] > 0:  # Buy signal
                if balance > 0:
                    holdings += balance / signals['price'][i]
                    balance -= holdings * signals['price'][i] * transaction_cost
                    signals['transactions'].append((signals.index[i], 'buy', holdings, signals['price'][i]))
            elif signals['signal'][i] < 0:  # Sell signal
                if holdings > 0:
                    balance += holdings * signals['price'][i]
                    balance -= holdings * signals['price'][i] * transaction_cost
                    signals['transactions'].append((signals.index[i], 'sell', holdings, signals['price'][i]))
                    holdings = 0

            signals.loc[signals.index[i], 'balance'] = balance
            signals.loc[signals.index[i], 'holdings'] = holdings * signals['price'][i]

        signals['total'] = signals['balance'] + signals['holdings']
        return signals
    except Exception as e:
        logger.error(f"Error in backtesting strategy: {e}")
        raise

# Function to evaluate strategy
def evaluate_strategy(signals, benchmark_data=None):
    try:
        returns = signals['total'].pct_change()
        cumulative_returns = (1 + returns).cumprod() - 1
        annualized_return = cumulative_returns[-1] ** (252 / len(returns)) - 1
        annualized_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility
        max_drawdown = (signals['total'].cummax() - signals['total']).max()
        sortino_ratio = annualized_return / returns[returns < 0].std() * np.sqrt(252)
        calmar_ratio = annualized_return / max_drawdown

        # Regression metrics
        mse = mean_squared_error(signals['price'], signals['signal'])
        mae = mean_absolute_percentage_error(signals['price'], signals['signal'])
        r2 = r2_score(signals['price'], signals['signal'])
        explained_variance = explained_variance_score(signals['price'], signals['signal'])

        mse_gauge.set(mse)
        mae_gauge.set(mae)
        r2_gauge.set(r2)
        explained_variance_gauge.set(explained_variance)

        # Calculate beta and alpha if benchmark data is available
        if benchmark_data is not None:
            beta, alpha = calculate_beta_alpha(signals, benchmark_data)
            beta_gauge.set(beta)
            alpha_gauge.set(alpha)
        else:
            logger.info("Benchmark data not available, skipping beta and alpha calculations.")

        # Update Prometheus metrics
        cumulative_returns_gauge.set(cumulative_returns[-1])
        annualized_return_gauge.set(annualized_return)
        annualized_volatility_gauge.set(annualized_volatility)
        sharpe_ratio_gauge.set(sharpe_ratio)
        max_drawdown_gauge.set(max_drawdown)
        sortino_ratio_gauge.set(sortino_ratio)
        calmar_ratio_gauge.set(calmar_ratio)

        logging.info(f"Cumulative Returns: {cumulative_returns[-1]}")
        logging.info(f"Annualized Return: {annualized_return}")
        logging.info(f"Annualized Volatility: {annualized_volatility}")
        logging.info(f"Sharpe Ratio: {sharpe_ratio}")
        logging.info(f"Max Drawdown: {max_drawdown}")
        logging.info(f"Sortino Ratio: {sortino_ratio}")
        logging.info(f"Calmar Ratio: {calmar_ratio}")
        logging.info(f"MSE: {mse}")
        logging.info(f"MAE: {mae}")
        logging.info(f"R2 Score: {r2}")
        logging.info(f"Explained Variance: {explained_variance}")

        return cumulative_returns[
            -1], annualized_return, annualized_volatility, sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio
    except Exception as e:
        logger.error(f"Error evaluating strategy: {e}")
        raise

# Function to calculate beta and alpha
def calculate_beta_alpha(signals, benchmark_data):
    try:
        returns = signals['total'].pct_change().dropna()
        benchmark_returns = benchmark_data['price'].pct_change().dropna()
        covariance_matrix = np.cov(returns, benchmark_returns)
        beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
        alpha = returns.mean() - beta * benchmark_returns.mean()
        logger.info(f"Calculated beta: {beta}, alpha: {alpha}")
        return beta, alpha
    except Exception as e:
        logger.error(f"Error calculating beta and alpha: {e}")
        raise

# Function for hyperparameter optimization using Bayesian Optimization (Optuna)
def optimize_hyperparameters(trial):
    try:
        look_back = trial.suggest_int('look_back', 10, 100)
        transaction_cost = trial.suggest_float('transaction_cost', 0.0001, 0.01)

        # Ensure validation set is used
        X_train, X_val, y_train, y_val = train_test_split(data.drop(columns=['price']), data['price'], test_size=0.2,
                                                          shuffle=False)

        # Decide which model to use based on configuration
        if config['model'].get('use_rf', False):
            model_type = 'RandomForestRegressor'
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X_train, y_train)
            signals = generate_signals_rf(pd.concat([X_val, y_val], axis=1), rf_model)
        else:
            model_type = 'LSTM'
            signals = generate_signals_lstm(pd.concat([X_val, y_val], axis=1), lstm_model, scaler, look_back)

        signals = backtest_strategy(signals, config['trading_env']['initial_balance'], transaction_cost)
        cumulative_returns, annualized_return, annualized_volatility, sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio = evaluate_strategy(
            signals)

        logging.info(f"Optimization using {model_type}: Sharpe Ratio: {sharpe_ratio}")
        return -sharpe_ratio  # Minimize negative Sharpe ratio
    except Exception as e:
        logger.error(f"Error during hyperparameter optimization: {e}")
        raise

# Function to normalize and validate data
def normalize_and_validate_data(data):
    try:
        data = validate_data(data)
        data['price'] = data['price'].astype(float)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        data = calculate_returns(data)

        # Apply sentiment analysis
        data = analyze_price_sentiment(data)

        return data
    except Exception as e:
        logger.error(f"Error normalizing and validating data: {e}")
        raise

# Function to monitor feature importance
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
        raise

# Function to generate ensemble predictions
def ensemble_predictions(data):
    try:
        rf_model = RandomForestRegressor(n_estimators=100)
        lstm_model = load_model('lstm_model.h5')

        # Generate predictions from both models
        rf_signals = generate_signals_rf(data, rf_model)
        lstm_signals = generate_signals_lstm(data, lstm_model, scaler, config['model']['look_back'])

        # Combine predictions by averaging
        data['ensemble_signal'] = (rf_signals['signal'] + lstm_signals['signal']) / 2
        logger.info("Ensemble predictions generated successfully.")
        return data
    except Exception as e:
        logger.error(f"Error generating ensemble predictions: {e}")
        raise

# Function to apply feature transformations
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
        raise

if __name__ == "__main__":
    try:
        # Start Prometheus metrics server
        start_http_server(config['prometheus']['port'])

        # Load data from feature engineering output database
        feature_engineering_output_engine = create_engine(config['database']['feature_engineering_output'])
        data = load_data_from_db(feature_engineering_output_engine, 'preprocessed_data_for_backtesting')

        # Normalize and validate data
        data = normalize_and_validate_data(data)

        # Apply PCA if necessary
        if config.get('pca', {}).get('apply', False):
            data = apply_pca(data, n_components=config['pca']['n_components'])

        # Optimize hyperparameters using Optuna with parallel trials
        study = optuna.create_study(direction='minimize')
        study.optimize(optimize_hyperparameters, n_trials=50, n_jobs=-1)  # Parallelize trials
        logging.info(f"Best hyperparameters: {study.best_params}")

        # Generate ensemble predictions with optimized hyperparameters
        data = ensemble_predictions(data)

        # Apply dynamic stop loss and take profit
        data = dynamic_stop_loss_take_profit(data)

        # Backtest strategy
        signals = backtest_strategy(data, config['trading_env']['initial_balance'],
                                    study.best_params['transaction_cost'])

        # Evaluate strategy with benchmark data for beta and alpha calculation
        try:
            benchmark_data = load_data_from_db(feature_engineering_output_engine,
                                               'benchmark_data')  # Assuming benchmark data is stored
        except Exception as e:
            logger.warning(f"Benchmark data not found: {e}")
            benchmark_data = None

        cumulative_returns, annualized_return, annualized_volatility, sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio = evaluate_strategy(
            signals, benchmark_data)

        # Perform risk metrics calculations including benchmark data
        risk_metrics = calculate_risk_metrics(data, benchmark_data)
        logger.info(f"Risk Metrics: {risk_metrics}")

        # Perform correlation analysis
        correlation_matrix = perform_correlation_analysis(data)
        logger.info(f"Correlation Analysis: {correlation_matrix}")

        # Perform scenario analysis
        scenarios = {
            'Bear Market': scenario_bear_market,
            'Bull Market': scenario_bull_market,
            'High Volatility': scenario_high_volatility,
            'Low Volatility': scenario_low_volatility,
            'Flash Crash': scenario_flash_crash,
            'Interest Rate Hike': scenario_interest_rate_hike,
            'Currency Devaluation': scenario_currency_devaluation,
            'Sector Shock': scenario_sector_shock
        }
        scenario_results = perform_scenario_analysis(data, scenarios)
        logger.info(f"Scenario Analysis Results: {scenario_results}")

        # Generate and save risk report
        report = generate_risk_report(data, risk_metrics, correlation_matrix)
        save_risk_report(report, config['risk_management']['report_file'])

        # Save backtesting results
        signals.to_csv('backtesting_results.csv', index=True)

        # Save results to backtesting database
        backtesting_db_engine = create_engine(config['database']['backtesting'])
        save_data_to_db(signals, backtesting_db_engine, 'backtesting_results')
        logging.info("Backtesting completed and results saved successfully.")

        # Monitor feature importance for adaptive feature selection
        X = data.drop(columns=['price'])
        y = data['price']
        monitor_feature_importance(rf_model, X, y)

        # Start fetching real-time data
        asyncio.run(fetch_real_time_data())

    except Exception as e:
        logger.error(f"Error during backtesting: {e}")
        raise

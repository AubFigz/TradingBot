import pandas as pd
import numpy as np
import logging
import asyncio
import websockets
import json
from datetime import datetime
from sqlalchemy import create_engine
from prometheus_client import start_http_server, Summary, Counter, Gauge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from utils import load_data_from_db, save_data_to_db, load_config, setup_logging, validate_data, \
    extract_datetime_features, send_alert

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
RISK_METRICS_GAUGE = Gauge('risk_metrics', 'Risk Metrics', ['metric'])
DATA_AGGREGATION_COUNTER = Counter('data_aggregation_calls', 'Number of data aggregation calls')
REAL_TIME_DATA_COUNTER = Counter('real_time_data_processed', 'Number of real-time data points processed')
LATENCY_GAUGE = Gauge('data_processing_latency', 'Latency in processing real-time data')


# Machine Learning Model for Risk Prediction
class RiskPredictor:
    def __init__(self, data):
        self.model = RandomForestRegressor()
        self.X = data.drop(columns=['close'])
        self.y = data['close'].pct_change().dropna()
        self.train_model()

    def train_model(self):
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=5)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        y_pred = self.model.predict(X_test)
        mse = np.mean((y_test - y_pred) ** 2)
        logger.info(f"Model training completed with MSE: {mse}")
        logger.info(f"Best model parameters: {grid_search.best_params_}")

    def predict(self, X):
        return self.model.predict(X)


# Calculate Value at Risk (VaR)
def calculate_var(data, confidence_level=0.95):
    try:
        returns = data['close'].pct_change().dropna()
        var = np.percentile(returns, (1 - confidence_level) * 100)
        RISK_METRICS_GAUGE.labels('var').set(var)
        return var
    except Exception as e:
        logger.error(f"Error calculating VaR: {e}")
        return None


# Calculate Conditional Value at Risk (CVaR)
def calculate_cvar(data, confidence_level=0.95):
    try:
        returns = data['close'].pct_change().dropna()
        var = calculate_var(data, confidence_level)
        cvar = returns[returns <= var].mean()
        RISK_METRICS_GAUGE.labels('cvar').set(cvar)
        return cvar
    except Exception as e:
        logger.error(f"Error calculating CVaR: {e}")
        return None


# Calculate Sharpe Ratio
def calculate_sharpe_ratio(data, risk_free_rate=0.01):
    try:
        returns = data['close'].pct_change().dropna()
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
        RISK_METRICS_GAUGE.labels('sharpe_ratio').set(sharpe_ratio)
        return sharpe_ratio
    except Exception as e:
        logger.error(f"Error calculating Sharpe Ratio: {e}")
        return None


# Calculate Sortino Ratio
def calculate_sortino_ratio(data, risk_free_rate=0.01):
    try:
        returns = data['close'].pct_change().dropna()
        downside_returns = returns[returns < 0]
        excess_returns = returns - risk_free_rate / 252
        sortino_ratio = np.mean(excess_returns) / np.std(downside_returns)
        RISK_METRICS_GAUGE.labels('sortino_ratio').set(sortino_ratio)
        return sortino_ratio
    except Exception as e:
        logger.error(f"Error calculating Sortino Ratio: {e}")
        return None


# Calculate Maximum Drawdown
def calculate_max_drawdown(data):
    try:
        cumulative_returns = (1 + data['close'].pct_change()).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = cumulative_returns / rolling_max - 1
        max_drawdown = drawdown.min()
        RISK_METRICS_GAUGE.labels('max_drawdown').set(max_drawdown)
        return max_drawdown
    except Exception as e:
        logger.error(f"Error calculating Maximum Drawdown: {e}")
        return None


# Calculate Beta
def calculate_beta(data, benchmark_data):
    try:
        returns = data['close'].pct_change().dropna()
        benchmark_returns = benchmark_data['close'].pct_change().dropna()
        covariance_matrix = np.cov(returns, benchmark_returns)
        beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
        RISK_METRICS_GAUGE.labels('beta').set(beta)
        return beta
    except Exception as e:
        logger.error(f"Error calculating Beta: {e}")
        return None


# Calculate Information Ratio
def calculate_information_ratio(data, benchmark_data):
    try:
        returns = data['close'].pct_change().dropna()
        benchmark_returns = benchmark_data['close'].pct_change().dropna()
        active_returns = returns - benchmark_returns
        information_ratio = np.mean(active_returns) / np.std(active_returns)
        RISK_METRICS_GAUGE.labels('information_ratio').set(information_ratio)
        return information_ratio
    except Exception as e:
        logger.error(f"Error calculating Information Ratio: {e}")
        return None


# Calculate Omega Ratio
def calculate_omega_ratio(data, threshold=0):
    try:
        returns = data['close'].pct_change().dropna()
        gains = returns[returns > threshold]
        losses = returns[returns <= threshold]
        omega_ratio = gains.sum() / -losses.sum()
        RISK_METRICS_GAUGE.labels('omega_ratio').set(omega_ratio)
        return omega_ratio
    except Exception as e:
        logger.error(f"Error calculating Omega Ratio: {e}")
        return None


# Calculate Ulcer Index
def calculate_ulcer_index(data):
    try:
        cumulative_returns = (1 + data['close'].pct_change()).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
        RISK_METRICS_GAUGE.labels('ulcer_index').set(ulcer_index)
        return ulcer_index
    except Exception as e:
        logger.error(f"Error calculating Ulcer Index: {e}")
        return None


# Calculate Tracking Error
def calculate_tracking_error(data, benchmark_data):
    try:
        returns = data['close'].pct_change().dropna()
        benchmark_returns = benchmark_data['close'].pct_change().dropna()
        tracking_error = np.std(returns - benchmark_returns)
        RISK_METRICS_GAUGE.labels('tracking_error').set(tracking_error)
        return tracking_error
    except Exception as e:
        logger.error(f"Error calculating Tracking Error: {e}")
        return None


# Calculate Jensen's Alpha
def calculate_jensens_alpha(data, benchmark_data, risk_free_rate=0.01):
    try:
        returns = data['close'].pct_change().dropna()
        benchmark_returns = benchmark_data['close'].pct_change().dropna()
        beta = calculate_beta(data, benchmark_data)
        alpha = np.mean(returns) - risk_free_rate / 252 - beta * (np.mean(benchmark_returns) - risk_free_rate / 252)
        RISK_METRICS_GAUGE.labels('jensens_alpha').set(alpha)
        return alpha
    except Exception as e:
        logger.error(f"Error calculating Jensen's Alpha: {e}")
        return None


# Calculate comprehensive risk metrics
def calculate_risk_metrics(data, benchmark_data=None):
    try:
        var = calculate_var(data)
        cvar = calculate_cvar(data)
        sharpe_ratio = calculate_sharpe_ratio(data)
        sortino_ratio = calculate_sortino_ratio(data)
        max_drawdown = calculate_max_drawdown(data)
        beta = calculate_beta(data, benchmark_data) if benchmark_data is not None else None
        information_ratio = calculate_information_ratio(data, benchmark_data) if benchmark_data is not None else None
        omega_ratio = calculate_omega_ratio(data)
        ulcer_index = calculate_ulcer_index(data)
        tracking_error = calculate_tracking_error(data, benchmark_data) if benchmark_data is not None else None
        jensens_alpha = calculate_jensens_alpha(data, benchmark_data) if benchmark_data is not None else None

        risk_metrics = {
            'VaR': var,
            'CVaR': cvar,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Maximum Drawdown': max_drawdown,
            'Beta': beta,
            'Information Ratio': information_ratio,
            'Omega Ratio': omega_ratio,
            'Ulcer Index': ulcer_index,
            'Tracking Error': tracking_error,
            'Jensen’s Alpha': jensens_alpha
        }

        logger.info("Risk metrics calculated successfully.")
        return risk_metrics
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        return {}


# Evaluate basic risk metrics against thresholds
def evaluate_basic_risks(risk_metrics, thresholds):
    try:
        if risk_metrics['Maximum Drawdown'] < thresholds['max_drawdown']:
            logger.warning(
                f"Maximum Drawdown {risk_metrics['Maximum Drawdown']} is below the threshold {thresholds['max_drawdown']}.")
            send_alert("Maximum Drawdown Alert", f"Maximum Drawdown {risk_metrics['Maximum Drawdown']} is below the threshold.")
            return False

        if risk_metrics['Sharpe Ratio'] < thresholds['sharpe_ratio']:
            logger.warning(
                f"Sharpe Ratio {risk_metrics['Sharpe Ratio']} is below the threshold {thresholds['sharpe_ratio']}.")
            send_alert("Sharpe Ratio Alert", f"Sharpe Ratio {risk_metrics['Sharpe Ratio']} is below the threshold.")
            return False

        if risk_metrics['Sortino Ratio'] < thresholds['sortino_ratio']:
            logger.warning(
                f"Sortino Ratio {risk_metrics['Sortino Ratio']} is below the threshold {thresholds['sortino_ratio']}.")
            send_alert("Sortino Ratio Alert", f"Sortino Ratio {risk_metrics['Sortino Ratio']} is below the threshold.")
            return False

        logger.info("Basic risk metrics evaluation passed.")
        return True

    except Exception as e:
        logger.error(f"Error in evaluating basic risk metrics: {e}")
        send_alert("Risk Evaluation Error", "Error occurred during risk evaluation.")
        return False


# Perform correlation analysis
def perform_correlation_analysis(data):
    try:
        correlation_matrix = data.corr()
        logger.info("Correlation analysis performed successfully.")
        return correlation_matrix
    except Exception as e:
        logger.error(f"Error performing correlation analysis: {e}")
        return pd.DataFrame()


# Dynamic stop loss and take profit based on volatility
def dynamic_stop_loss_take_profit(data, stop_loss_multiplier=1.5, take_profit_multiplier=2.0):
    try:
        data['stop_loss'] = data['close'] - stop_loss_multiplier * data['volatility']
        data['take_profit'] = data['close'] + take_profit_multiplier * data['volatility']
        logger.info("Dynamic stop loss and take profit calculated successfully.")
        return data
    except Exception as e:
        logger.error(f"Error calculating dynamic stop loss and take profit: {e}")
        send_alert("Stop Loss/Take Profit Calculation Error", "Error occurred during dynamic stop loss and take profit calculation.")
        return data


# Scenario analysis
def scenario_bear_market(data):
    return data['close'] * 0.8


def scenario_bull_market(data):
    return data['close'] * 1.2


def scenario_high_volatility(data):
    return data['close'] * (1 + np.random.normal(0, 0.1, len(data)))


def scenario_low_volatility(data):
    return data['close'] * (1 + np.random.normal(0, 0.02, len(data)))


def scenario_flash_crash(data):
    data['close'] = data['close'].apply(lambda x: x * 0.7 if np.random.rand() < 0.05 else x)
    return data['close']


def scenario_interest_rate_hike(data):
    return data['close'] * (1 - np.random.normal(0, 0.05, len(data)))


def scenario_currency_devaluation(data):
    return data['close'] * 0.85


def scenario_sector_shock(data):
    return data['close'] * (1 - np.random.normal(0, 0.1, len(data)))


def perform_scenario_analysis(data, scenarios):
    try:
        scenario_results = {}
        for name, scenario in scenarios.items():
            modified_data = scenario(data)
            scenario_results[name] = calculate_risk_metrics(modified_data)
            logger.info(f"Scenario analysis for {name}: {scenario_results[name]}")
        return scenario_results
    except Exception as e:
        logger.error(f"Error performing scenario analysis: {e}")
        send_alert("Scenario Analysis Error", "Error occurred during scenario analysis.")
        return {}


# Generate and save risk report
def generate_risk_report(data, risk_metrics, correlation_matrix):
    try:
        report = {
            'Risk Metrics': risk_metrics,
            'Correlation Matrix': correlation_matrix.to_dict(),
            'Timestamp': datetime.now().isoformat()
        }
        return report
    except Exception as e:
        logger.error(f"Error generating risk report: {e}")
        send_alert("Risk Report Generation Error", "Error occurred while generating risk report.")
        return {}


def save_risk_report(report, report_file):
    try:
        with open(report_file, 'w') as file:
            json.dump(report, file)
        logger.info(f"Risk report saved to {report_file}")
    except Exception as e:
        logger.error(f"Error saving risk report: {e}")
        send_alert("Risk Report Save Error", "Error occurred while saving risk report.")


async def fetch_real_time_data(uri, data_queue, config):
    async for websocket in websockets.connect(uri):
        try:
            while True:
                data = await websocket.recv()
                await data_queue.put(json.loads(data))
                REAL_TIME_DATA_COUNTER.inc()
                LATENCY_GAUGE.set(0)  # Update with actual latency measurement
        except websockets.ConnectionClosed:
            logger.warning("WebSocket connection closed. Reconnecting...")
            await asyncio.sleep(config['risk_management']['websocket']['reconnect_interval'])
        except Exception as e:
            logger.error(f"Error fetching real-time data: {e}")
            send_alert("WebSocket Error", "Error occurred during WebSocket data fetch.")
            await asyncio.sleep(config['risk_management']['websocket']['reconnect_interval'])


async def process_real_time_data(data_queue, config, risk_predictor):
    while True:
        data_batch = []
        while len(data_batch) < config['risk_management']['data_aggregation']['batch_size']:
            data_batch.append(await data_queue.get())
        df = pd.DataFrame(data_batch)
        df = validate_data(df, required_columns=['close'])  # Ensure 'close' column exists
        df = extract_datetime_features(df)
        df = risk_predictor.predict(df)
        save_data_to_db(df, create_engine(config['database']['market_data']), 'real_time_predictions')
        logger.info("Processed real-time data and saved predictions.")
        DATA_AGGREGATION_COUNTER.inc()


def user_alerts(data, alert_thresholds):
    try:
        max_drawdown = data['max_drawdown'].iloc[-1]
        sharpe_ratio = data['sharpe_ratio'].iloc[-1]
        sortino_ratio = data['sortino_ratio'].iloc[-1]

        if max_drawdown < alert_thresholds['max_drawdown']:
            logger.warning(f"Max Drawdown Alert: {max_drawdown}")
            send_alert("Max Drawdown Alert", f"Max Drawdown {max_drawdown} is below the threshold.")

        if sharpe_ratio < alert_thresholds['sharpe_ratio']:
            logger.warning(f"Sharpe Ratio Alert: {sharpe_ratio}")
            send_alert("Sharpe Ratio Alert", f"Sharpe Ratio {sharpe_ratio} is below the threshold.")

        if sortino_ratio < alert_thresholds['sortino_ratio']:
            logger.warning(f"Sortino Ratio Alert: {sortino_ratio}")
            send_alert("Sortino Ratio Alert", f"Sortino Ratio {sortino_ratio} is below the threshold.")

        logger.info("User alerts checked and processed.")
    except Exception as e:
        logger.error(f"Error processing user alerts: {e}")
        send_alert("User Alerts Error", "Error occurred during user alerts processing.")


def main():
    try:
        # Load configuration and setup logging
        config = load_config()
        setup_logging(config)

        # Initialize Prometheus metrics server
        start_http_server(config['prometheus']['port'])

        # Initialize database connection to the market_data database
        market_data_engine = create_engine(config['database']['market_data'])

        # Load the main data from the final_data table within the market_data database
        data = load_data_from_db(market_data_engine, 'final_data')

        # Load benchmark data from the benchmark_data table within the market_data database
        benchmark_data = load_data_from_db(market_data_engine, 'benchmark_data')

        # Train risk prediction model
        risk_predictor = RiskPredictor(data)

        # Perform risk metrics calculations including benchmark data
        risk_metrics = calculate_risk_metrics(data, benchmark_data)
        correlation_matrix = perform_correlation_analysis(data)

        # Evaluate basic risks using thresholds from the config file
        risk_evaluation_passed = evaluate_basic_risks(risk_metrics, config['risk_management']['alert_threshold'])

        if not risk_evaluation_passed:
            logger.warning("Risk evaluation did not pass. Adjusting strategy or alerting user.")
            send_alert("Risk Evaluation Failed", "The strategy failed basic risk evaluation.")
            return  # Exit the script if the strategy does not pass basic risk evaluation

        # Dynamic stop loss and take profit calculations
        data = dynamic_stop_loss_take_profit(data)

        # Scenario analysis
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

        # Generate risk report
        report = generate_risk_report(data, risk_metrics, correlation_matrix)

        # Save risk report
        save_risk_report(report, config['risk_management']['report_file'])

        # Real-time monitoring
        data_queue = asyncio.Queue()
        uri = config['risk_management']['websocket']['uri']
        asyncio.run(fetch_real_time_data(uri, data_queue, config))
        asyncio.run(process_real_time_data(data_queue, config, risk_predictor))

        # User alerts
        user_alerts(data, config['risk_management']['alert_threshold'])

    except Exception as e:
        logger.error(f"Error in risk management process: {e}")
        send_alert("Risk Management Error", "An error occurred in the risk management process.")


if __name__ == "__main__":
    main()

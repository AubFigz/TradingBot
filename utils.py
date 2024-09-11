import os
import logging
from logging.handlers import RotatingFileHandler
import aiohttp
import asyncio
import json
import yaml
import numpy as np
import pandas as pd
import spacy
from spacy.cli import download
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline
from gensim import corpora, models
from tenacity import retry, stop_after_attempt, wait_exponential
from asyncio_throttle import Throttler
from scipy.stats import linregress
from typing import Optional, Dict, Any, Tuple, List, Union, Callable
from sklearn.ensemble import IsolationForest
import websockets
import time
from prometheus_client import start_http_server
from telethon import TelegramClient
from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import nltk
from nltk.stem import PorterStemmer
from telethon import TelegramClient, errors
from decrypt_keys import alchemy_key

# Load configuration
def load_config() -> Dict[str, Any]:
    with open("config.yaml", 'r') as stream:
        return yaml.safe_load(stream)

# Setup logging
def setup_logging(config: Dict[str, Any], level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Create a file handler with log rotation
    handler = RotatingFileHandler(
        filename=config['logging']['file'],
        maxBytes=config['logging'].get('maxBytes', 10485760),  # Default 10MB
        backupCount=config['logging'].get('backupCount', 5)  # Default 5 backup files
    )

    # Set formatter from config
    formatter = logging.Formatter(config['logging']['format'])
    handler.setFormatter(formatter)

    # Add handler to the logger
    logger.addHandler(handler)

    # Optionally add a console handler if needed
    if config['logging'].get('console', False):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

# Ensure spaCy model is installed
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Initialize sentiment analysis models
analyzer = SentimentIntensityAnalyzer()
transformer_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text: str, model_weights: Optional[Dict[str, float]] = None, confidence_method: str = 'std') -> Tuple[Optional[float], List[str], List[Tuple[str, str]], float]:
    try:
        # Function to calculate individual sentiment scores
        def calculate_sentiment():
            return {
                'vader': analyzer.polarity_scores(text)['compound'],
                'textblob': TextBlob(text).sentiment.polarity,
                'transformer': transformer_analyzer(text)[0]
            }

        # Get sentiment scores in parallel
        with ThreadPoolExecutor() as executor:
            future = executor.submit(calculate_sentiment)
            sentiments = future.result()

        transformer_sentiment = sentiments['transformer']['score']
        transformer_sentiment = transformer_sentiment if sentiments['transformer']['label'] == 'POSITIVE' else -transformer_sentiment

        # Set default model weights if none are provided
        if not model_weights:
            model_weights = {
                'vader': 1.0,
                'textblob': 1.0,
                'transformer': 1.0
            }

        # Normalize weights
        total_weight = sum(model_weights.values())
        normalized_weights = {k: v / total_weight for k, v in model_weights.items()}

        # Weighted average for sentiment score
        sentiment_score = np.average(
            [sentiments['vader'], sentiments['textblob'], transformer_sentiment],
            weights=[normalized_weights['vader'], normalized_weights['textblob'], normalized_weights['transformer']]
        )

        # NLP processing for keywords and entities
        doc = nlp(text)
        keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Confidence calculation based on method
        sentiment_scores = [sentiments['vader'], sentiments['textblob'], transformer_sentiment]
        if confidence_method == 'std':
            confidence = 1 - np.std(sentiment_scores)
        elif confidence_method == 'range':
            confidence = 1 - (max(sentiment_scores) - min(sentiment_scores))
        else:
            confidence = 1 - np.std(sentiment_scores)  # Default to standard deviation if method is unknown

        return sentiment_score, keywords, entities, confidence

    except Exception as e:
        logging.error(f"Error processing sentiment: {e}")
        return None, [], [], 0.0

# Download NLTK resources
nltk.download('punkt')

def preprocess_text(text: str, custom_stop_words: Optional[List[str]] = None, replace_stop_words: bool = False,
                    use_stemming: bool = False) -> List[str]:
    """
    Process text by removing stopwords, non-alpha tokens, and performing lemmatization or stemming.

    Parameters:
    - text: The input text to preprocess.
    - custom_stop_words: A list of custom stopwords to add to or replace the default stop words.
    - replace_stop_words: If True, custom stopwords will replace the default stop words.
    - use_stemming: If True, stemming will be used instead of lemmatization.

    Returns:
    - List of processed tokens.
    """
    # Update or replace stop words
    if replace_stop_words:
        stop_words = set(custom_stop_words or [])
    else:
        stop_words = nlp.Defaults.stop_words.copy()
        if custom_stop_words:
            stop_words.update(custom_stop_words)

    doc = nlp(text)

    # Choose between lemmatization and stemming
    if use_stemming:
        stemmer = PorterStemmer()
        processed_tokens = [stemmer.stem(token.text) for token in doc if
                            token.is_alpha and not token.is_stop and token.text.lower() not in stop_words]
    else:
        processed_tokens = [token.lemma_ for token in doc if
                            token.is_alpha and not token.is_stop and token.lemma_ not in stop_words]

    return processed_tokens

# Parallel processing version for large texts
def preprocess_text_parallel(texts: List[str], custom_stop_words: Optional[List[str]] = None,
                             replace_stop_words: bool = False, use_stemming: bool = False) -> List[List[str]]:
    """
    Parallel processing of text preprocessing for a list of texts.

    Parameters:
    - texts: A list of texts to preprocess.
    - custom_stop_words: A list of custom stopwords to add to or replace the default stop words.
    - replace_stop_words: If True, custom stopwords will replace the default stop words.
    - use_stemming: If True, stemming will be used instead of lemmatization.

    Returns:
    - List of processed tokens for each text.
    """
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(preprocess_text, text, custom_stop_words, replace_stop_words, use_stemming) for text
                   in texts]
        return [future.result() for future in futures]

def topic_modeling(texts: List[str], num_topics: int = 5, passes: int = 10,
                    alpha: str = 'auto', eta: str = 'auto',
                    custom_stop_words: Optional[List[str]] = None,
                    num_words: int = 5, n_jobs: int = -1) -> Dict[int, List[str]]:
    """
    Perform topic modeling using LDA with advanced text preprocessing and hyperparameter tuning.
    """
    try:
        # Parallelize text preprocessing for speed
        processed_texts = Parallel(n_jobs=n_jobs)(
            delayed(preprocess_text)(text, custom_stop_words) for text in texts
        )

        # Create dictionary and corpus
        dictionary = corpora.Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]

        # Build LDA model with specified hyperparameters
        lda_model = models.LdaModel(
            corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=passes,
            alpha=alpha,
            eta=eta
        )

        # Extract topics with specified number of words
        topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
        topics_dict = {topic_id: [word for word, _ in words] for topic_id, words in topics}

        return topics_dict

    except Exception as e:
        logging.error(f"Error in topic modeling: {e}")
        return {}

def analyze_on_chain_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        # Initialize the dictionary for metrics
        metrics = {}

        # Transaction Volume metrics
        if 'transactionVolume' in df.columns:
            metrics['totalTransactionVolume'] = df['transactionVolume'].sum()
            metrics['averageTransactionVolume'] = df['transactionVolume'].mean()
            metrics['medianTransactionVolume'] = df['transactionVolume'].median()
            metrics['transactionVolumeStdDev'] = df['transactionVolume'].std()

        # Wallet Activity metrics
        if 'walletActivity' in df.columns:
            metrics['totalWalletActivity'] = df['walletActivity'].sum()
            metrics['averageWalletActivity'] = df['walletActivity'].mean()
            metrics['walletActivityStdDev'] = df['walletActivity'].std()

        # Token Transfer metrics
        if 'tokenTransfers' in df.columns:
            metrics['totalTokenTransfers'] = df['tokenTransfers'].sum()
            metrics['averageTokenTransfers'] = df['tokenTransfers'].mean()

        # Smart Contract Interaction metrics
        if 'smartContractInteractions' in df.columns:
            metrics['totalSmartContractInteractions'] = df['smartContractInteractions'].sum()
            metrics['averageSmartContractInteractions'] = df['smartContractInteractions'].mean()

        # Gas Usage metrics
        if 'gasUsed' in df.columns:
            metrics['totalGasUsed'] = df['gasUsed'].sum()
            metrics['averageGasUsed'] = df['gasUsed'].mean()
            metrics['gasUsedStdDev'] = df['gasUsed'].std()

        # Unique Addresses metrics
        if 'uniqueAddresses' in df.columns:
            metrics['totalUniqueAddresses'] = df['uniqueAddresses'].nunique()
            metrics['averageUniqueAddresses'] = df.groupby('uniqueAddresses')['uniqueAddresses'].count().mean()

        # Trend Analysis: Calculate trends over time for key metrics
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            # Transaction Volume Trend (daily trend)
            if 'transactionVolume' in df.columns:
                daily_transaction_volume = df['transactionVolume'].resample('D').sum()
                metrics['transactionVolumeTrend'] = daily_transaction_volume.pct_change().mean()

            # Wallet Activity Trend (daily trend)
            if 'walletActivity' in df.columns:
                daily_wallet_activity = df['walletActivity'].resample('D').sum()
                metrics['walletActivityTrend'] = daily_wallet_activity.pct_change().mean()

            # Gas Used Trend (daily trend)
            if 'gasUsed' in df.columns:
                daily_gas_used = df['gasUsed'].resample('D').sum()
                metrics['gasUsedTrend'] = daily_gas_used.pct_change().mean()

        logging.info("On-chain metrics analyzed successfully.")
        return metrics

    except KeyError as e:
        logging.error(f"KeyError in analyzing on-chain metrics: {e}")
        return {}
    except Exception as e:
        logging.error(f"Unexpected error in analyzing on-chain metrics: {e}")
        return {}

def detect_anomalies(df: pd.DataFrame, columns: List[str], method: str = 'isolation_forest',
                     contamination: float = 0.05, n_estimators: int = 100, max_samples: str = 'auto',
                     n_neighbors: int = 20) -> pd.DataFrame:
    try:
        if not all(col in df.columns for col in columns):
            missing_cols = [col for col in columns if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[columns])

        if method == 'isolation_forest':
            # Using Isolation Forest for anomaly detection
            model = IsolationForest(contamination=contamination, n_estimators=n_estimators,
                                    max_samples=max_samples, random_state=42)
            df['anomaly_score'] = model.fit_predict(scaled_data)
        elif method == 'local_outlier_factor':
            # Using Local Outlier Factor for anomaly detection
            model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination,
                                       novelty=True)
            df['anomaly_score'] = model.fit_predict(scaled_data)
        else:
            raise ValueError(f"Unsupported anomaly detection method: {method}")

        # Filtering anomalies
        anomalies = df[df['anomaly_score'] == -1]

        # Calculate the severity of the anomalies by using the distance from the decision function
        if hasattr(model, 'decision_function'):
            anomalies['anomaly_severity'] = -model.decision_function(scaled_data[df['anomaly_score'] == -1])

        logging.info(f"Anomalies detected: {len(anomalies)}")

        return anomalies.drop(columns=['anomaly_score'])

    except ValueError as ve:
        logging.error(f"Validation error: {ve}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error detecting anomalies: {e}")
        return pd.DataFrame()

def load_data_from_db(engine, table_name: str, columns: Optional[List[str]] = None,
                      condition: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from a database table with optional filtering and column selection.

    Parameters:
    - engine: The SQLAlchemy engine object for the database connection.
    - table_name: The name of the table to load data from.
    - columns: Optional list of columns to select. If None, all columns are selected.
    - condition: Optional SQL WHERE condition to filter the rows.

    Returns:
    - A pandas DataFrame containing the loaded data.
    """
    try:
        query = f"SELECT {', '.join(columns) if columns else '*'} FROM {table_name}"
        if condition:
            query += f" WHERE {condition}"

        data = pd.read_sql_query(query, engine)
        logging.info(f"Loaded {len(data)} records from '{table_name}' table.")
        return data
    except Exception as e:
        logging.error(f"Error loading data from '{table_name}' table: {e}")
        return pd.DataFrame()

def save_data_to_db(data: pd.DataFrame, engine, table_name: str, if_exists: str = 'append', chunksize: Optional[int] = None, dtype: Optional[Dict[str, Any]] = None) -> None:
    """
    Save a pandas DataFrame to a database table with enhanced options.

    Parameters:
    - data: The pandas DataFrame to save.
    - engine: The SQLAlchemy engine object for the database connection.
    - table_name: The name of the table to save data to.
    - if_exists: How to behave if the table already exists. Options: 'fail', 'replace', 'append'. Default is 'append'.
    - chunksize: Number of rows to write at a time. Defaults to None, writing all rows at once.
    - dtype: Optional dictionary of column types for the SQL table.

    Returns:
    - None
    """
    try:
        data.to_sql(table_name, engine, if_exists=if_exists, index=False, chunksize=chunksize, dtype=dtype)
        logging.info(f"Data saved to database table '{table_name}' successfully with {len(data)} records.")
    except ValueError as ve:
        logging.error(f"ValueError saving data to '{table_name}': {ve}")
    except Exception as e:
        logging.error(f"Unexpected error saving data to '{table_name}': {e}")

def validate_data(
        data: pd.DataFrame,
        required_columns: List[str],
        column_types: Optional[Dict[str, Union[str, type]]] = None,
        allow_empty: bool = False
) -> pd.DataFrame:
    """
    Validate that a DataFrame contains required columns and that columns match expected types.

    Parameters:
    - data: The pandas DataFrame to validate.
    - required_columns: List of columns that must be present in the DataFrame.
    - column_types: Optional dictionary where keys are column names and values are expected types (either as strings or Python types).
    - allow_empty: If False, raises an error if the DataFrame is empty. Default is False.

    Returns:
    - The validated pandas DataFrame.

    Raises:
    - ValueError: If any required column is missing or if the DataFrame is empty when allow_empty is False.
    - TypeError: If any column's data type does not match the expected type.
    """
    try:
        # Check if DataFrame is empty
        if not allow_empty and data.empty:
            raise ValueError("The DataFrame is empty and 'allow_empty' is set to False.")

        # Validate the presence of required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Validate column types if specified
        if column_types:
            for column, expected_type in column_types.items():
                if column in data.columns:
                    actual_dtype = data[column].dtype
                    expected_dtype = np.dtype(expected_type).type if isinstance(expected_type, str) else expected_type
                    if not np.issubdtype(actual_dtype, expected_dtype):
                        raise TypeError(f"Column '{column}' has dtype '{actual_dtype}', expected '{expected_dtype}'")
                else:
                    raise ValueError(f"Column '{column}' specified in 'column_types' is missing from the DataFrame")

        logging.info("Data validation successful.")
        return data

    except (ValueError, TypeError) as ve:
        logging.error(f"Validation error: {ve}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during data validation: {e}")
        raise

def extract_datetime_features(data):
    try:
        data['datetime'] = pd.to_datetime(data['timestamp'])
        data['day_of_week'] = data['datetime'].dt.dayofweek
        data['hour'] = data['datetime'].dt.hour
        data['month'] = data['datetime'].dt.month
        data['day'] = data['datetime'].dt.day
        return data.drop(columns=['datetime'])
    except Exception as e:
        logging.error(f"Error extracting datetime features: {e}")
        return data

async def fetch_data(
        session: aiohttp.ClientSession,
        url: str,
        params: Optional[Dict[str, Any]],
        source: str,
        method: str = 'GET',
        config: Dict[str, Any] = None
) -> Tuple[aiohttp.ClientResponse, Dict[str, Any]]:
    """
    Fetch data asynchronously with rate limiting, retries, and error handling.

    Parameters:
    - session: The aiohttp.ClientSession instance.
    - url: The URL to fetch data from.
    - params: The parameters to send with the request.
    - source: The source name for logging purposes.
    - method: The HTTP method to use (GET or POST). Defaults to 'GET'.
    - config: The configuration dictionary.

    Returns:
    - A tuple containing the response object and the JSON response as a dictionary.
    """
    throttler = Throttler(rate_limit=config['data_fetching']['rate_limit'],
                          period=config['data_fetching']['rate_period'])

    @retry(
        stop=stop_after_attempt(config['data_fetching']['retry_attempts']),
        wait=wait_exponential(multiplier=1, min=config['data_fetching']['retry_delay'], max=10)
    )
    async def fetch() -> Tuple[aiohttp.ClientResponse, Dict[str, Any]]:
        async with throttler:
            try:
                timeout = aiohttp.ClientTimeout(total=config['data_fetching'].get('timeout', 10))
                async with session.request(
                        method,
                        url,
                        json=params if method == 'POST' else None,
                        params=params if method == 'GET' else None,
                        timeout=timeout
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    logging.info(f"{source} - Successfully fetched data: {data}")
                    return response, data
            except aiohttp.ClientResponseError as e:
                logging.error(f"{source} - Client response error ({response.status}): {e}")
                raise
            except aiohttp.ClientConnectionError as e:
                logging.error(f"{source} - Connection error: {e}")
                raise
            except aiohttp.ClientError as e:
                logging.error(f"{source} - General client error: {e}")
                raise
            except Exception as e:
                logging.error(f"{source} - Unexpected error: {e}")
                raise

    return await fetch()

async def fetch_websocket_data(uri: str, source: str, message: Dict[str, Any]) -> None:
    """
    Fetch data from a WebSocket connection, handle errors, and automatically reconnect on failure.

    Parameters:
    - uri: The WebSocket URI.
    - source: The source name for logging purposes.
    - message: The message to send after establishing the WebSocket connection.
    """
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                await websocket.send(json.dumps(message))
                while True:
                    try:
                        response = await websocket.recv()
                        data = json.loads(response)
                        logging.info(f"{source} - Received WebSocket data: {data}")
                    except websockets.ConnectionClosed as e:
                        logging.warning(f"{source} - WebSocket connection closed: {e}. Reconnecting...")
                        break
                    except Exception as e:
                        logging.error(f"{source} - Error processing WebSocket message: {e}")
                        await asyncio.sleep(1)
        except Exception as e:
            logging.error(f"{source} - WebSocket connection error: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

async def fetch_token_mint_address(config: Dict[str, Any]) -> str:
    """
    Fetch the token mint address from a Solana API, with retries on failure.

    Parameters:
    - config: The configuration dictionary containing the API endpoint and fallback mint address.

    Returns:
    - The token mint address as a string.
    """

    @retry(
        stop=stop_after_attempt(config['data_fetching']['retry_attempts']),
        wait=wait_exponential(multiplier=1, min=config['data_fetching']['retry_delay'], max=10)
    )
    async def fetch() -> str:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(config['solana']['mint_address_api']) as response:
                    response.raise_for_status()
                    data = await response.json()
                    mint_address = data.get('token_mint_address', config['solana']['token_mint_address'])
                    logging.info(f"Fetched token mint address: {mint_address}")
                    return mint_address
        except Exception as e:
            logging.error(f"Error fetching token mint address: {e}")
            return config['solana']['token_mint_address']

    return await fetch()

def append_to_csv(data: pd.DataFrame, file_path: str) -> None:
    try:
        file_exists = os.path.isfile(file_path)
        if file_exists:
            existing_df = pd.read_csv(file_path)
            if not existing_df[(existing_df['timestamp'] == data['timestamp']) & (existing_df['price'] == data['price'])].empty:
                logging.info(f"Duplicate data detected, skipping append.")
                return

        data.to_csv(file_path, mode='a', header=not file_exists, index=False)
        logging.info(f"Appended data to {file_path}")
    except Exception as e:
        logging.error(f"Error appending data to CSV: {e}")

def cross_validate(prices: Dict[str, Dict[str, float]], config: Dict[str, Any]) -> Tuple[
    Optional[Dict[str, float]], Dict[str, Dict[str, float]]]:
    try:
        if not prices:
            return None, {}

        validated_prices = {}
        valid_prices = {}

        for key in ['open', 'high', 'low', 'close']:
            price_values = {source: price[key] for source, price in prices.items()}

            avg_price = np.mean(list(price_values.values()))
            volatility = np.std(list(price_values.values()))

            # Dynamically adjust the threshold based on volatility
            threshold = config['price_validation']['base_threshold'] * (1 + volatility)

            # Filter prices that are within the acceptable range defined by the threshold
            valid_prices[key] = {source: price for source, price in price_values.items() if
                                 abs(price - avg_price) <= threshold}

            # Ensure at least a minimum number of valid prices
            if len(valid_prices[key]) < len(price_values) * config['price_validation']['min_valid_fraction']:
                logging.warning(f"Insufficient valid {key} prices: {len(valid_prices[key])} out of {len(price_values)}")
                return None, prices

            # Calculate a weighted average, where weights are inversely related to deviation from the average price
            weights = [1 / (abs(price - avg_price) + 1e-6) for price in valid_prices[key].values()]
            weighted_avg = np.average(list(valid_prices[key].values()), weights=weights)

            validated_prices[key] = weighted_avg

        return validated_prices, valid_prices
    except Exception as e:
        logging.error(f"Error in price cross-validation: {e}")
        return None, prices

def get_alchemy_url(network: str, config: Dict[str, Any]) -> str:
    try:
        # Use the decrypted alchemy_key instead of the encrypted one in config
        base_url = config['alchemy']['base_url']
        return base_url.format(network=network, api_key=alchemy_key)
    except KeyError as e:
        logging.error(f"Error retrieving Alchemy URL: {e}")
        return ""

def analyze_price_sentiment(data: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the relationship between sentiment and price changes, combining them into a single metric.

    Parameters:
    - data: A DataFrame containing 'close', 'open', and 'sentiment' columns.

    Returns:
    - A DataFrame with additional columns for price change, percentage change, filtered sentiment,
      combined sentiment, and a log entry for sentiment-price correlation.
    """
    try:
        # Calculate price change and percentage change
        data['price_change'] = data['close'] - data['open']
        data['price_change_pct'] = data['price_change'] / data['open']

        # Remove entries where 'open' is zero or negative to avoid division errors
        data = data[data['open'] > 0].copy()

        # Filter out neutral sentiments
        sentiment_threshold = 0.1
        data['filtered_sentiment'] = np.where(abs(data['sentiment']) > sentiment_threshold, data['sentiment'], 0)

        # Combine sentiment and price change into a single metric
        sentiment_weight = 0.6
        price_change_weight = 0.4
        data['combined_sentiment'] = (
                sentiment_weight * data['filtered_sentiment'] +
                price_change_weight * data['price_change_pct']
        )

        # Calculate and log the correlation between combined sentiment and price change percentage
        correlation = data['combined_sentiment'].corr(data['price_change_pct'])
        logging.info(f"Sentiment-Price Correlation: {correlation:.4f}")

        return data
    except KeyError as ke:
        logging.error(f"Missing key columns in data: {ke}")
        return data
    except Exception as e:
        logging.error(f"Error in price sentiment analysis: {e}")
        return data

def calculate_vwap(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the Volume-Weighted Average Price (VWAP) and its rolling average.

    Parameters:
    - data: A DataFrame containing 'close' and 'volume' columns.

    Returns:
    - A Series representing the VWAP.
    """
    try:
        data[['close', 'volume']] = data[['close', 'volume']].fillna(method='ffill')
        data.dropna(subset=['close', 'volume'], inplace=True)

        vwap = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        data['rolling_vwap'] = vwap.rolling(window=20).mean()

        logging.info("VWAP and rolling VWAP calculated successfully.")
        return pd.Series(vwap, name='VWAP')
    except Exception as e:
        logging.error(f"Error calculating VWAP: {e}")
        return pd.Series()

def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) and generate signals for overbought/oversold conditions.

    Parameters:
    - data: A DataFrame containing the 'close' column.
    - window: The look-back period for RSI calculation.

    Returns:
    - A Series representing the RSI.
    """
    try:
        delta = data['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
        rsi = 100 - (100 / (1 + rs))

        data['rsi_signal'] = np.select([rsi > 70, rsi < 30], ['overbought', 'oversold'], default='neutral')
        logging.info("RSI calculated and signals detected.")
        return pd.Series(rsi, name='RSI')
    except Exception as e:
        logging.error(f"Error calculating RSI: {e}")
        return pd.Series()

def calculate_moving_averages(data: pd.DataFrame, short_window: int = 20, long_window: int = 50,
                              ema: bool = False) -> pd.DataFrame:
    """
    Calculate short-term and long-term moving averages, with optional use of Exponential Moving Averages (EMA).

    Parameters:
    - data: A DataFrame containing the 'close' column.
    - short_window: The short-term window size.
    - long_window: The long-term window size.
    - ema: If True, calculates EMA instead of SMA.

    Returns:
    - The DataFrame with additional columns for moving averages and cross signals.
    """
    try:
        if ema:
            data['SMA'] = data['close'].ewm(span=short_window, adjust=False).mean()
            data['LMA'] = data['close'].ewm(span=long_window, adjust=False).mean()
        else:
            data['SMA'] = data['close'].rolling(window=short_window).mean()
            data['LMA'] = data['close'].rolling(window=long_window).mean()

        data['cross_signal'] = np.select(
            [data['SMA'] > data['LMA'], data['SMA'] < data['LMA']],
            ['golden_cross', 'death_cross'],
            default='neutral'
        )
        logging.info("Moving averages and cross signals calculated.")
        return data
    except Exception as e:
        logging.error(f"Error calculating moving averages: {e}")
        return data

def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26,
                   signal_period: int = 9) -> pd.DataFrame:
    """
    Calculate the MACD, Signal Line, and MACD Histogram, and generate signals.

    Parameters:
    - data: A DataFrame containing the 'close' column.
    - fast_period: The short-term EMA period.
    - slow_period: The long-term EMA period.
    - signal_period: The signal line EMA period.

    Returns:
    - The DataFrame with additional columns for MACD, Signal Line, MACD Histogram, and MACD signals.
    """
    try:
        data['MACD'] = data['close'].ewm(span=fast_period, adjust=False).mean() - data['close'].ewm(span=slow_period,
                                                                                                    adjust=False).mean()
        data['Signal_Line'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()
        data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']

        data['macd_signal'] = np.where(data['MACD'] > data['Signal_Line'], 'bullish', 'bearish')
        logging.info("MACD and signals calculated.")
        return data
    except Exception as e:
        logging.error(f"Error calculating MACD: {e}")
        return data

def calculate_obv(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the On-Balance Volume (OBV) and apply smoothing.

    Parameters:
    - data: A DataFrame containing 'close' and 'volume' columns.

    Returns:
    - A Series representing the smoothed OBV.
    """
    try:
        obv = (np.sign(data['close'].diff()) * data['volume']).cumsum()
        data['smoothed_obv'] = obv.rolling(window=10).mean()

        logging.info("OBV calculated and smoothed.")
        return pd.Series(data['smoothed_obv'], name='OBV')
    except Exception as e:
        logging.error(f"Error calculating OBV: {e}")
        return pd.Series()

def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate Bollinger Bands and associated signals.

    Parameters:
    - data: A DataFrame containing the 'close' column.
    - window: The rolling window size for calculating Bollinger Bands.

    Returns:
    - The DataFrame with additional columns for Bollinger Bands and band signals.
    """
    try:
        data['Middle_Band'] = data['close'].rolling(window=window).mean()
        data['Upper_Band'] = data['Middle_Band'] + 2 * data['close'].rolling(window=window).std()
        data['Lower_Band'] = data['Middle_Band'] - 2 * data['close'].rolling(window=window).std()

        data['band_signal'] = np.where(
            data['Band_Width'] < data['Band_Width'].rolling(window=window).mean(), 'squeeze',
            np.where(data['close'] > data['Upper_Band'], 'breakout', 'normal')
        )

        logging.info("Bollinger Bands and signals calculated.")
        return data
    except Exception as e:
        logging.error(f"Error calculating Bollinger Bands: {e}")
        return data

def calculate_momentum(data: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Calculate the momentum and identify momentum shifts.

    Parameters:
    - data: A DataFrame containing the 'close' column.
    - window: The window size for calculating momentum.

    Returns:
    - A Series representing the normalized momentum.
    """
    try:
        momentum = data['close'].diff(window)
        normalized_momentum = momentum / momentum.abs().max()
        data['momentum_shift'] = np.where(normalized_momentum > 0, 'up', 'down')

        logging.info("Momentum calculated and normalized.")
        return pd.Series(normalized_momentum, name='Momentum')
    except Exception as e:
        logging.error(f"Error calculating Momentum: {e}")
        return pd.Series()

def calculate_roc(data: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Calculate the Rate of Change (ROC) and associated signals.

    Parameters:
    - data: A DataFrame containing the 'close' column.
    - window: The window size for calculating ROC.

    Returns:
    - A Series representing the smoothed ROC.
    """
    try:
        roc = data['close'].pct_change(periods=window) * 100
        smoothed_roc = roc.rolling(window=3).mean()

        data['roc_signal'] = np.where(smoothed_roc > 5, 'strong', np.where(smoothed_roc < -5, 'weak', 'neutral'))
        logging.info("ROC calculated and signals detected.")
        return pd.Series(smoothed_roc, name='ROC')
    except Exception as e:
        logging.error(f"Error calculating ROC: {e}")
        return pd.Series()

def calculate_sharpe_ratio(data: pd.DataFrame, risk_free_rate: float = 0.01, window: int = 252) -> pd.Series:
    """
    Calculate the Sharpe Ratio and its rolling average.

    Parameters:
    - data: A DataFrame containing the 'close' column.
    - risk_free_rate: The risk-free rate for Sharpe Ratio calculation.
    - window: The rolling window size for calculating Sharpe Ratio.

    Returns:
    - A Series representing the Sharpe Ratio.
    """
    try:
        excess_returns = data['close'].pct_change() - risk_free_rate / window
        sharpe_ratio = np.sqrt(window) * excess_returns.mean() / excess_returns.std()
        data['rolling_sharpe'] = sharpe_ratio.rolling(window=20).mean()

        logging.info("Sharpe Ratio calculated with rolling analysis.")
        return pd.Series(sharpe_ratio, name='Sharpe Ratio')
    except Exception as e:
        logging.error(f"Error calculating Sharpe Ratio: {e}")
        return pd.Series()

def calculate_alpha_beta(data: pd.DataFrame, benchmark_data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate the Alpha and Beta relative to a benchmark.

    Parameters:
    - data: A DataFrame containing the 'close' column for the asset.
    - benchmark_data: A DataFrame containing the 'close' column for the benchmark.

    Returns:
    - A tuple of two Series representing Alpha and Beta, respectively.
    """
    try:
        returns = data['close'].pct_change()
        benchmark_returns = benchmark_data['close'].pct_change()

        covariance_matrix = np.cov(returns[1:], benchmark_returns[1:])
        beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
        alpha = returns.mean() - beta * benchmark_returns.mean()

        data['rolling_alpha'] = alpha.rolling(window=20).mean()
        data['rolling_beta'] = beta.rolling(window=20).mean()

        logging.info("Alpha and Beta calculated with rolling analysis.")
        return pd.Series(alpha, name='Alpha'), pd.Series(beta, name='Beta')
    except Exception as e:
        logging.error(f"Error calculating Alpha and Beta: {e}")
        return pd.Series(), pd.Series()

def setup_prometheus(port: int) -> None:
    """
    Initialize and start the Prometheus metrics server on the specified port.

    Parameters:
    - port: The port on which to start the Prometheus server.
    """
    try:
        start_http_server(port)
        logging.info(f"Prometheus metrics server started on port {port}")
    except Exception as e:
        logging.error(f"Error setting up Prometheus: {e}")
        raise e  # Reraise the exception after logging it to halt the process if Prometheus setup fails

def send_alert(message: str) -> None:
    """
    Send an alert message via Telegram using the TelegramClient.

    Parameters:
    - message: The alert message to be sent.
    """
    try:
        telegram_api_id = os.getenv('TELEGRAM_API_ID')
        telegram_api_hash = os.getenv('TELEGRAM_API_HASH')
        telegram_phone_number = os.getenv('TELEGRAM_PHONE_NUMBER')

        if not telegram_api_id or not telegram_api_hash or not telegram_phone_number:
            raise EnvironmentError("Missing Telegram API credentials")

        with TelegramClient('anon', telegram_api_id, telegram_api_hash) as client:
            client.start(phone=telegram_phone_number)
            client.send_message(telegram_phone_number, message)
            logging.info("Alert sent successfully via Telegram.")
    except errors.TelegramClientError as te:
        logging.error(f"Telegram client error: {te}")
    except Exception as e:
        logging.error(f"Error sending alert via Telegram: {e}")

def adjust_rate_limit(response_headers: dict, rate_limit_tracker: dict) -> None:
    """
    Adjust the rate limit delay based on the response headers received from an API call.

    Parameters:
    - response_headers: The headers from the API response containing rate limit information.
    - rate_limit_tracker: A dictionary tracking the base, minimum, and maximum delay for rate limiting.
    """
    try:
        limit = int(response_headers.get('X-RateLimit-Limit', 0))
        remaining = int(response_headers.get('X-RateLimit-Remaining', 0))
        reset_time = int(response_headers.get('X-RateLimit-Reset', time.time()))

        if remaining <= 0:
            delay = max(0, int(np.floor(reset_time - time.time())))
        else:
            delay = rate_limit_tracker['base_delay'] * (limit / max(remaining, 1))

        rate_limit_tracker['delay'] = min(max(delay, rate_limit_tracker['min_delay']), rate_limit_tracker['max_delay'])

        logging.info(f"Rate limit adjusted: delay set to {rate_limit_tracker['delay']} seconds.")
    except (ValueError, TypeError) as e:
        logging.error(f"Error processing rate limit values: {e}")
        rate_limit_tracker['delay'] = rate_limit_tracker['base_delay']  # Fallback to base delay
    except Exception as e:
        logging.error(f"Error adjusting rate limit: {e}")

def recognize_chart_patterns(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Recognize common chart patterns in the given dataframe.

    Parameters:
    - df: DataFrame containing the 'open', 'high', 'low', and 'close' price data.

    Returns:
    - A dictionary indicating the presence of various chart patterns.
    """
    try:
        patterns = {}
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']

        # Recognize Head and Shoulders pattern
        patterns['head_and_shoulders'] = (close.iloc[-1] < close.iloc[-2] > close.iloc[-3] < close.iloc[-4])

        # Recognize Double Top pattern
        patterns['double_top'] = (close.iloc[-1] < close.iloc[-2] > close.iloc[-3] > close.iloc[-4])

        # Recognize Triple Bottom pattern
        patterns['triple_bottom'] = (close.iloc[-1] > close.iloc[-2] < close.iloc[-3] < close.iloc[-4])

        # Recognize Bullish Wedge pattern
        patterns['bullish_wedge'] = (close.iloc[-1] > close.iloc[-2] > close.iloc[-3] > close.iloc[-4] and
                                     high.iloc[-1] < high.iloc[-2] < high.iloc[-3] < high.iloc[-4])

        # Recognize Bearish Wedge pattern
        patterns['bearish_wedge'] = (close.iloc[-1] < close.iloc[-2] < close.iloc[-3] < close.iloc[-4] and
                                     low.iloc[-1] > low.iloc[-2] > low.iloc[-3] > low.iloc[-4])

        # Recognize Pennant pattern
        patterns['pennant'] = (high.iloc[-1] < high.iloc[-2] < high.iloc[-3] and
                               low.iloc[-1] > low.iloc[-2] > low.iloc[-3])

        # Recognize Flag pattern
        patterns['flag'] = (high.iloc[-1] < high.iloc[-2] and low.iloc[-1] > low.iloc[-2] and
                            close.iloc[-1] > open_price.iloc[-1] > close.iloc[-2] > open_price.iloc[-2])

        logging.info("Chart patterns recognized.")
        return patterns
    except Exception as e:
        logging.error(f"Error recognizing chart patterns: {e}")
        return {}

def recognize_candlestick_patterns(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Recognize common candlestick patterns in the given dataframe.

    Parameters:
    - df: DataFrame containing 'open', 'high', 'low', and 'close' price data.

    Returns:
    - A dictionary indicating the presence of various candlestick patterns.
    """
    try:
        patterns = {}
        open_price = df['open']
        close = df['close']
        high = df['high']
        low = df['low']

        # Recognize Doji pattern
        patterns['doji'] = abs(open_price.iloc[-1] - close.iloc[-1]) <= (high.iloc[-1] - low.iloc[-1]) * 0.1

        # Recognize Engulfing pattern
        patterns['engulfing'] = (open_price.iloc[-1] < close.iloc[-2] < open_price.iloc[-2] < close.iloc[-1] and
                                 open_price.iloc[-1] < close.iloc[-1])

        # Recognize Hammer pattern
        patterns['hammer'] = (close.iloc[-1] > open_price.iloc[-1] and
                              (low.iloc[-1] < open_price.iloc[-1]) and
                              (close.iloc[-1] - open_price.iloc[-1]) >= 2 * (open_price.iloc[-1] - low.iloc[-1]))

        # Recognize Morning Star pattern
        patterns['morning_star'] = (close.iloc[-3] < open_price.iloc[-3] and
                                    close.iloc[-2] < open_price.iloc[-2] and
                                    close.iloc[-1] > open_price.iloc[-1] > (open_price.iloc[-3] + close.iloc[-3]) / 2)

        # Recognize Evening Star pattern
        patterns['evening_star'] = (close.iloc[-3] > open_price.iloc[-3] and
                                    close.iloc[-2] > open_price.iloc[-2] and
                                    close.iloc[-1] < open_price.iloc[-1] < (open_price.iloc[-3] + close.iloc[-3]) / 2)

        # Recognize Dark Cloud Cover pattern
        patterns['dark_cloud_cover'] = (close.iloc[-2] > open_price.iloc[-2] and
                                        open_price.iloc[-1] > high.iloc[-2] and
                                        close.iloc[-1] < open_price.iloc[-2] and
                                        close.iloc[-1] > (open_price.iloc[-2] + close.iloc[-2]) / 2)

        logging.info("Candlestick patterns recognized.")
        return patterns
    except Exception as e:
        logging.error(f"Error recognizing candlestick patterns: {e}")
        return {}

def identify_whale_transactions(df: pd.DataFrame, threshold: Optional[float] = None,
                                dynamic: bool = True) -> pd.DataFrame:
    """
    Identify whale transactions based on transaction volume.

    Parameters:
    - df: DataFrame containing transaction data with a 'transactionVolume' column.
    - threshold: The fixed threshold for identifying whales. If None, dynamic calculation is applied.
    - dynamic: Whether to calculate the threshold dynamically based on the 99th percentile.

    Returns:
    - DataFrame containing whale transactions with additional analysis, including price impact and cluster metrics.
    """
    try:
        if dynamic:
            threshold = df['transactionVolume'].quantile(0.99)
        elif threshold is None:
            threshold = 1_000_000  # Default threshold

        whales = df[df['transactionVolume'] > threshold]

        # Analyzing whale activity and price impact
        whales['price_impact'] = whales['close'].pct_change() * whales['transactionVolume']

        # Grouping whale transactions into clusters
        whale_clusters = whales.groupby(whales.index // 5).agg({
            'transactionVolume': 'sum',
            'price_impact': 'mean'
        }).rename(columns={'transactionVolume': 'cluster_transaction_volume', 'price_impact': 'cluster_price_impact'})

        # Merging cluster data back into the whale transactions DataFrame
        whales = whales.join(whale_clusters, on=(whales.index // 5))

        logging.info(f"Identified {len(whales)} whale transactions with {len(whale_clusters)} clusters.")
        return whales[['transactionVolume', 'price_impact', 'cluster_transaction_volume', 'cluster_price_impact']]
    except Exception as e:
        logging.error(f"Error identifying whale transactions: {e}")
        return pd.DataFrame()

def track_wallet_profitability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Track the profitability of wallets based on transaction activity and price changes.

    Parameters:
    - df: DataFrame containing wallet transaction data.

    Returns:
    - DataFrame with profitability metrics for each wallet.
    """
    try:
        required_columns = ['transactionVolume', 'walletActivity', 'fees', 'price_change']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

        df['profitability'] = df['transactionVolume'] - df['walletActivity'] - df['fees']
        df['profitability_percentage'] = df['profitability'] / df['walletActivity'] * 100

        # Time-based profitability (e.g., weekly)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['weekly_profitability'] = df['profitability'].resample('W').sum()

        # Consider price changes
        df['adjusted_profitability'] = df['profitability'] + df['price_change']

        # Filter significant profitability
        profitable_wallets = df[(df['adjusted_profitability'] > 0) & (df['profitability_percentage'] > 10)]

        logging.info(f"Tracked profitability for {len(profitable_wallets)} wallets.")
        return profitable_wallets[['wallet_id', 'profitability', 'profitability_percentage', 'weekly_profitability',
                                   'adjusted_profitability']]
    except ValueError as ve:
        logging.error(f"Validation error: {ve}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error tracking wallet profitability: {e}")
        return pd.DataFrame()

def compare_cross_chain_metrics(engine, networks):
    try:
        comparison_results = {}

        for network in networks:
            query = f"SELECT * FROM {network}_data"
            df = pd.read_sql(query, engine)

            if df.empty:
                logging.warning(f"No data found for {network}")
                continue

            total_transactions = len(df)
            avg_transaction_value = df['transaction_value'].mean() if 'transaction_value' in df.columns else None
            std_dev_transaction_value = df['transaction_value'].std() if 'transaction_value' in df.columns else None
            avg_gas_price = df['gas_price'].mean() if 'gas_price' in df.columns else None
            total_gas_used = df['gas_used'].sum() if 'gas_used' in df.columns else None
            avg_block_time = df['block_time'].mean() if 'block_time' in df.columns else None
            std_dev_block_time = df['block_time'].std() if 'block_time' in df.columns else None
            avg_hashrate = df['hashrate'].mean() if 'hashrate' in df.columns else None
            avg_difficulty = df['difficulty'].mean() if 'difficulty' in df.columns else None

            # Correlation between transaction value and gas price, if both are available
            if 'transaction_value' in df.columns and 'gas_price' in df.columns:
                correlation_transaction_gas = np.corrcoef(df['transaction_value'], df['gas_price'])[0, 1]
            else:
                correlation_transaction_gas = None

            comparison_results[network] = {
                'total_transactions': total_transactions,
                'avg_transaction_value': avg_transaction_value,
                'std_dev_transaction_value': std_dev_transaction_value,
                'avg_gas_price': avg_gas_price,
                'total_gas_used': total_gas_used,
                'avg_block_time': avg_block_time,
                'std_dev_block_time': std_dev_block_time,
                'avg_hashrate': avg_hashrate,
                'avg_difficulty': avg_difficulty,
                'correlation_transaction_gas': correlation_transaction_gas,
            }

        # Pairwise comparisons
        cross_chain_comparisons = {}
        networks_list = list(networks)
        for i in range(len(networks_list)):
            for j in range(i + 1, len(networks_list)):
                net1 = networks_list[i]
                net2 = networks_list[j]

                avg_trans_diff = abs(comparison_results[net1]['avg_transaction_value'] - comparison_results[net2][
                    'avg_transaction_value']) if comparison_results[net1]['avg_transaction_value'] and \
                                                comparison_results[net2]['avg_transaction_value'] else None
                avg_gas_diff = abs(
                    comparison_results[net1]['avg_gas_price'] - comparison_results[net2]['avg_gas_price']) if \
                comparison_results[net1]['avg_gas_price'] and comparison_results[net2]['avg_gas_price'] else None
                block_time_stability_diff = abs(
                    comparison_results[net1]['std_dev_block_time'] - comparison_results[net2]['std_dev_block_time']) if \
                comparison_results[net1]['std_dev_block_time'] and comparison_results[net2][
                    'std_dev_block_time'] else None

                cross_chain_comparisons[f"{net1}_vs_{net2}"] = {
                    'avg_trans_diff': avg_trans_diff,
                    'avg_gas_diff': avg_gas_diff,
                    'block_time_stability_diff': block_time_stability_diff,
                }

        return {
            'individual_network_metrics': comparison_results,
            'cross_chain_comparisons': cross_chain_comparisons,
        }
    except Exception as e:
        logging.error(f"Error in compare_cross_chain_metrics: {e}")
        return {}

async def fetch_mining_data(session: aiohttp.ClientSession, network: str, config):
    try:
        # Fetch all necessary endpoints for mining data
        hashrate_url = f"{config['mining_data']['api_url']}{config['mining_data']['endpoints']['hashrate']}"
        difficulty_url = f"{config['mining_data']['api_url']}{config['mining_data']['endpoints']['difficulty']}"
        block_reward_url = f"{config['mining_data']['api_url']}{config['mining_data']['endpoints']['block_reward']}"
        orphaned_blocks_url = f"{config['mining_data']['api_url']}/orphaned_blocks"

        # Make asynchronous API calls
        tasks = [
            fetch_data(session, hashrate_url, {}, f"{network} Hashrate API", config=config),
            fetch_data(session, difficulty_url, {}, f"{network} Difficulty API", config=config),
            fetch_data(session, block_reward_url, {}, f"{network} Block Reward API", config=config),
            fetch_data(session, orphaned_blocks_url, {}, f"{network} Orphaned Blocks API", config=config)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle cases where tasks might fail
        hashrate_data, difficulty_data, block_reward_data, orphaned_blocks_data = (result if not isinstance(result, Exception) else {} for result in results)

        if not any([hashrate_data, difficulty_data, block_reward_data, orphaned_blocks_data]):
            logging.warning(f"No mining data returned for {network}")
            return {}

        # Process and combine the data
        mining_data = {
            'network': network,
            'hashrate': hashrate_data.get('hashrate', None),
            'difficulty': difficulty_data.get('difficulty', None),
            'block_reward': block_reward_data.get('block_reward', None),
            'orphaned_blocks': orphaned_blocks_data.get('orphaned_blocks', None),
            'timestamp': pd.to_datetime('now')
        }

        df = pd.DataFrame([mining_data])

        # Validate and analyze the mining data
        required_columns = ['hashrate', 'difficulty', 'block_reward', 'orphaned_blocks', 'timestamp']
        df = validate_data(df, required_columns)

        # Trend analysis: Calculate the trend over time (linear regression)
        if len(df) > 1:
            df['timestamp_ordinal'] = pd.to_datetime(df['timestamp']).map(pd.Timestamp.toordinal)
            slope, intercept, r_value, p_value, std_err = linregress(df['timestamp_ordinal'], df['hashrate'])
            mining_trend = {
                'hashrate_trend_slope': slope,
                'hashrate_r_squared': r_value ** 2,
                'hashrate_p_value': p_value
            }
            df = df.assign(**mining_trend)

        return df.to_dict(orient='records')
    except aiohttp.ClientError as e:
        logging.error(f"Network error fetching mining data for {network}: {e}")
        return {}
    except Exception as e:
        logging.error(f"Unexpected error fetching mining data for {network}: {e}")
        return {}

def parallel_process(
        func: Callable[[Any], Any],
        data: List[Any],
        num_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        use_chunks: bool = False,
        verbose: bool = False
) -> List[pd.DataFrame]:  # Updated return type to be a list of DataFrames
    """
    Executes a given function in parallel across multiple processes.

    Parameters:
    - func: The function to apply to each element in the data.
    - data: A list of data elements or chunks to process.
    - num_workers: The number of worker processes to use. Defaults to the number of CPUs available.
    - chunk_size: The size of each data chunk to process when use_chunks is True.
    - use_chunks: If True, splits the data into chunks before processing. Useful for large datasets.
    - verbose: If True, logs detailed information about task completion.

    Returns:
    - A list of DataFrames from the applied function.
    """

    results = []

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            if use_chunks and chunk_size:
                # Split data into chunks
                chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
                futures = {executor.submit(func, chunk): chunk for chunk in chunks}
            else:
                # Process each item individually
                futures = {executor.submit(func, item): item for item in data}

            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                    if isinstance(result, pd.DataFrame):
                        results.append(result)
                    else:
                        logging.error(f"Task for {item} did not return a DataFrame.")
                    if verbose:
                        logging.info(f"Task completed successfully for: {item}")
                except Exception as e:
                    logging.error(f"Task failed for {item} with exception: {e}")

    except Exception as e:
        logging.error(f"Parallel processing encountered an error: {e}")

    return results


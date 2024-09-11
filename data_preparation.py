import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from pyspark.sql import SparkSession
from utils import load_config, setup_logging, append_to_csv, save_data_to_db, send_alert, setup_prometheus, parallel_process
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from pyod.models.auto_encoder import AutoEncoder
from sklearn.decomposition import PCA
from tenacity import retry, stop_after_attempt, wait_exponential


# Load configuration
config = load_config()

# Setup logging
logger = setup_logging(config)

# Setup Prometheus for monitoring
setup_prometheus(config['prometheus']['port'])

# Initialize database connections
news_engine = create_engine(config['database']['news_data'])
x_engine = create_engine(config['database']['x_data'])
blockchain_engine = create_engine(config['database']['blockchain_data'])
market_engine = create_engine(config['database']['market_data'])
benchmark_engine = create_engine(config['database']['benchmark_data'])
output_engine = create_engine(config['database']['output_data'])

# Initialize Spark session
spark = SparkSession.builder \
    .appName("DataPreparation") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.sql.broadcastTimeout", "600") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# Schema validation function
def schema_validation(df, expected_schema):
    try:
        for column, expected_type in expected_schema.items():
            if column not in df.columns:
                logger.error(f"Column '{column}' is missing from the DataFrame.")
                return False
            if not isinstance(df[column].iloc[0], expected_type):
                logger.error(f"Column '{column}' has incorrect type. Expected: {expected_type}, Found: {type(df[column].iloc[0])}")
                return False
        logger.info("Schema validation passed.")
        return True
    except Exception as e:
        logger.error(f"Error in schema validation: {e}")
        return False

# Function to fetch network configurations
def fetch_networks_from_config(config):
    try:
        networks = list(config['networks'].keys())
        logger.info(f"Fetched networks: {networks}")
        return networks
    except KeyError as e:
        logger.error(f"KeyError in fetching networks: {e}")
        return []
    except Exception as e:
        logger.error(f"Error fetching networks from config: {e}")
        return []

# Enhanced Data Imputation and Augmentation
def impute_and_augment_data(df):
    try:
        # MICE Imputation
        imputer = IterativeImputer(max_iter=10, random_state=0)
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

        # Autoencoder Imputation
        autoencoder = AutoEncoder(hidden_neurons=[64, 32, 64], epochs=100, batch_size=10)
        df[numeric_columns] = autoencoder.fit_transform(df[numeric_columns])

        # Data Augmentation: Polynomial Features
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        augmented_data = poly.fit_transform(df[numeric_columns])
        augmented_df = pd.DataFrame(augmented_data, columns=poly.get_feature_names_out(numeric_columns))
        df = pd.concat([df, augmented_df], axis=1)

        logger.info("Data imputed and augmented successfully.")
        return df
    except Exception as e:
        logger.error(f"Error in data imputation and augmentation: {e}")
        return df

# Advanced Anomaly Detection using Robust PCA and Autoencoders
def detect_anomalies(df):
    try:
        # Robust PCA for anomaly detection
        scaler = StandardScaler()
        pca = PCA(n_components=0.95)
        df_scaled = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))
        pca_result = pca.fit_transform(df_scaled)

        # Detect anomalies using an AutoEncoder
        autoencoder = AutoEncoder(hidden_neurons=[64, 32, 64], epochs=100, batch_size=10)
        predictions = autoencoder.fit_predict(pca_result)
        anomalies = df[predictions == 1]

        logger.info(f"Anomalies detected: {len(anomalies)}")
        return anomalies
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        return pd.DataFrame()

# Data Quality Monitoring with Enhanced Checks
def monitor_data_quality(df):
    try:
        anomalies = detect_anomalies(df)
        if not anomalies.empty:
            send_alert("Data quality issue: anomalies detected.")
            logger.warning("Data quality issue detected: anomalies found.")
            return False

        if df.isnull().sum().sum() > 0:
            send_alert("Data quality issue: missing values found.")
            logger.warning("Data quality issue: missing values found.")
            return False

        if (df.select_dtypes(include=['float64', 'int64']) < 0).sum().sum() > 0:
            send_alert("Data quality issue: negative values found.")
            logger.warning("Data quality issue: negative values found.")
            return False

        logger.info("Data quality checks passed.")
        return True
    except Exception as e:
        logger.error(f"Error in data quality monitoring: {e}")
        return False

# Cache intermediate results in Spark to avoid recomputation
def cache_df(df):
    try:
        df.cache()
        logger.info("DataFrame cached successfully.")
        return df
    except Exception as e:
        logger.error(f"Error caching DataFrame: {e}")
        return df

# Merge all dataframes using advanced techniques
def merge_data(data_frames, keys=None):
    try:
        merged_data = None
        for df in data_frames:
            if merged_data is None:
                merged_data = df
            else:
                if keys:
                    merged_data = merged_data.join(df, on=keys, how='left')
                else:
                    merged_data = merged_data.join(df, on='timestamp', how='left')
        logger.info("Data merged successfully.")
        return merged_data
    except Exception as e:
        logger.error(f"Error merging data: {e}")
        return None

# Process data in chunks with retry mechanism
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=60))
def process_data_in_chunks(engine, table_name, process_func, chunk_size=10000, network_name=None):
    try:
        chunks = []
        for chunk in pd.read_sql_table(table_name, engine, chunksize=chunk_size):
            if network_name:
                processed_chunk = parallel_process(lambda df: process_func(df, network_name), [chunk])
            else:
                processed_chunk = parallel_process(process_func, [chunk])

            for chunk_result in processed_chunk:
                if isinstance(chunk_result, pd.DataFrame) and not chunk_result.empty:
                    chunks.append(chunk_result)
                else:
                    logger.error(f"Processed chunk is not a DataFrame or is empty: {type(chunk_result)}")

        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    except SQLAlchemyError as e:
        logger.error(f"Error reading data in chunks from {table_name}: {e}")
        raise

# Define all process functions for each table
def process_market_data(df):
    try:
        # Impute and augment data
        df = impute_and_augment_data(df)

        # Log data processing completion
        logger.info("Market data processed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error processing market data: {e}")
        return pd.DataFrame()

def process_order_book_data(df):
    try:
        # Impute and augment data
        df = impute_and_augment_data(df)

        # Log data processing completion
        logger.info("Order book data processed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error processing order book data: {e}")
        return pd.DataFrame()

def process_news_data(df):
    try:
        # Impute and augment data
        df = impute_and_augment_data(df)

        # Log data processing completion
        logger.info("News data processed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error processing news data: {e}")
        return pd.DataFrame()

def process_analysis_results_data(df):
    try:
        # Impute and augment data
        df = impute_and_augment_data(df)

        # Log data processing completion
        logger.info("Analysis results data processed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error processing analysis results data: {e}")
        return pd.DataFrame()

def process_x_data(df):
    try:
        # Impute and augment data
        df = impute_and_augment_data(df)

        # Log data processing completion
        logger.info("X data processed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error processing X data: {e}")
        return pd.DataFrame()

def process_network_analysis_data(df):
    try:
        # Impute and augment data
        df = impute_and_augment_data(df)

        # Log data processing completion
        logger.info("Network analysis data processed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error processing network analysis data: {e}")
        return pd.DataFrame()

def process_blockchain_data(df, network_name=None):
    try:
        # Impute and augment data
        df = impute_and_augment_data(df)

        # Log data processing completion
        logger.info(f"Blockchain data for network {network_name} processed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error processing blockchain data for network {network_name}: {e}")
        return pd.DataFrame()

def process_benchmark_data(engine):
    try:
        benchmark_data = pd.read_sql_table('benchmark_data', engine)

        # Impute and augment data
        benchmark_data = impute_and_augment_data(benchmark_data)

        logger.info("Benchmark data processed successfully.")
        return benchmark_data
    except SQLAlchemyError as e:
        logger.error(f"Error processing benchmark data: {e}")
        return pd.DataFrame()

def process_cross_chain_comparisons(engine):
    try:
        cross_chain_data = pd.read_sql_table('cross_chain_comparisons', engine)

        # Impute and augment data
        cross_chain_data = impute_and_augment_data(cross_chain_data)

        logger.info("Cross-chain comparisons data processed successfully.")
        return cross_chain_data
    except SQLAlchemyError as e:
        logger.error(f"Error processing cross-chain comparisons data: {e}")
        return pd.DataFrame()

# Main function to orchestrate the data preparation process
def main():
    logger.info("Starting data preparation")

    try:
        market_data = process_data_in_chunks(market_engine, 'market_data', process_market_data)
        order_book_data = process_data_in_chunks(market_engine, 'order_book_data', process_order_book_data)
        news_data = process_data_in_chunks(news_engine, 'news_data', process_news_data)
        analysis_results_data = process_data_in_chunks(news_engine, 'analysis_results', process_analysis_results_data)
        x_data = process_data_in_chunks(x_engine, 'x_data', process_x_data)
        network_analysis_data = process_data_in_chunks(x_engine, 'network_analysis', process_network_analysis_data)

        # Process blockchain data
        bsc_data = process_data_in_chunks(blockchain_engine, 'bsc_data', process_blockchain_data, network_name='bsc')

        network_list = fetch_networks_from_config(config)
        blockchain_data = pd.concat([
            process_data_in_chunks(blockchain_engine, f'{network}_data', process_blockchain_data, network_name=network)
            for network in network_list
        ], ignore_index=True)

        mining_data = pd.concat([
            process_data_in_chunks(blockchain_engine, f'{network}_mining_data', process_blockchain_data,
                                    network_name=network)
            for network in network_list
        ], ignore_index=True)

        # Process benchmark data
        benchmark_data = process_benchmark_data(benchmark_engine)

        # Process cross-chain comparisons
        cross_chain_comparisons = process_cross_chain_comparisons(blockchain_engine)

    except SQLAlchemyError as e:
        logger.error(f"Error loading data from database: {e}")
        return

    if not any(df.empty for df in [
        market_data, order_book_data, news_data, analysis_results_data, x_data, bsc_data, blockchain_data, mining_data,
        network_analysis_data, benchmark_data, cross_chain_comparisons
    ]):
        data_frames = [
            market_data, order_book_data, news_data, analysis_results_data, x_data, bsc_data, blockchain_data,
            mining_data, network_analysis_data, benchmark_data, cross_chain_comparisons
        ]

        # Merging data with advanced techniques
        merged_data = merge_data(data_frames)

        # Monitor data quality before final save
        if monitor_data_quality(merged_data):
            try:
                merged_data = merged_data.toPandas()  # Convert back to Pandas before saving
                save_data_to_db(merged_data, output_engine, 'final_data')
                append_to_csv(merged_data, config['data_fetching']['combined_data_with_features_file'])
                logger.info("Data preparation complete. Final data saved to database and CSV.")
            except Exception as e:
                logger.error(f"Error saving final data: {e}")
        else:
            logger.error("Data quality issues detected, aborting save operation.")
    else:
        logger.error("One or more data sources are empty")

if __name__ == "__main__":
    main()



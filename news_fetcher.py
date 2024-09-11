import pandas as pd
from sqlalchemy import create_engine
import aiohttp
import asyncio
import feedparser
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
from decrypt_keys import newsapi_key, fred_api_key
from utils import load_config, setup_logging, append_to_csv, save_data_to_db, validate_data, setup_prometheus, \
    send_alert, analyze_sentiment, topic_modeling, adjust_rate_limit
import redis
from prometheus_client import Summary
import websockets
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from statistics import stdev
import time
import requests
from datetime import datetime

# Load configuration
config_data = load_config()

# Setup logging
logger = setup_logging(config_data)

# Monitoring metrics
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')

# Initialize single database connection
engine = create_engine(config_data['database']['news_data'])

# Initialize Redis for distributed caching
redis_client = redis.StrictRedis(host=config_data['redis']['host'], port=config_data['redis']['port'],
                                db=config_data['redis']['db'])

# Setup Prometheus
setup_prometheus(config_data['prometheus']['port'])

# Rate limit tracker
rate_limit_tracker = {
    'base_delay': 1,
    'min_delay': 0.1,
    'max_delay': 10
}

# Enhanced error handling
def retry_on_failure(func, retries=3, delay=2, backoff=2):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"Error executing {func.__name__}: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= backoff
            else:
                logger.error(f"Failed to execute {func.__name__} after {retries} attempts. Error: {e}")
                raise

async def fetch_data_with_retries(session, url, params, cache_key=None, retries=3):
    for attempt in range(retries):
        try:
            async with session.get(url, params=params) as response:
                adjust_rate_limit(response.headers, rate_limit_tracker)
                data = await response.json()
                if cache_key:
                    redis_client.set(cache_key, pd.DataFrame(data).to_json(), ex=3600)
                return data
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"Request failed (attempt {attempt + 1}/{retries}): {e}. Retrying...")
                await asyncio.sleep(2 ** attempt)
            else:
                logger.error(f"Request failed after {retries} attempts. Error: {e}")
                raise

def keyword_filtering(content):
    keywords = config_data['google_trends']['keywords']
    return any(keyword.lower() in content.lower() for keyword in keywords)

# Fetching and processing RSS Feeds
def fetch_rss_feeds():
    rss_feed_urls = config_data['rss_feeds']['urls']
    articles = []
    for url in rss_feed_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            if keyword_filtering(entry.title + entry.summary):
                articles.append({
                    'title': entry.title,
                    'summary': entry.summary,
                    'link': entry.link,
                    'published': entry.published
                })
    return pd.DataFrame(articles)

# Fetch FRED data directly in the script
def fetch_fred_data():
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={config_data['fred']['series_id']}&api_key={fred_api_key}&file_type=json"
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data['observations'])

def get_date_ranges():
    today = datetime.today()
    last_year_start = datetime(today.year - 1, 1, 1)
    last_year_end = datetime(today.year - 1, 12, 31)
    this_year_start = datetime(today.year, 1, 1)
    this_year_end = today

    return {
        "last_year": (last_year_start.strftime('%Y-%m-%d'), last_year_end.strftime('%Y-%m-%d')),
        "this_year": (this_year_start.strftime('%Y-%m-%d'), this_year_end.strftime('%Y-%m-%d'))
    }

# Fetching Historical NewsAPI Data
async def fetch_historical_newsapi_data(start_date, end_date):
    url = config_data['newsapi']['url']
    params = {
        'q': config_data['newsapi']['query'],
        'from': start_date,
        'to': end_date,
        'apiKey': newsapi_key,
        'pageSize': 100,
        'language': 'en',
    }
    data = await fetch_data_with_retries(aiohttp.ClientSession(), url, params)
    filtered_data = [article for article in data.get('articles', []) if keyword_filtering(article['content'])]
    return pd.DataFrame(filtered_data)

# Fetching Historical GDELT Data
async def fetch_historical_gdelt_data(start_date, end_date):
    url = config_data['gdelt_api']['url']
    params = {
        'query': config_data['gdelt_api']['query'],
        'mode': config_data['gdelt_api']['mode'],
        'format': config_data['gdelt_api']['format'],
        'startdatetime': start_date,
        'enddatetime': end_date,
        'maxrecords': config_data['gdelt_api']['maxrecords']
    }
    data = await fetch_data_with_retries(aiohttp.ClientSession(), url, params)
    filtered_data = [item for item in data if keyword_filtering(item['content'])]
    return pd.DataFrame(filtered_data)

# Fetching Historical Google Trends Data
async def fetch_historical_google_trends_data(start_date, end_date):
    url = config_data['google_trends']['url']
    params = {
        'q': config_data['google_trends']['query'],
        'geo': config_data['google_trends']['geo'],
        'hl': config_data['google_trends']['hl'],
        'tz': config_data['google_trends']['tz'],
        'start_date': start_date,
        'end_date': end_date
    }
    data = await fetch_data_with_retries(aiohttp.ClientSession(), url, params)
    return pd.DataFrame(data.get('default', {}).get('trendingSearchesDays', []))

# Fetching Real-Time News Feeds via WebSockets
async def fetch_real_time_news_data():
    uri = config_data['websockets']['uri']
    async with websockets.connect(uri) as websocket:
        data = await websocket.recv()
        filtered_data = [item for item in data if keyword_filtering(item['content'])]
        return pd.DataFrame(filtered_data)

# Processing articles using Spark
def process_articles_with_spark(articles):
    spark = None  # Initialize spark variable

    try:
        spark = SparkSession.builder.appName("NewsProcessing").getOrCreate()

        articles_spark_df = spark.createDataFrame(articles)

        sentiment_udf = udf(lambda text: analyze_sentiment(text)[0], StringType())  # Using the first element of the returned tuple for sentiment score

        articles_spark_df = articles_spark_df.withColumn("sentiment_analysis", sentiment_udf(col("content")))

        articles_df = articles_spark_df.toPandas()
        logger.info("Processed articles with Spark successfully.")
        return articles_df
    except Exception as e:
        logger.error(f"Error processing articles with Spark: {e}")
        return pd.DataFrame()
    finally:
        if spark:
            spark.stop()

def validate_articles(articles_df):
    required_columns = ['title', 'description', 'content', 'sentiment_analysis', 'keywords', 'entities', 'published_at']
    validated_df = validate_data(articles_df, required_columns)
    return validated_df

def parallel_process(func, data, *args):
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(func, item, *args) for item in data]
        for future in as_completed(futures):
            results.append(future.result())
    return results

def perform_topic_modeling(documents, num_topics=10, passes=15, custom_stop_words=None, n_jobs=-1):
    try:
        # Align with the new return type (dictionary)
        topics_dict = topic_modeling(documents, num_topics=num_topics, passes=passes,
                                    custom_stop_words=custom_stop_words, n_jobs=n_jobs)

        # Convert the dictionary format to a list of topics if necessary
        topics_list = []
        for topic_id, words in topics_dict.items():
            topics_list.append({
                'topic': topic_id,
                'words': ', '.join(words),
                'weight': None  # Weight handling can be customized if needed
            })

        return topics_list
    except Exception as e:
        logger.error(f"Error performing topic modeling: {e}")
        return []

def network_analysis(entities):
    G = nx.Graph()
    for entity_list in entities:
        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                G.add_edge(entity_list[i], entity_list[j])
    return dict(nx.degree(G))

def notify_if_high_volatility(volatility):
    message = f"High sentiment volatility detected: {volatility:.2f}"
    logger.warning(message)
    # Sending an alert via Telegram
    send_alert(message)

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()

        # Get dynamic date ranges
        date_ranges = get_date_ranges()

        # Historical Data Fetching for Last Year
        historical_news_task = loop.create_task(
            fetch_historical_newsapi_data(date_ranges['last_year'][0], date_ranges['last_year'][1])
        )
        historical_gdelt_task = loop.create_task(
            fetch_historical_gdelt_data(date_ranges['last_year'][0], date_ranges['last_year'][1])
        )
        historical_google_trends_task = loop.create_task(
            fetch_historical_google_trends_data(date_ranges['last_year'][0], date_ranges['last_year'][1])
        )
        historical_fred_data = fetch_fred_data()

        # Real-Time and Current Data Fetching for This Year
        rss_feed_task = loop.run_in_executor(None, fetch_rss_feeds)
        newsapi_task = loop.create_task(
            fetch_historical_newsapi_data(date_ranges['this_year'][0], date_ranges['this_year'][1])
        )
        gdelt_task = loop.create_task(
            fetch_historical_gdelt_data(date_ranges['this_year'][0], date_ranges['this_year'][1])
        )
        google_trends_task = loop.create_task(
            fetch_historical_google_trends_data(date_ranges['this_year'][0], date_ranges['this_year'][1])
        )
        real_time_news_task = loop.create_task(fetch_real_time_news_data())

        loop.run_until_complete(
            asyncio.gather(
                rss_feed_task,
                newsapi_task,
                gdelt_task,
                google_trends_task,
                real_time_news_task,
                historical_news_task,
                historical_gdelt_task,
                historical_google_trends_task
            )
        )

        # Process fetched data
        rss_feed_results = loop.run_until_complete(rss_feed_task)
        newsapi_results = loop.run_until_complete(newsapi_task)
        gdelt_results = loop.run_until_complete(gdelt_task)
        google_trends_results = loop.run_until_complete(google_trends_task)
        real_time_news_results = loop.run_until_complete(real_time_news_task)

        historical_news_results = loop.run_until_complete(historical_news_task)
        historical_gdelt_results = loop.run_until_complete(historical_gdelt_task)
        historical_google_trends_results = loop.run_until_complete(historical_google_trends_task)

        # Combine all fetched articles
        all_articles = pd.concat([rss_feed_results, newsapi_results, google_trends_results, gdelt_results,
                                real_time_news_results, historical_news_results, historical_gdelt_results,
                                historical_google_trends_results, historical_fred_data])

        # Process articles with Spark
        articles_df = process_articles_with_spark(all_articles)

        if not articles_df.empty:
            validated_articles = validate_articles(articles_df)

            # Ensure 'content' and 'sentiment_analysis' are strings or lists
            validated_articles['content'] = validated_articles['content'].astype(str).fillna('')
            validated_articles['sentiment_analysis'] = validated_articles['sentiment_analysis'].astype(float).fillna(0)

            documents = [doc.split() for doc in validated_articles['content']]
            topics = parallel_process(perform_topic_modeling, documents)
            topic_df = pd.DataFrame(topics, columns=['topic', 'words', 'weight'])

            # Safely apply function on entities
            entities = validated_articles['entities'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
            network_results = network_analysis(entities)
            network_df = pd.DataFrame(network_results.items(), columns=['entity', 'relationship_strength'])

            # Ensure sentiment_analysis is iterable and calculate volatility
            sentiments = [analysis for analysis in validated_articles['sentiment_analysis']]
            sentiment_volatility = stdev(sentiments)
            sentiment_volatility_df = pd.DataFrame([{'sentiment_volatility': sentiment_volatility}])

            # Consolidate and store data
            save_data_to_db(validated_articles, engine, 'news_data')
            save_data_to_db(topic_df, engine, 'analysis_results')
            save_data_to_db(network_df, engine, 'analysis_results')
            save_data_to_db(sentiment_volatility_df, engine, 'analysis_results')

            # The correct call for appending to CSV
            append_to_csv(validated_articles, config_data['data_fetching']['news_data_file'])
            append_to_csv(topic_df, config_data['data_fetching']['news_data_file'])
            append_to_csv(network_df, config_data['data_fetching']['news_data_file'])
            append_to_csv(sentiment_volatility_df, config_data['data_fetching']['news_data_file'])

            if sentiment_volatility > config_data['volatility_threshold']:
                notify_if_high_volatility(sentiment_volatility)

            logger.info("Processed news data saved to database and CSV")
        else:
            logger.warning("No valid articles processed.")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        send_alert("Error in main execution of news fetcher script.")

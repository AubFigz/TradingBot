import pandas as pd
from sqlalchemy import create_engine
import aiohttp
import asyncio
from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from decrypt_keys import bearer_token, telegram_api_id, telegram_api_hash, discord_token, reddit_client_id, reddit_client_secret, reddit_user_agent
from utils import load_config, setup_logging, append_to_csv, fetch_data, save_data_to_db, validate_data, setup_prometheus, send_alert, analyze_sentiment, topic_modeling, adjust_rate_limit
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, window
from pyspark.sql.types import StringType, FloatType
from discord.ext import commands
from sklearn.linear_model import LinearRegression
import praw
import networkx as nx
import redis
import re
from datetime import datetime

# Load configuration
config_data = load_config()

# Setup logging
logger = setup_logging(config_data)

# Initialize Prometheus monitoring with the port specified in the configuration
setup_prometheus(config_data['prometheus']['port'])

# Initialize database connection
engine = create_engine(config_data['database']['x_data'])

# Initialize Redis connection
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Real-time sentiment tracking interval (seconds)
real_time_interval = config_data['data_fetching'].get('fetch_interval', 60)

# Twitter API query parameters
twitter_query = config_data['x']['query']

# Subreddits to monitor
subreddits = config_data['reddit']['subreddits']

# Telegram channels to monitor
telegram_channels = config_data['telegram']['channels']

# Discord channels to monitor
discord_channels = config_data['discord']['channels']

# Initialize Reddit client
reddit = praw.Reddit(
    client_id=reddit_client_id,
    client_secret=reddit_client_secret,
    user_agent=reddit_user_agent
)

# Real-time sentiment tracking
async def real_time_sentiment_tracking():
    while True:
        try:
            # Fetch posts from Twitter, Reddit, Telegram, and Discord
            twitter_posts = await fetch_twitter_posts()
            reddit_posts = await fetch_reddit_posts()
            telegram_posts = await fetch_telegram_posts()
            discord_posts = await fetch_discord_posts()

            # Combine all posts
            all_posts = twitter_posts + reddit_posts + telegram_posts + discord_posts
            all_posts_df = process_with_spark(all_posts)

            # Perform topic modeling
            all_posts_df = perform_topic_modeling(all_posts_df)

            # Perform network analysis
            network_graph = perform_network_analysis(all_posts_df)

            # Load market data
            market_data = pd.read_csv(config_data['data_fetching']['market_data_file'])

            # Quantify influencer impact
            impact_df = quantify_influencer_impact(all_posts_df, market_data)

            # Calculate additional metrics
            metrics_df = calculate_additional_metrics(impact_df)

            # Validate post data
            valid_posts_df = validate_post_data(metrics_df)

            # Save to database
            save_data_to_db(valid_posts_df, engine, 'x_data')

            # Save to CSV
            append_to_csv(valid_posts_df, config_data['data_fetching']['x_sentiment_data_file'])

            # Optionally save network analysis results
            save_network_analysis_to_db(network_graph, engine)

            logger.info("Real-time sentiment metrics updated and saved")
        except Exception as e:
            logger.error(f"Error during real-time sentiment tracking: {e}")
            send_alert(f"Real-time sentiment tracking encountered an error: {e}")

        # Sleep for the specified interval before fetching the next batch of posts
        await asyncio.sleep(real_time_interval)

# Function to fetch posts from Twitter
async def fetch_twitter_posts():
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    params = {
        'query': twitter_query,
        'max_results': 100,
    }

    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            response, data = await fetch_data(session, url, params, "TwitterAPI", config=config_data)
            adjust_rate_limit(response.headers, config_data)
            if isinstance(data, dict):
                return data.get('data', [])
            else:
                logger.error("Unexpected data format received from Twitter API")
                return []
    except Exception as e:
        logger.error(f"Error fetching posts from Twitter: {e}")
        send_alert(f"Error fetching posts from Twitter: {e}")
        return []

# Function to fetch posts from Reddit
async def fetch_reddit_posts():
    reddit_posts = []
    try:
        for subreddit in subreddits:
            subreddit_instance = reddit.subreddit(subreddit)
            for post in subreddit_instance.new(limit=100):
                reddit_posts.append({
                    'text': post.title + ' ' + post.selftext,
                    'created_at': datetime.utcfromtimestamp(post.created_utc),
                    'upvotes': post.score,
                    'comments': post.num_comments,
                    'subreddit': subreddit
                })
        logger.info("Fetched posts from Reddit successfully.")
    except Exception as e:
        logger.error(f"Error fetching posts from Reddit: {e}")
        send_alert(f"Error fetching posts from Reddit: {e}")
    return reddit_posts

# Function to fetch posts from Telegram
async def fetch_telegram_posts():
    async with TelegramClient('anon', telegram_api_id, telegram_api_hash) as client:
        await client.start()
        all_messages = []
        try:
            for channel in telegram_channels:
                entity = await client.get_entity(channel)
                history = await client(GetHistoryRequest(peer=entity, limit=100))
                all_messages.extend(history.messages)
        except Exception as e:
            logger.error(f"Error fetching messages from Telegram: {e}")
            send_alert(f"Error fetching messages from Telegram: {e}")
        return all_messages

# Function to fetch posts from Discord
async def fetch_discord_posts():
    discord_client = commands.Bot(command_prefix='!')
    discord_posts = []

    @discord_client.event
    async def on_ready():
        logger.info(f'Logged in as {discord_client.user}')
        for channel_id in discord_channels:
            channel = discord_client.get_channel(int(channel_id))
            if channel:
                try:
                    async for message in channel.history(limit=100):
                        discord_posts.append({
                            'text': message.content,
                            'created_at': message.created_at,
                            'channel': channel_id,
                            'author': str(message.author)
                        })
                except Exception as e:
                    logger.error(f"Error fetching posts from Discord: {e}")
                    send_alert(f"Error fetching posts from Discord: {e}")

    await discord_client.start(discord_token)
    return discord_posts

# Advanced text cleaning function
def advanced_text_cleaning(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words (e.g., one or two characters)
    text = text.lower().strip()  # Lowercase and strip whitespace
    return text

# Process posts with Apache Spark for scalability
def process_with_spark(posts):
    try:
        spark = SparkSession.builder.appName("SocialMediaProcessing").getOrCreate()
        posts_spark_df = spark.createDataFrame(posts)

        # Apply text cleaning and sentiment analysis from utils
        posts_spark_df = posts_spark_df.withColumn("cleaned_text", udf(advanced_text_cleaning, StringType())(col("text"))) \
                                        .withColumn("sentiment", udf(lambda text: analyze_sentiment(text)[0], FloatType())(col("cleaned_text"))) \
                                        .withColumn("keywords", udf(lambda text: ','.join(analyze_sentiment(text)[1]), StringType())(col("cleaned_text"))) \
                                        .withColumn("entities", udf(lambda text: ','.join([ent[0] for ent in analyze_sentiment(text)[2]]), StringType())(col("cleaned_text")))

        # Calculate time-based and source-based aggregations
        posts_spark_df = posts_spark_df.withColumn("source", udf(lambda text: "twitter" if "twitter.com" in text else "reddit", StringType())(col("text")))

        # Time-based aggregation
        time_window = window("created_at", "1 hour")  # Adjust the window size as needed
        time_agg_df = posts_spark_df.groupBy(time_window, "source").agg(
            {"sentiment": "avg"}).withColumnRenamed("avg(sentiment)", "avg_sentiment")

        posts_df = time_agg_df.toPandas()
        spark.stop()

        logger.info("Data processed with Spark successfully.")
        return posts_df
    except Exception as e:
        logger.error(f"Error processing data with Spark: {e}")
        send_alert(f"Error processing data with Spark: {e}")
        return pd.DataFrame()

# Function to perform topic modeling using utils
def perform_topic_modeling(posts_df, num_topics=5, passes=15, custom_stop_words=None, n_jobs=-1):
    try:
        # Extract texts for topic modeling
        texts = posts_df['cleaned_text'].tolist()

        # Align with the new return type (dictionary)
        topics_dict = topic_modeling(texts, num_topics=num_topics, passes=passes,
                                    custom_stop_words=custom_stop_words, n_jobs=n_jobs)

        # Convert the dictionary format to a string representation suitable for the DataFrame
        topics_list = []
        for topic_id, words in topics_dict.items():
            topic_str = f"Topic {topic_id}: " + ', '.join(words)
            topics_list.append(topic_str)

        # Assign the structured topics to the DataFrame
        posts_df['topics'] = topics_list

        logger.info("Topic modeling performed successfully.")
        return posts_df
    except Exception as e:
        logger.error(f"Error performing topic modeling: {e}")
        send_alert(f"Error performing topic modeling: {e}")
        return posts_df

# Function to perform network analysis
def perform_network_analysis(posts_df):
    try:
        G = nx.Graph()

        for index, row in posts_df.iterrows():
            entities = row['entities'].split(',')
            for entity in entities:
                if entity not in G:
                    G.add_node(entity)
                for related_entity in entities:
                    if entity != related_entity:
                        if not G.has_edge(entity, related_entity):
                            G.add_edge(entity, related_entity)

        centrality = nx.degree_centrality(G)
        nx.set_node_attributes(G, centrality, 'centrality')

        logger.info("Network analysis performed successfully.")
        return G
    except Exception as e:
        logger.error(f"Error performing network analysis: {e}")
        send_alert(f"Error performing network analysis: {e}")
        return None

# Function to save network analysis results to the database
def save_network_analysis_to_db(network_graph, engine):
    try:
        if network_graph is not None:
            centrality_data = [{'entity': node, 'centrality': centrality} for node, centrality in nx.get_node_attributes(network_graph, 'centrality').items()]
            df_centrality = pd.DataFrame(centrality_data)
            save_data_to_db(df_centrality, engine, 'network_analysis')
            logger.info("Network analysis data saved to database successfully.")
    except Exception as e:
        logger.error(f"Error saving network analysis to database: {e}")
        send_alert(f"Error saving network analysis to database: {e}")

# Function to quantify influencer impact
def quantify_influencer_impact(posts_df, market_data):
    try:
        posts_df['created_at'] = pd.to_datetime(posts_df['created_at'])
        market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
        merged_df = pd.merge_asof(posts_df.sort_values('created_at'), market_data.sort_values('timestamp'),
                                left_on='created_at', right_on='timestamp')

        model = LinearRegression()
        X = merged_df[['followers_count', 'retweets', 'likes', 'replies', 'shares', 'views', 'comments', 'mentions']]
        y = merged_df['price']
        model.fit(X, y)
        merged_df['impact_score'] = model.predict(X)

        logger.info("Influencer impact quantified successfully.")
        return merged_df
    except Exception as e:
        logger.error(f"Error quantifying influencer impact: {e}")
        send_alert(f"Error quantifying influencer impact: {e}")
        return posts_df

# Function to calculate additional metrics
def calculate_additional_metrics(posts_df):
    try:
        # Calculate engagement rate
        posts_df['engagement_rate'] = (posts_df['retweets'] + posts_df['likes'] + posts_df['replies'] + posts_df['shares']) / posts_df['followers_count']

        # Calculate sentiment trend and volatility
        posts_df['sentiment_trend'] = posts_df['sentiment'].rolling(window=7).mean()
        posts_df['sentiment_volatility'] = posts_df['sentiment'].rolling(window=7).std()

        logger.info("Additional metrics calculated successfully.")
        return posts_df
    except Exception as e:
        logger.error(f"Error calculating additional metrics: {e}")
        send_alert(f"Error calculating additional metrics: {e}")
        return posts_df

# Function to validate post data
def validate_post_data(data):
    required_columns = ['text', 'sentiment', 'keywords', 'entities', 'created_at', 'followers_count', 'retweets', 'likes', 'replies', 'shares', 'views', 'comments', 'mentions', 'engagement_rate', 'sentiment_trend', 'sentiment_volatility', 'topics']
    return validate_data(data, required_columns)

# Main function with real-time sentiment tracking
if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(real_time_sentiment_tracking())
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        send_alert(f"Main execution encountered an error: {e}")

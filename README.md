About the Project

This project focuses on building a robust, automated cryptocurrency trading system that leverages advanced machine learning techniques and real-time data analysis. The system integrates various sources of data, including price trends, social media sentiment, and blockchain metrics, to make informed trading decisions. By combining modern data-driven approaches, the project aims to optimize profitability and minimize risk in a highly volatile market environment.

Project Overview

The project encompasses multiple stages, starting with data collection from APIs, data preprocessing, and feature engineering. Currently, I am developing the scripts for model training, evaluation, and live trading, indicating that the project is ongoing and the portfolio repository is incomplete at this stage. The system is designed to continuously update itself with new data to refine predictions and strategies. Once trained, the models will be evaluated on unseen data, and the best-performing model will be selected for live trading.

As part of the project, I will have a dedicated trading strategies directory, which will house a collection of pre-defined trading strategies. The live trading script, currently under development, will be responsible for dynamically selecting the most appropriate strategy for each trade. This process will be based on real-time analysis of market conditions, model predictions, and risk assessment metrics. The script will analyze key indicators such as market volatility, momentum, sentiment scores, and blockchain metrics to determine which strategy aligns best with the current market environment.

The project also explores the use of model ensembles to combine strengths from various machine learning algorithms, enhancing prediction accuracy. Additionally, automated risk management strategies and backtesting are integrated, enabling the simulation of trading strategies using historical data before they are deployed in live market conditions.

Workflow:
Currently in Repository

Data Collection:

Collect real-time and historical market data, social media sentiment, and blockchain metrics.
Use APIs such as CoinGecko, Binance, and other relevant financial data sources.
Store data in databases for use in model training and analysis.

Data Preprocessing:

Clean, normalize, and merge the data.
Handle missing values, detect and remove outliers, and augment the dataset with derived metrics.
Create a unified dataset suitable for training machine learning models.

Feature Engineering:

Extract relevant technical indicators, sentiment scores, and blockchain metrics.
Perform feature selection to identify key variables that have the most impact on market prediction.

Backtesting and Risk Management:

Simulate trading strategies on historical data to evaluate performance and risk.
Use metrics like Sharpe Ratio, Max Drawdown, and others to analyze risks.


Overview of Main Files:

market_data.py:

Libraries: pandas, numpy, sqlalchemy, asyncio, aiohttp, websockets, cachetools, logging

Data Sources: Binance, Coinbase Pro, FTX, Bitfinex, Alpaca, Serum DEX, Solana, DexScreener, Dextools, Messari, TradingView, CryptoCompare, AlphaVantage, CoinGecko, Kraken, CoinMarketCap, Nomics, HistoricalData, Benchmark

Calculations: Volume Weighted Average Price (VWAP), Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Bollinger Bands, chart and candlestick patterns (from utils.py), whale transaction identification

Key Processing Steps and Methods/Tools Used: Fetching data via WebSocket and REST APIs, retry logic for failed operations, cross-validating fetched prices, calculating technical indicators, detecting chart and candlestick patterns, saving processed data to databases.

news_fetcher.py:

Libraries: pandas, SQLAlchemy, aiohttp, requests, feedparser, networkX, Concurrent Futures, Redis, Prometheus, Websockets, PySpark, statistics

Data Sources: RSS, FRED API, NewsAPI, GDELT API, Google Trends, WebSocket

Calculations: Sentiment scores, Sentiment Volatility, entity relationships

Key Processing Steps and Methods/Tools Used: Fetching news data, sentiment analysis using NLP, parallel processing, topic modeling, network analysis

blockchain_data.py:

Libraries: pandas, aiohttp, asyncio, SQLAlchemy

Data Sources: Alchemy (ethereum, polygon, shape, zksync, optimism, starknet, arbitrum, arbitrum_nova, astar, zetachain, fantom_opera, mantle, blast, linea, zora, polynomial, base, solana, frax), BSCscan (Binance), MiningPool

Calculations: On-chain metrics, Whale transactions, chart patterns, candlestick patterns, wallet profitability, cross-chain metrics

Key Processing Steps and Methods/Tools Used: Retrieves both real-time and historical data, analyzes on-chain metrics, and performs additional analytics such as whale transaction identification, chart pattern recognition, and cross-chain comparisons. The data is then stored in a structured database and saved to CSV files. 

x_sentiment.py:

Libraries: pandas, PySpark, SQLAlchemy, aiohttp, asyncio, Telethon, praw, discord.ext.commands, networkX, Redis

Data Sources: Twitter API, Reddit API, Telegram API, Discord

Calculations: Sentiment scores, trends, volatility, engagement rates, influencer impact

Key Processing Steps and Methods/Tools Used: Social media data collection, text cleaning, sentiment analysis, real-time sentiment tracking, topic modeling, network analysis.

data_preparation.py:

Libraries: pandas, SQLAlchemy, scikit-learn, PySpark, PyOD, Tenacity 

Data Sources: market_data, benchmark_data, order_book_data, news_data, x_data, blockchain_data

Calculations: Data imputation and augmentation (MICE, AutoEncoder, PolynomialFeatures), anomaly detection (PCA, AutoEncoder)

Key Processing Steps and Methods/Tools Used: Process and integrate datasets from various sources, ensuring they are cleaned, processed, and merged into a single dataset for further analysis. Leverages advanced data processing techniques such as data imputation, anomaly detection, and data augmentation, ensuring that the output dataset is ready for downstream tasks like feature engineering and model training.

feature_engineering.py:

Libraries: pandas, numpy, scikit-learn, featuretools, TA-lib, prometheus, Dask, asyncio

Data Sources: output_data (from data_preparation.py output database)

Calculations: Technical indicators, rolling statistics, risk metrics

Key Processing Steps and Methods/Tools Used: Calculates technical indicators, generates lagged features, computes rolling statistics, uses Featuretools for automated feature engineering, train-test split, performs backtesting and risk management (risk metrics calculations, risk evaluation), feature selection and transformation (RFE, LASSO, Gradient Boosting, polynomial transformations, and log scaling), trains a RandomForest model and logs the importance of each feature to Prometheus for monitoring, performs scenario analysis and applies dynamic stop-loss/take-profit strategies, ensuring the strategy is adaptable to changing market conditions.



Future of Project - Work in Progress - Not in Repository yet

Model Training:

Train multiple models such as LSTM, GRU, Transformer, TCN, and PPO.
Use grid search and hyperparameter tuning to optimize model performance.
Evaluate models based on accuracy, precision, and other performance metrics.

Model Evaluation and Ensemble:

Validate models using unseen data to ensure generalization.
Experiment with ensembles of models to combine their predictive strengths.
Compare individual models and ensembles to select the best-performing approach.

More Backtesting:

Perform rigorous backtesting of the selected model and strategy to ensure robustness under different market conditions.
Use historical data to simulate real trading scenarios and refine strategy parameters.

Trading Strategy Integration:

Implement trading strategies such as day trading, swing trading, and breakout trading based on model predictions.
Select an optimal trading strategy for deployment.

Live Trading Deployment:

After model selection, the best model is deployed to execute live trades via connection to exchanges.
The system continues to monitor the market, updating its predictions and adjusting trades in real time.
Risk management strategies, such as stop-loss and take-profit mechanisms, are integrated.

Performance Monitoring and Updates:

Continuously monitor system performance in live trading, ensuring the model adapts to new market data.
Set up a pipeline for automated retraining of the model with new data to ensure long-term success.

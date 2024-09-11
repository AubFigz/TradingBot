import json
import os
import signal
import logging
import pandas as pd
import aiohttp
import asyncio
import websockets
from datetime import datetime
from cachetools import TTLCache
from sqlalchemy import create_engine
from asyncio_throttle import Throttler
from utils import fetch_data, fetch_token_mint_address, append_to_csv, cross_validate, load_config, setup_logging, \
    save_data_to_db, recognize_chart_patterns, recognize_candlestick_patterns, calculate_vwap, calculate_rsi, \
    calculate_moving_averages, calculate_macd, calculate_obv, calculate_bollinger_bands, calculate_momentum, \
    calculate_roc, calculate_sharpe_ratio, calculate_alpha_beta, setup_prometheus, send_alert, adjust_rate_limit, \
    identify_whale_transactions
from decrypt_keys import alpaca_key, alpaca_secret, alphavantage_key

# Load configuration
config = load_config()

# Configure logging based on environment
logger = setup_logging(config, level=logging.INFO)

# Initialize database connections
engine_market_data = create_engine(config['database']['market_data'])
engine_benchmark_data = create_engine(config['database']['benchmark_data'])
engine_order_book_data = create_engine(config['database']['order_book_data'])

# Cache for API calls
cache = TTLCache(maxsize=100, ttl=60)

# Prometheus setup
setup_prometheus(config['data_fetching']['prometheus_port'])

# Throttler for API rate limiting
throttler = Throttler(rate_limit=10, period=1)

# Monitoring counters
websocket_errors = {}
rest_fallbacks = {}
response_times = {}
success_rates = {}

# Initialize counters for monitoring
def init_monitoring_counters():
    exchanges = ['Binance', 'Coinbase Pro', 'FTX', 'Bitfinex', 'Alpaca', 'Serum DEX', 'Solana',
                 'DexScreener', 'Dextools', 'Messari', 'TradingView', 'CryptoCompare', 'AlphaVantage',
                 'CoinGecko', 'Kraken', 'CoinMarketCap', 'Nomics', 'HistoricalData', 'Benchmark']
    for exchange in exchanges:
        websocket_errors[exchange] = 0
        rest_fallbacks[exchange] = 0
        response_times[exchange] = []
        success_rates[exchange] = []

init_monitoring_counters()

# Fallback to REST API if WebSocket fails
async def fallback_to_rest_api(session, source, url, params, prices):
    start_time = datetime.now()
    try:
        async with throttler:
            response, data = await fetch_data(session, url, params, f"{source}REST")
            adjust_rate_limit(response.headers, throttler)
        response_time = (datetime.now() - start_time).total_seconds()
        response_times[source].append(response_time)
        prices[source] = {
            'open': data['open'],
            'high': data['high'],
            'low': data['low'],
            'close': data['close'],
            'volume': data['volume']
        }
        rest_fallbacks[source] += 1
        logger.info(f"{source} - Fallback to REST API successful (Response time: {response_time} seconds)")
    except Exception as e:
        logger.error(f"{source} - Error in REST API fallback: {e}")
        success_rates[source].append(0)
        send_alert(f"{source} - REST API fallback failed: {e}")
    else:
        success_rates[source].append(1)

# WebSocket connection handlers with enhanced reconnection logic
async def fetch_websocket_data(uri, subscribe_message, source, prices, parse_data, fallback_params=None, headers=None):
    retry_attempts = 0
    max_retries = 5
    backoff_delay = 2
    while retry_attempts < max_retries:
        try:
            async with throttler:
                async with websockets.connect(uri, extra_headers=headers) as websocket:
                    await websocket.send(json.dumps(subscribe_message))

                    # Reset retry_attempts after a successful connection
                    retry_attempts = 0

                    while True:
                        start_time = datetime.now()
                        message = await websocket.recv()
                        response_time = (datetime.now() - start_time).total_seconds()
                        response_times[source].append(response_time)
                        data = json.loads(message)
                        parse_data(data, prices)
                        success_rates[source].append(1)
                        logger.debug(f"{source} - Fetched market data (Response time: {response_time} seconds)")
        except Exception as e:
            websocket_errors[source] += 1
            logger.error(f"{source} WebSocket error: {e}. Retrying...")
            retry_attempts += 1
            await asyncio.sleep(min(backoff_delay ** retry_attempts, throttler._rate_limit.period))
            if retry_attempts == max_retries and fallback_params:
                send_alert(f"{source} WebSocket connection failed after {max_retries} attempts. Falling back to REST API.")
                await fallback_to_rest_api(fallback_params['session'], source, fallback_params['url'], fallback_params['params'], fallback_params['prices'])

# Function to fetch historical data from the HistoricalData API
async def fetch_historical_data(session, prices):
    url = config['api']['historical_data']['url'].format(symbol=config['trading']['symbol'])
    params = {
        'resolution': config['api']['historical_data']['params']['resolution'],
        'start_time': config['trading']['start_time'],
        'end_time': config['trading']['end_time']
    }
    try:
        async with throttler:
            response, data = await fetch_data(session, url, params, 'HistoricalData')
            adjust_rate_limit(response.headers, throttler)
        if data:
            for candle in data['candles']:
                prices['HistoricalData'] = {
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close'],
                    'volume': candle['volume'],
                    'timestamp': candle['timestamp']
                }
            logger.info(f"HistoricalData - Data fetched successfully.")
        else:
            send_alert("HistoricalData - No data returned from the API.")
    except Exception as e:
        logger.error(f"HistoricalData - Error fetching data: {e}")
        send_alert(f"HistoricalData - Error fetching data: {e}")

async def fetch_benchmark_data(session, prices):
    try:
        apikey = alphavantage_key

        url = config['api']['benchmark']['url']
        params = {
            'function': config['api']['benchmark']['params']['function'],
            'symbol': config['benchmark']['symbol'],
            'apikey': apikey
        }

        time_series_key = f"Time Series ({config['api']['benchmark']['params']['interval'].capitalize()})"

        async with throttler:
            response, data = await fetch_data(session, url, params, 'Benchmark')
            adjust_rate_limit(response.headers, throttler)

        if data and time_series_key in data:
            time_series = data[time_series_key]

            benchmark_prices = []
            for timestamp, values in time_series.items():
                benchmark_prices.append({
                    'timestamp': datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': float(values['5. volume'])
                })

            prices['Benchmark'] = pd.DataFrame(benchmark_prices)

            logger.info("Benchmark data fetched and processed successfully.")
        else:
            logger.warning("No benchmark data returned from the API.")
            send_alert("Benchmark - No data returned from the API.")

    except Exception as e:
        logger.error(f"Error fetching benchmark data: {e}")
        send_alert(f"Benchmark - Error fetching data: {e}")

async def fetch_coinbase_pro_data(session, prices):
    uri = config['api']['coinbase_pro']['websocket_url']
    subscribe_message = {
        "type": "subscribe",
        "channels": [{"name": "ticker", "product_ids": [config['trading']['symbol']]}]
    }
    fallback_params = {
        'url': config['api']['coinbase_pro']['url'].format(config['trading']['symbol']),
        'params': {},
        'prices': prices,
        'session': session
    }
    await fetch_websocket_data(uri, subscribe_message, 'Coinbase Pro', prices, parse_coinbase_pro_data, fallback_params)

def parse_coinbase_pro_data(data, prices):
    if 'price' in data:
        prices['Coinbase Pro'] = {
            'open': float(data['open']),
            'high': float(data['high']),
            'low': float(data['low']),
            'close': float(data['price']),
            'volume': float(data['last_size'])
        }

async def fetch_ftx_data(session, prices):
    uri = config['api']['ftx']['websocket_url']
    subscribe_message = {
        "op": "subscribe",
        "channel": "ticker",
        "market": config['trading']['symbol']
    }
    fallback_params = {
        'url': config['api']['ftx']['url'].format(config['trading']['symbol']),
        'params': {'resolution': config['api']['ftx']['params']['resolution']},
        'prices': prices,
        'session': session
    }
    await fetch_websocket_data(uri, subscribe_message, 'FTX', prices, parse_ftx_data, fallback_params)

def parse_ftx_data(data, prices):
    if 'data' in data:
        prices['FTX'] = {
            'open': float(data['data']['open']),
            'high': float(data['data']['high']),
            'low': float(data['data']['low']),
            'close': float(data['data']['last']),
            'volume': float(data['data']['volumeUsd24h'])
        }

async def fetch_bitfinex_data(session, prices):
    uri = config['api']['bitfinex']['websocket_url']
    subscribe_message = {
        "event": "subscribe",
        "channel": "ticker",
        "symbol": config['trading']['symbol']
    }
    fallback_params = {
        'url': config['api']['bitfinex']['url'].format(config['trading']['interval'], config['trading']['symbol']),
        'params': {},
        'prices': prices,
        'session': session
    }
    await fetch_websocket_data(uri, subscribe_message, 'Bitfinex', prices, parse_bitfinex_data, fallback_params)

def parse_bitfinex_data(data, prices):
    if isinstance(data, list) and len(data) > 7:
        prices['Bitfinex'] = {
            'open': float(data[1]),
            'high': float(data[3]),
            'low': float(data[4]),
            'close': float(data[7]),
            'volume': float(data[8])
        }

async def fetch_alpaca_data(session, prices):
    uri = config['api']['alpaca']['websocket_url']
    headers = {
        "APCA-API-KEY-ID": alpaca_key,
        "APCA-API-SECRET-KEY": alpaca_secret
    }
    subscribe_message = {
        "action": "subscribe",
        "bars": [config['trading']['symbol']]
    }
    fallback_params = {
        'url': config['api']['alpaca']['url'].format(config['trading']['symbol']),
        'params': {'timeframe': config['api']['alpaca']['params']['timeframe']},
        'prices': prices,
        'session': session,
        'headers': headers
    }
    await fetch_websocket_data(uri, subscribe_message, 'Alpaca', prices, parse_alpaca_data, fallback_params, headers=headers)

def parse_alpaca_data(data, prices):
    if 'bars' in data:
        prices['Alpaca'] = {
            'open': float(data['bars'][0]['o']),
            'high': float(data['bars'][0]['h']),
            'low': float(data['bars'][0]['l']),
            'close': float(data['bars'][0]['c']),
            'volume': float(data['bars'][0]['v'])
        }

async def fetch_serum_dex_data(session, prices):
    uri = config['api']['serum_dex']['websocket_url']
    subscribe_message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "subscribe",
        "params": [config['api']['serum_dex']['params']['pair']]
    }
    fallback_params = {
        'url': config['api']['serum_dex']['url'],
        'params': {'pair': config['api']['serum_dex']['params']['pair'], 'interval': config['api']['serum_dex']['params']['interval']},
        'prices': prices,
        'session': session
    }
    await fetch_websocket_data(uri, subscribe_message, 'Serum DEX', prices, parse_serum_dex_data, fallback_params)

def parse_serum_dex_data(data, prices):
    if 'result' in data:
        prices['Serum DEX'] = {
            'open': float(data['result']['open']),
            'high': float(data['result']['high']),
            'low': float(data['result']['low']),
            'close': float(data['result']['last']),
            'volume': float(data['result']['volume'])
        }

async def fetch_solana_data(session, prices):
    uri = config['api']['solana_websocket']['url'].format(config['solana']['token_mint_address'])
    subscribe_message = {
        "jsonrpc": "2.0",
        "method": "accountSubscribe",
        "params": [config['solana']['token_mint_address'], {"encoding": "jsonParsed"}],
        "id": 1
    }
    fallback_params = {
        'url': config['api']['solana_json_rpc']['url'],
        'params': {'method': 'getTokenLargestAccounts', 'params': [config['solana']['token_mint_address']], 'id': 1},
        'prices': prices,
        'session': session
    }
    await fetch_websocket_data(uri, subscribe_message, 'Solana', prices, parse_solana_data, fallback_params)

def parse_solana_data(data, prices):
    if 'result' in data:
        prices['Solana'] = {
            'open': float(data['result']['value']['data']['open']),
            'high': float(data['result']['value']['data']['high']),
            'low': float(data['result']['value']['data']['low']),
            'close': float(data['result']['value']['data']['price']),
            'volume': float(data['result']['value']['data']['volume'])
        }

async def fetch_alphavantage_data(session, prices):
    url = config['api']['alphavantage']['url']
    params = {
        'function': config['api']['alphavantage']['params']['function'],
        'symbol': config['trading']['symbol'],
        'outputsize': config['api']['alphavantage']['params']['outputsize'],
        'datatype': config['api']['alphavantage']['params']['datatype'],
        'apikey': alphavantage_key
    }
    try:
        async with throttler:
            response, data = await fetch_data(session, url, params, 'AlphaVantage')
            adjust_rate_limit(response.headers, throttler)
        prices['AlphaVantage'] = {
            'open': float(data['Time Series (Daily)'][0]['1. open']),
            'high': float(data['Time Series (Daily)'][0]['2. high']),
            'low': float(data['Time Series (Daily)'][0]['3. low']),
            'close': float(data['Time Series (Daily)'][0]['4. close']),
            'volume': float(data['Time Series (Daily)'][0]['5. volume'])
        }
    except Exception as e:
        logger.error(f"AlphaVantage - Error fetching data: {e}")
        send_alert(f"AlphaVantage - Error fetching data: {e}")

async def fetch_coingecko_data(session, prices):
    url = config['api']['coingecko']['url'].format(symbol=config['trading']['symbol'])
    params = config['api']['coingecko']['params']
    try:
        async with throttler:
            response, data = await fetch_data(session, url, params, 'CoinGecko')
            adjust_rate_limit(response.headers, throttler)
        prices['CoinGecko'] = {
            'open': float(data['prices'][-1][1]),  # Assuming the latest prices data contains open price
            'high': float(data['high'][1]),  # Adjust the key if needed
            'low': float(data['low'][1]),  # Adjust the key if needed
            'close': float(data['prices'][-1][1]),  # Assuming the latest prices data contains close price
            'volume': data['total_volumes'][-1][1]
        }
    except Exception as e:
        logger.error(f"CoinGecko - Error fetching data: {e}")
        send_alert(f"CoinGecko - Error fetching data: {e}")

async def fetch_kraken_data(session, prices):
    url = config['api']['kraken']['url']
    params = {
        'pair': config['api']['kraken']['params']['pair'].format(symbol=config['trading']['symbol']),
        'interval': config['api']['kraken']['params']['interval']
    }
    try:
        async with throttler:
            response, data = await fetch_data(session, url, params, 'Kraken')
            adjust_rate_limit(response.headers, throttler)
        prices['Kraken'] = {
            'open': float(data['result'][list(data['result'].keys())[0]][-1][1]),
            'high': float(data['result'][list(data['result'].keys())[0]][-1][2]),
            'low': float(data['result'][list(data['result'].keys())[0]][-1][3]),
            'close': float(data['result'][list(data['result'].keys())[0]][-1][4]),
            'volume': float(data['result'][list(data['result'].keys())[0]][-1][6])
        }
    except Exception as e:
        logger.error(f"Kraken - Error fetching data: {e}")
        send_alert(f"Kraken - Error fetching data: {e}")

async def fetch_coinmarketcap_data(session, prices):
    url = config['api']['coinmarketcap']['url']
    params = {
        'symbol': config['trading']['symbol'],
        'convert': 'USD',
        'CMC_PRO_API_KEY': config['api']['coinmarketcap']['key']
    }
    try:
        async with throttler:
            response, data = await fetch_data(session, url, params, 'CoinMarketCap')
            adjust_rate_limit(response.headers, throttler)
        prices['CoinMarketCap'] = {
            'open': float(data['data'][config['trading']['symbol']]['quote']['USD']['open']),
            'high': float(data['data'][config['trading']['symbol']]['quote']['USD']['high']),
            'low': float(data['data'][config['trading']['symbol']]['quote']['USD']['low']),
            'close': float(data['data'][config['trading']['symbol']]['quote']['USD']['price']),
            'volume': data['data'][config['trading']['symbol']]['quote']['USD']['volume_24h']
        }
    except Exception as e:
        logger.error(f"CoinMarketCap - Error fetching data: {e}")
        send_alert(f"CoinMarketCap - Error fetching data: {e}")

async def fetch_nomics_data(session, prices):
    url = config['api']['nomics']['url']
    params = {
        'ids': config['trading']['symbol'],
        'convert': 'USD',
        'key': config['api']['nomics']['key']
    }
    try:
        async with throttler:
            response, data = await fetch_data(session, url, params, 'Nomics')
            adjust_rate_limit(response.headers, throttler)

            if isinstance(data, list):
                prices['Nomics'] = {
                    'open': data[0]['open'],  # Assuming open price data is available
                    'high': data[0]['high'],  # Assuming high price data is available
                    'low': data[0]['low'],  # Assuming low price data is available
                    'close': data[0]['price'],  # Assuming close price data is available
                    'volume': data[0]['1d']['volume']
                }
            else:
                raise ValueError("Unexpected data format received from Nomics API")

    except Exception as e:
        logger.error(f"Nomics - Error fetching data: {e}")
        send_alert(f"Nomics - Error fetching data: {e}")

async def fetch_dexscreener_data(session, prices):
    url = config['api']['dexscreener']['url'].format(pair=config['api']['dexscreener']['pair'])
    params = config['api']['dexscreener']['params']
    try:
        async with throttler:
            response, data = await fetch_data(session, url, params, 'DexScreener')
            adjust_rate_limit(response.headers, throttler)
        prices['DexScreener'] = {
            'open': float(data['pairs'][0]['open']),  # Assuming open price data is available
            'high': float(data['pairs'][0]['high']),  # Assuming high price data is available
            'low': float(data['pairs'][0]['low']),  # Assuming low price data is available
            'close': data['pairs'][0]['priceUsd'],
            'volume': data['pairs'][0]['volumeUsd24h']
        }
    except Exception as e:
        logger.error(f"DexScreener - Error fetching data: {e}")
        send_alert(f"DexScreener - Error fetching data: {e}")

async def fetch_dextools_data(session, prices):
    url = config['api']['dextools']['url']
    params = config['api']['dextools']['params']
    try:
        async with throttler:
            response, data = await fetch_data(session, url, params, 'DexTools')
            adjust_rate_limit(response.headers, throttler)
        prices['DexTools'] = {
            'open': float(data['pairs'][0]['open']),  # Assuming open price data is available
            'high': float(data['pairs'][0]['high']),  # Assuming high price data is available
            'low': float(data['pairs'][0]['low']),  # Assuming low price data is available
            'close': data['pairs'][0]['priceUsd'],
            'volume': data['pairs'][0]['volumeUsd24h']
        }
    except Exception as e:
        logger.error(f"DexTools - Error fetching data: {e}")
        send_alert(f"DexTools - Error fetching data: {e}")

async def fetch_messari_data(session, prices):
    url = config['api']['messari']['url'].format(asset=config['trading']['symbol'])
    params = config['api']['messari']['params']
    try:
        async with throttler:
            response, data = await fetch_data(session, url, params, 'Messari')
            adjust_rate_limit(response.headers, throttler)
        prices['Messari'] = {
            'open': data['data']['market_data']['open'],
            'high': data['data']['market_data']['high'],
            'low': data['data']['market_data']['low'],
            'close': data['data']['market_data']['price_usd'],
            'volume': data['data']['market_data']['volume_last_24_hours']
        }
    except Exception as e:
        logger.error(f"Messari - Error fetching data: {e}")
        send_alert(f"Messari - Error fetching data: {e}")

async def fetch_tradingview_data(session, prices):
    url = config['api']['tradingview']['url'].format(symbol=config['trading']['symbol'])
    params = config['api']['tradingview']['params']
    try:
        async with throttler:
            response, data = await fetch_data(session, url, params, 'TradingView')
            adjust_rate_limit(response.headers, throttler)
        prices['TradingView'] = {
            'open': data['open'],  # Adjust this key to match actual structure
            'high': data['high'],  # Adjust this key to match actual structure
            'low': data['low'],  # Adjust this key to match actual structure
            'close': data['price'],
            'volume': data['volume']
        }
    except Exception as e:
        logger.error(f"TradingView - Error fetching data: {e}")
        send_alert(f"TradingView - Error fetching data: {e}")

async def fetch_cryptocompare_data(session, prices):
    url = config['api']['cryptocompare']['url'].format(symbol=config['trading']['symbol'])
    params = config['api']['cryptocompare']['params']
    try:
        async with throttler:
            response, data = await fetch_data(session, url, params, 'CryptoCompare')
            adjust_rate_limit(response.headers, throttler)
            parse_cryptocompare_data(data, prices)
    except Exception as e:
        logger.error(f"CryptoCompare - Error fetching data: {e}")
        send_alert(f"CryptoCompare - Error fetching data: {e}")

def parse_cryptocompare_data(data, prices):
    try:
        if 'USD' in data:
            prices['CryptoCompare'] = {
                'open': float(data['USD']['open']),  # Assuming open price data is available
                'high': float(data['USD']['high']),  # Assuming high price data is available
                'low': float(data['USD']['low']),  # Assuming low price data is available
                'close': float(data['USD']['price']),
                'volume': float(data['USD']['volume_24h'])
            }
            logger.info("CryptoCompare data parsed successfully.")
        else:
            logger.warning("CryptoCompare data does not contain expected 'USD' field.")
            send_alert("CryptoCompare data does not contain expected 'USD' field.")
    except KeyError as e:
        logger.error(f"KeyError in parsing CryptoCompare data: {e}")
        send_alert(f"CryptoCompare - KeyError: {e}")
    except Exception as e:
        logger.error(f"Error parsing CryptoCompare data: {e}")
        send_alert(f"CryptoCompare - Error parsing data: {e}")

# Binance WebSocket remains as implemented
async def fetch_binance_data(session, prices):
    uri = config['api']['binance_websocket']['url']
    subscribe_message = {
        "method": "SUBSCRIBE",
        "params": [f"{config['trading']['symbol'].lower()}@kline_{config['trading']['interval']}",
                   f"{config['trading']['symbol'].lower()}@depth"],
        "id": 1
    }
    fallback_params = {
        'url': config['api']['binance']['url'],
        'params': {'symbol': config['trading']['symbol'], 'interval': config['trading']['interval']},
        'prices': prices,
        'session': session
    }
    await fetch_websocket_data(uri, subscribe_message, 'Binance', prices, parse_binance_data, fallback_params)

def parse_binance_data(data, prices):
    if 'k' in data:
        kline = data['k']
        prices['Binance'] = {
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v'])
        }

# Calculate technical indicators
def calculate_indicators(df, benchmark_data):
    df['vwap'] = calculate_vwap(df)
    df['rsi'] = calculate_rsi(df)
    df['ma_short'], df['ma_long'] = calculate_moving_averages(df)
    df['macd'], df['signal_line'] = calculate_macd(df)
    df['obv'] = calculate_obv(df)
    df['bollinger_upper'], df['bollinger_lower'] = calculate_bollinger_bands(df)
    df['momentum'] = calculate_momentum(df)
    df['roc'] = calculate_roc(df)
    df['sharpe_ratio'] = calculate_sharpe_ratio(df)
    df['alpha'], df['beta'] = calculate_alpha_beta(df, benchmark_data)
    return df

# Log the monitoring stats periodically
async def log_monitoring_stats():
    monitoring_interval = config['monitoring'].get('interval_seconds', 3600)
    while True:
        logger.info(f"WebSocket errors: {websocket_errors}")
        logger.info(f"REST API fallbacks: {rest_fallbacks}")
        for exchange in response_times:
            if response_times[exchange]:
                avg_response_time = sum(response_times[exchange]) / len(response_times[exchange])
                success_rate = sum(success_rates[exchange]) / len(success_rates[exchange]) * 100
                logger.info(f"{exchange} - Avg response time: {avg_response_time:.2f} seconds, Success rate: {success_rate:.2f}%")
                if success_rate < 80:  # Alert if success rate falls below 80%
                    send_alert(f"{exchange} - Success rate below 80%! Current rate: {success_rate:.2f}%")
        await asyncio.sleep(monitoring_interval)

# Handle graceful shutdown
async def shutdown(signal):
    logger.info(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    list(map(lambda task: task.cancel(), tasks))
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info(f"Shutdown complete. Cancelled {len(tasks)} tasks.")
    send_alert(f"Market data script shut down gracefully after receiving signal: {signal.name}")

# Main function to gather data
async def main():
    retry_attempts = config['data_fetching']['retry_attempts']
    retry_delay = config['data_fetching']['retry_delay']
    attempt = 0

    sol_mint_address = await fetch_token_mint_address(config)
    config['solana']['token_mint_address'] = sol_mint_address

    while attempt < retry_attempts:
        try:
            prices = {}
            order_books = {}
            async with aiohttp.ClientSession() as session:
                tasks = [
                    fetch_binance_data(session, prices),
                    fetch_coinbase_pro_data(session, prices),
                    fetch_ftx_data(session, prices),
                    fetch_bitfinex_data(session, prices),
                    fetch_alpaca_data(session, prices),
                    fetch_serum_dex_data(session, prices),
                    fetch_solana_data(session, prices),
                    fetch_dexscreener_data(session, prices),
                    fetch_dextools_data(session, prices),
                    fetch_messari_data(session, prices),
                    fetch_tradingview_data(session, prices),
                    fetch_cryptocompare_data(session, prices),
                    fetch_alphavantage_data(session, prices),
                    fetch_coingecko_data(session, prices),
                    fetch_kraken_data(session, prices),
                    fetch_coinmarketcap_data(session, prices),
                    fetch_nomics_data(session, prices),
                    fetch_historical_data(session, prices),
                    fetch_benchmark_data(session, prices),
                    log_monitoring_stats()
                ]
                await asyncio.gather(*tasks)

                # Cross-validate prices and save to databases
                validated_price, valid_prices = cross_validate(prices, config)
                if validated_price is not None:
                    price_data = {
                        'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                        'open': validated_price['open'],
                        'high': validated_price['high'],
                        'low': validated_price['low'],
                        'close': validated_price['close'],
                        'volume': validated_price['volume'],
                    }
                    df = pd.DataFrame([price_data])

                    # Calculate technical indicators
                    df = calculate_indicators(df, prices['Benchmark'])

                    # Detect chart and candlestick patterns
                    df['chart_patterns'] = recognize_chart_patterns(df)
                    df['candlestick_patterns'] = recognize_candlestick_patterns(df)

                    # Identify whale transactions
                    df['whale_transactions'] = identify_whale_transactions(df, threshold=1_000_000)

                    append_to_csv(df, config['data_fetching']['market_data_file'])

                    # Save the validated prices, technical indicators, and whale transactions to the databases
                    save_data_to_db(df, engine_market_data, 'market_data')
                    save_data_to_db(prices['Benchmark'], engine_benchmark_data, 'benchmark_data')

                    logger.info(f"Valid prices and technical indicators from sources: {valid_prices}")
                else:
                    logger.warning(f"Inconsistent prices from sources: {prices}")
                    send_alert(f"Inconsistent prices detected from multiple sources. No valid data to save.")

                # Process and save order book data
                if order_books:
                    order_books_data = []
                    for source, order_book in order_books.items():
                        for level in order_book['bids']:
                            order_books_data.append({
                                'source': source,
                                'side': 'bid',
                                'price': level[0],
                                'quantity': level[1],
                                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                            })
                        for level in order_book['asks']:
                            order_books_data.append({
                                'source': source,
                                'side': 'ask',
                                'price': level[0],
                                'quantity': level[1],
                                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                            })

                    order_books_df = pd.DataFrame(order_books_data)
                    append_to_csv(order_books_df, config['data_fetching']['order_book_data_file'])
                    logger.info(f"Order book data from sources: {order_books}")

                    # Save the order book data to the order_book_data database
                    save_data_to_db(order_books_df, engine_order_book_data, 'order_book_data')

            attempt = retry_attempts
        except Exception as e:
            logger.error(f"Error: {e}")
            send_alert(f"Market data script encountered an error: {e}")
            attempt += 1
            logger.info(f"Retrying in {retry_delay} seconds... (Attempt {attempt}/{retry_attempts})")
            await asyncio.sleep(retry_delay)

    logger.error(f"Max retry attempts reached. Exiting...")
    send_alert("Market data script exiting after maximum retry attempts.")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    # Handle exit signals for graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))

    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
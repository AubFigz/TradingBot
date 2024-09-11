import asyncio
import aiohttp
import pandas as pd
from sqlalchemy import create_engine
from utils import fetch_data, fetch_token_mint_address, get_alchemy_url, load_config, \
    setup_logging, append_to_csv, save_data_to_db, analyze_on_chain_metrics, validate_data, \
    recognize_chart_patterns, recognize_candlestick_patterns, identify_whale_transactions, \
    track_wallet_profitability, compare_cross_chain_metrics, fetch_mining_data, fetch_websocket_data
from decrypt_keys import bscscan_key

# Load configuration
config = load_config()

# Configure logging
logger = setup_logging(config)

# Initialize database connection
engine = create_engine(config['database']['blockchain_data'])


async def fetch_alchemy_data(session: aiohttp.ClientSession, network: str, mint_address: str) -> None:
    try:
        url = get_alchemy_url(network, config)
        endpoints = {
            "node_api_url": f"{url}/getBlockNumber",
            "token_api_url": f"{url}/getTokenBalances",
            "gas_api_url": f"{url}/getGasOracle",
            "simulate_tx_url": f"{url}/simulateTransaction",
            "mempool_url": f"{url}/alchemy_mempool"
        }

        tasks = [
            fetch_data(session, endpoints["node_api_url"],
                        {"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1},
                        f"{network} Node API", method='POST', config=config),
            fetch_data(session, endpoints["token_api_url"], {}, f"{network} Token API", method='GET', config=config),
            fetch_data(session, endpoints["gas_api_url"], {}, f"{network} Gas Manager API", method='GET', config=config),
            fetch_data(session, endpoints["simulate_tx_url"], {"to": mint_address, "data": "0x"},
                        f"{network} Transaction Simulation API", method='POST', config=config),
            fetch_data(session, endpoints["mempool_url"], {}, f"{network} Mempool API", method='GET', config=config)
        ]
        results = await asyncio.gather(*tasks)

        # Combine and process fetched data
        data_to_save = [result for result in results if result]
        if data_to_save:
            df = pd.DataFrame(data_to_save)

            # Validate data
            required_columns = ['jsonrpc', 'id', 'result']
            df = validate_data(df, required_columns)

            # Analyze on-chain metrics
            metrics = analyze_on_chain_metrics(df)
            logger.info(f"{network} - On-Chain Metrics: {metrics}")

            # Identify whale transactions
            whale_transactions = identify_whale_transactions(df)
            logger.info(f"{network} - Whale Transactions: {whale_transactions}")

            # Recognize chart and candlestick patterns
            chart_patterns = recognize_chart_patterns(df)
            candlestick_patterns = recognize_candlestick_patterns(df)
            logger.info(f"{network} - Chart Patterns: {chart_patterns}")
            logger.info(f"{network} - Candlestick Patterns: {candlestick_patterns}")

            # Track wallet profitability
            profitable_wallets = track_wallet_profitability(df)
            logger.info(f"{network} - Profitable Wallets: {profitable_wallets}")

            # Save the fetched data to the database
            save_data_to_db(df, engine, f"{network}_data")
            logger.info(f"{network} - Data saved to database")

            # Save to CSV
            append_to_csv(df, config['data_fetching']['blockchain_data_file'])
            logger.info(f"{network} - Data saved to CSV")
    except aiohttp.ClientError as e:
        logger.error(f"Network error fetching {network} data: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching {network} data: {e}")


async def fetch_bscscan_data(session: aiohttp.ClientSession) -> None:
    try:
        bsc_url = f"https://api.bscscan.com/api?module=proxy&action=eth_blockNumber&apikey={bscscan_key}"
        result = await fetch_data(session, bsc_url, {}, "BSC Node API", method='GET', config=config)

        if result:
            df = pd.DataFrame([result])

            # Validate data
            required_columns = ['jsonrpc', 'id', 'result']
            df = validate_data(df, required_columns)

            # Analyze on-chain metrics
            metrics = analyze_on_chain_metrics(df)
            logger.info(f"BSC - On-Chain Metrics: {metrics}")

            # Identify whale transactions
            whale_transactions = identify_whale_transactions(df)
            logger.info(f"BSC - Whale Transactions: {whale_transactions}")

            # Recognize chart and candlestick patterns
            chart_patterns = recognize_chart_patterns(df)
            candlestick_patterns = recognize_candlestick_patterns(df)
            logger.info(f"BSC - Chart Patterns: {chart_patterns}")
            logger.info(f"BSC - Candlestick Patterns: {candlestick_patterns}")

            # Track wallet profitability
            profitable_wallets = track_wallet_profitability(df)
            logger.info(f"BSC - Profitable Wallets: {profitable_wallets}")

            # Save the fetched data to the database
            save_data_to_db(df, engine, "bsc_data")
            logger.info("BSC - Data saved to database")

            # Save to CSV
            append_to_csv(df, config['data_fetching']['blockchain_data_file'])
            logger.info("BSC - Data saved to CSV")
    except aiohttp.ClientError as e:
        logger.error(f"Network error fetching BSC data: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching BSC data: {e}")


async def fetch_mining_data_for_network(session: aiohttp.ClientSession, network: str) -> None:
    try:
        mining_data = await fetch_mining_data(session, network, config)
        if mining_data:
            df = pd.DataFrame(mining_data)

            # Validate and save mining data
            required_columns = ['hashrate', 'difficulty', 'block_reward']
            df = validate_data(df, required_columns)

            # Save mining data to the database
            save_data_to_db(df, engine, f"{network}_mining_data")
            logger.info(f"{network} - Mining data saved to database")

            # Save to CSV
            append_to_csv(df, config['data_fetching']['blockchain_data_file'])
            logger.info(f"{network} - Mining data saved to CSV")
    except aiohttp.ClientError as e:
        logger.error(f"Network error fetching mining data for {network}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching mining data for {network}: {e}")


async def fetch_all_data() -> None:
    async with aiohttp.ClientSession() as session:
        try:
            mint_address = await fetch_token_mint_address(config)
            networks = config['networks']

            tasks = [fetch_alchemy_data(session, network, mint_address) for network in networks.keys()]

            # Fetch BSC data
            tasks.append(fetch_bscscan_data(session))

            # Fetch mining data for each network
            mining_tasks = [fetch_mining_data_for_network(session, network) for network in networks.keys()]
            tasks.extend(mining_tasks)

            await asyncio.gather(*tasks)

            # Perform cross-chain analytics and comparisons
            cross_chain_comparisons = compare_cross_chain_metrics(engine, networks.keys())
            logger.info(f"Cross-Chain Analytics: {cross_chain_comparisons}")

            # Save cross-chain comparisons to database
            cross_chain_df = pd.DataFrame([cross_chain_comparisons])
            save_data_to_db(cross_chain_df, engine, "cross_chain_comparisons")

            # Save cross-chain comparisons to CSV
            append_to_csv(cross_chain_df, config['data_fetching']['blockchain_data_file'])
            logger.info("Cross-chain comparison results saved to CSV")
        except Exception as e:
            logger.error(f"Error in fetch_all_data: {e}")


async def fetch_real_time_data_via_websocket() -> None:
    networks = config['networks']
    for network in networks.keys():
        websocket_uri = config['websockets'][network]['uri']
        subscribe_message = config['websockets'][network]['subscribe_message']

        await fetch_websocket_data(websocket_uri, f"{network} WebSocket", subscribe_message)


async def main() -> None:
    while True:
        await fetch_all_data()
        await fetch_real_time_data_via_websocket()
        await asyncio.sleep(config['data_fetching']['fetch_interval'])


if __name__ == "__main__":
    asyncio.run(main())

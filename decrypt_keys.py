from cryptography.fernet import Fernet
import yaml
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, filename='decrypt_keys.log', format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
try:
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    logger.info("Configuration loaded successfully.")
except FileNotFoundError:
    logger.error("config.yaml file not found.")
    raise
except yaml.YAMLError as e:
    logger.error(f"Error parsing config.yaml: {e}")
    raise

# Read decryption key from secret.key file
try:
    with open('secret.key', 'rb') as key_file:
        decryption_key = key_file.read()
    logger.info("Decryption key loaded successfully.")
except FileNotFoundError:
    logger.error("secret.key file not found.")
    raise

# Decrypt the keys
def decrypt_key(encrypted_key, decryption_key):
    try:
        fernet = Fernet(decryption_key)
        return fernet.decrypt(encrypted_key.encode()).decode()
    except Exception as e:
        logger.error(f"Error decrypting key: {e}")
        raise

# Function to decrypt all keys specified in the configuration
def decrypt_all_keys(config, decryption_key):
    decrypted_keys = {}
    try:
        for section, keys in config.items():
            if isinstance(keys, dict):
                for key_name, encrypted_key in keys.items():
                    if 'encrypted_' in key_name:
                        decrypted_key = decrypt_key(encrypted_key, decryption_key)
                        key_name = key_name.replace('encrypted_', '')
                        decrypted_keys[f"{section.upper()}_{key_name.upper()}"] = decrypted_key
        logger.info("All keys decrypted successfully.")
    except Exception as e:
        logger.error(f"Error decrypting keys: {e}")
        raise
    return decrypted_keys

# Decrypt all keys
decrypted_keys = decrypt_all_keys(config, decryption_key)

# Set environment variables
for key, value in decrypted_keys.items():
    os.environ[key] = value

# Making decrypted keys accessible to other modules
solana_private_key = os.getenv('SOLANA_PRIVATE_KEY')
solana_public_key = os.getenv('SOLANA_PUBLIC_KEY')
newsapi_key = os.getenv('NEWSAPI_KEY')
bearer_token = os.getenv('X_BEARER_TOKEN')
bscscan_key = os.getenv('BSCSCAN_KEY')
alchemy_key = os.getenv('ALCHEMY_KEY')
alpaca_key = os.getenv('ALPACA_KEY')
alpaca_secret = os.getenv('ALPACA_SECRET')
telegram_api_id = os.getenv('TELEGRAM_API_ID')
telegram_api_hash = os.getenv('TELEGRAM_API_HASH')
telegram_phone_number = os.getenv('TELEGRAM_PHONE_NUMBER')
discord_token = os.getenv('DISCORD_TOKEN')
fred_api_key = os.getenv('ECONOMIC_DATA_FRED_KEY')
alphavantage_key = os.getenv('API_ALPHAVANTAGE_KEY')
reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
reddit_user_agent = os.getenv('REDDIT_USER_AGENT')

if __name__ == "__main__":
    logger.info("Decryption script executed successfully.")

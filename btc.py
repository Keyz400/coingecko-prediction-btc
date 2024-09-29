import os
import logging
import requests
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

# Configurations
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7066257336:AAHiASvtYMLHHTldyiFMVfOeAfBLRSudDhY")  # Telegram bot token
USER_CHAT_ID = os.getenv("USER_CHAT_ID", "-1001824360922")  # Chat ID to send notifications to

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to fetch the current price of BTC
def fetch_btc_price():
    response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd")
    data = response.json()
    return data['bitcoin']['usd']

# Notify about current price
def notify_price_change(context: CallbackContext):
    current_price = fetch_btc_price()
    logger.info(f"Current BTC price: {current_price}")

    # Send message to the specified chat ID
    context.bot.send_message(chat_id=USER_CHAT_ID, text=f"Current BTC price: ${current_price}")

# Main function to start the bot
def main():
    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    job_queue = updater.job_queue

    # Schedule a job to send price updates every minute
    job_queue.run_repeating(notify_price_change, interval=60, first=0)

    updater.start_polling()
    logger.info("Bot started. Polling...")
    updater.idle()

if __name__ == "__main__":
    main()

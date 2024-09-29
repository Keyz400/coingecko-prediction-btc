import logging
import os
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import requests
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

# Configurations
ADMIN_IDS = [int(admin_id) for admin_id in os.getenv("1318663278").split(',')]  # Admin IDs from environment variables
CHAT_ID = os.getenv("-1001824360922")  # Chat ID where the bot sends notifications
TELEGRAM_BOT_TOKEN = os.getenv("7066257336:AAHiASvtYMLHHTldyiFMVfOeAfBLRSudDhY")  # Telegram bot token from environment variables
BTC_THRESHOLD = float(os.getenv("BTC_THRESHOLD", -2))  # Default percentage threshold for BTC drop
TIMEZONE_IST = pytz.timezone('Asia/Kolkata')  # Indian Standard Time (IST)

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Fetch the current price of Bitcoin
def fetch_current_price():
    url = 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd'
    response = requests.get(url)
    data = response.json()
    return data['bitcoin']['usd']

# Fetch the percentage change in Bitcoin price
def fetch_price_change_percentage():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1'
    response = requests.get(url)
    data = response.json()
    prices = [price[1] for price in data['prices']]
    if prices:
        initial_price = prices[0]
        latest_price = prices[-1]
        percentage_change = ((latest_price - initial_price) / initial_price) * 100
        return percentage_change
    return 0

# Notify if price drops below the threshold
def notify_if_price_drops(context):
    try:
        current_price = fetch_current_price()
        price_change_percentage = fetch_price_change_percentage()

        if price_change_percentage <= BTC_THRESHOLD:
            logging.info(f"Price drop detected: {price_change_percentage}%")
            context.bot.send_message(CHAT_ID, f"⚠️ BTC price dropped by {price_change_percentage:.2f}% to {current_price} USD!")
        else:
            logging.info(f"No significant drop. Current BTC price change: {price_change_percentage:.2f}%")
    except Exception as e:
        logging.error(f"Error in notify_if_price_drops: {e}")

# Command to get the current price of BTC
def current_price(update, context):
    current_price = fetch_current_price()
    update.message.reply_text(f"💰 Current Bitcoin price is {current_price} USD")

# Command for price drop prediction (based on 1-day price change)
def price_drop_prediction(update, context):
    try:
        price_change_percentage = fetch_price_change_percentage()
        update.message.reply_text(f"📊 Bitcoin price has changed by {price_change_percentage:.2f}% in the last 24 hours.")
    except Exception as e:
        logging.error(f"Error in price_drop_prediction: {e}")
        update.message.reply_text("An error occurred while fetching the price change data.")

# Command for the admin to set the price drop threshold
def set_threshold(update, context):
    user_id = update.message.from_user.id
    if user_id in ADMIN_IDS:
        try:
            new_threshold = float(context.args[0])
            global BTC_THRESHOLD
            BTC_THRESHOLD = new_threshold
            update.message.reply_text(f"✅ Price drop threshold updated to {BTC_THRESHOLD}%")
        except (IndexError, ValueError):
            update.message.reply_text("❌ Please provide a valid number for the threshold.")
    else:
        update.message.reply_text("❌ You are not authorized to change the threshold.")

# Error handler for unknown commands
def unknown_command(update, context):
    update.message.reply_text("⚠️ Sorry, I didn't understand that command. Use /pd to predict BTC price or /cp to get the current BTC price.")

# Check BTC price drop periodically
def check_btc_price_drop():
    scheduler = BackgroundScheduler(timezone=TIMEZONE_IST)
    scheduler.add_job(notify_if_price_drops, 'interval', minutes=15)
    scheduler.start()
    logging.info("Scheduler for BTC price drop monitoring started.")

def main():
    # Initialize the bot with the token
    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    # Add command handlers
    dp.add_handler(CommandHandler("cp", current_price))
    dp.add_handler(CommandHandler("pd", price_drop_prediction))
    dp.add_handler(CommandHandler("set_threshold", set_threshold, pass_args=True))

    # Handle unknown commands
    dp.add_handler(MessageHandler(Filters.command, unknown_command))

    # Start the bot and scheduler
    check_btc_price_drop()
    logging.info("Started BTC price drop monitoring.")
    
    # Start the bot
    updater.start_polling()
    logging.info("Bot started.")
    
    # Run the bot until you send a signal to stop it
    updater.idle()

if __name__ == '__main__':
    main()

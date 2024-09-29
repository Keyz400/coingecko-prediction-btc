import os
import logging
import requests
import sqlite3
import pytz
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Configurations
ADMIN_IDS = [int(admin_id) for admin_id in os.getenv("ADMIN_IDS", "1318663278").split(',')]  # Admin IDs from environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7066257336:AAHiASvtYMLHHTldyiFMVfOeAfBLRSudDhY")  # Telegram bot token
BTC_THRESHOLD = float(os.getenv("BTC_THRESHOLD", -0.2))  # Default percentage threshold for BTC drop
TIMEZONE_IST = pytz.timezone('Asia/Kolkata')  # Indian Standard Time (IST)
DB_NAME = "user_settings.db"

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
def create_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id INTEGER PRIMARY KEY,
            threshold REAL DEFAULT -0.2,
            subscribed INTEGER DEFAULT 0,
            coin TEXT DEFAULT 'bitcoin'
        )
    ''')
    conn.commit()
    conn.close()

# Function to fetch the current price of BTC
def fetch_btc_price():
    response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd")
    data = response.json()
    return data['bitcoin']['usd']

# Notify about price change
def notify_price_change(context: CallbackContext):
    current_price = fetch_btc_price()
    user_id = context.job.context
    logger.info(f"Current BTC price: {current_price}")

    # Send message to the user
    context.bot.send_message(chat_id=user_id, text=f"Current BTC price: ${current_price}")

# Command to set the alert threshold
def set_threshold(update: Update, context: CallbackContext):
    if update.effective_user.id in ADMIN_IDS:
        if context.args:
            try:
                new_threshold = float(context.args[0])
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute("UPDATE user_settings SET threshold = ? WHERE user_id = ?", (new_threshold, update.effective_user.id))
                conn.commit()
                conn.close()
                update.message.reply_text(f"Threshold set to {new_threshold}%")
            except ValueError:
                update.message.reply_text("Please provide a valid number.")
        else:
            update.message.reply_text("Please specify the new threshold.")
    else:
        update.message.reply_text("You are not authorized to use this command.")

# Command to set the coin
def set_coin(update: Update, context: CallbackContext):
    if update.effective_user.id in ADMIN_IDS:
        if context.args:
            new_coin = context.args[0].lower()
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("UPDATE user_settings SET coin = ? WHERE user_id = ?", (new_coin, update.effective_user.id))
            conn.commit()
            conn.close()
            update.message.reply_text(f"Coin set to {new_coin}.")
        else:
            update.message.reply_text("Please specify the new coin.")
    else:
        update.message.reply_text("You are not authorized to use this command.")

# Command to get current price
def current_price(update: Update, context: CallbackContext):
    price = fetch_btc_price()
    update.message.reply_text(f"Current BTC price: ${price}")

# Command to get price prediction (placeholder)
def price_prediction(update: Update, context: CallbackContext):
    update.message.reply_text("Price prediction for the next 30 days is not implemented yet.")

# Command to subscribe to alerts
def subscribe(update: Update, context: CallbackContext):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO user_settings (user_id, subscribed) VALUES (?, 1)", (update.effective_user.id,))
    conn.commit()
    conn.close()
    update.message.reply_text("You have subscribed to real-time price updates.")
    
    # Schedule a job to send price updates every minute
    context.job_queue.run_repeating(notify_price_change, interval=60, first=0, context=update.effective_user.id)

# Command to unsubscribe from alerts
def unsubscribe(update: Update, context: CallbackContext):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE user_settings SET subscribed = 0 WHERE user_id = ?", (update.effective_user.id,))
    conn.commit()
    conn.close()
    update.message.reply_text("You have unsubscribed from real-time price updates.")

# Command to show help
def help_command(update: Update, context: CallbackContext):
    update.message.reply_text(
        "/setthreshold <percentage> - Set the price change threshold\n"
        "/setcoin <coin_name> - Set the monitored coin\n"
        "/cp - Get current BTC price\n"
        "/prediction - Get price prediction for the next 30 days\n"
        "/subscribe - Subscribe to real-time price updates\n"
        "/unsubscribe - Unsubscribe from real-time price updates\n"
        "/help - Show this help message"
    )

# Main function to start the bot
def main():
    create_db()  # Create the database
    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    # Register command handlers
    dp.add_handler(CommandHandler("setthreshold", set_threshold))
    dp.add_handler(CommandHandler("setcoin", set_coin))
    dp.add_handler(CommandHandler("cp", current_price))
    dp.add_handler(CommandHandler("prediction", price_prediction))
    dp.add_handler(CommandHandler("subscribe", subscribe))
    dp.add_handler(CommandHandler("unsubscribe", unsubscribe))
    dp.add_handler(CommandHandler("help", help_command))

    updater.start_polling()
    logger.info("Bot started. Polling...")
    updater.idle()

if __name__ == "__main__":
    main()

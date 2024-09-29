import os
import requests
import logging
import pytz
import sqlite3
from datetime import datetime
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from apscheduler.schedulers.background import BackgroundScheduler

# Configurations
ADMIN_IDS = [int(admin_id) for admin_id in os.getenv("ADMIN_IDS", "1318663278,1318663278").split(',')]  # Admin IDs from environment variables
CHAT_ID = os.getenv("CHAT_ID", "-1001824360922")  # Chat ID where the bot sends notifications
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7066257336:AAHiASvtYMLHHTldyiFMVfOeAfBLRSudDhY")  # Telegram bot token from environment variables
BTC_THRESHOLD = float(os.getenv("BTC_THRESHOLD", -2))  # Default percentage threshold for BTC drop
TIMEZONE_IST = pytz.timezone('Asia/Kolkata')  # Indian Standard Time (IST)

# Database setup
conn = sqlite3.connect('user_settings.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    threshold REAL DEFAULT -2,
    subscribed INTEGER DEFAULT 0
)
''')
conn.commit()

# Setup logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Fetch current BTC price
def get_current_price():
    try:
        response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd')
        data = response.json()
        return data['bitcoin']['usd']
    except Exception as e:
        logging.error(f"Error fetching current price: {e}")
        return None

# Notify if the price drops below the threshold
def notify_if_price_drops(context: CallbackContext):
    current_price = get_current_price()
    if current_price is None:
        return  # Exit if unable to fetch price

    for user in get_subscribed_users():
        threshold = user[1]  # User-specific threshold
        last_price = context.job.last_run_time  # Previous price stored in job context

        if last_price is None:
            context.job.last_run_time = current_price  # Initialize the first run
            return

        price_change_percentage = ((last_price - current_price) / last_price) * 100
        if price_change_percentage <= threshold:
            context.bot.send_message(chat_id=user[0], text=f"BTC price dropped by {abs(price_change_percentage):.2f}% to ${current_price}.")

        context.job.last_run_time = current_price  # Update the last price for the next check

# Get users who are subscribed for price alerts
def get_subscribed_users():
    cursor.execute('SELECT user_id, threshold FROM users WHERE subscribed = 1')
    return cursor.fetchall()

# Command to get current price
def current_price(update: Update, context: CallbackContext):
    price = get_current_price()
    if price:
        update.message.reply_text(f"Current BTC price: ${price}")
    else:
        update.message.reply_text("Error fetching current BTC price.")

# Command to set the BTC drop threshold
def set_threshold(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    if update.effective_user.id in ADMIN_IDS:
        try:
            new_threshold = float(context.args[0])
            cursor.execute('INSERT OR REPLACE INTO users (user_id, threshold) VALUES (?, ?)', (user_id, new_threshold))
            conn.commit()
            update.message.reply_text(f"BTC drop threshold set to {new_threshold}%")
        except (IndexError, ValueError):
            update.message.reply_text("Usage: /setthreshold <percentage>")
    else:
        update.message.reply_text("You do not have permission to use this command.")

# Command to subscribe to price alerts
def subscribe(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    cursor.execute('INSERT OR REPLACE INTO users (user_id, subscribed) VALUES (?, ?)', (user_id, 1))
    conn.commit()
    update.message.reply_text("You have subscribed to BTC price alerts.")

# Command to unsubscribe from price alerts
def unsubscribe(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    cursor.execute('INSERT OR REPLACE INTO users (user_id, subscribed) VALUES (?, ?)', (user_id, 0))
    conn.commit()
    update.message.reply_text("You have unsubscribed from BTC price alerts.")

# Command to check current threshold
def current_threshold(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    cursor.execute('SELECT threshold FROM users WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()
    if result:
        update.message.reply_text(f"Your current BTC drop threshold is set to {result[0]}%")
    else:
        update.message.reply_text("You do not have a threshold set.")

# Command to get help
def help_command(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Commands:\n"
        "/cp - Get current BTC price\n"
        "/setthreshold <percentage> - Set your BTC drop threshold (Admin only)\n"
        "/subscribe - Subscribe to BTC price alerts\n"
        "/unsubscribe - Unsubscribe from BTC price alerts\n"
        "/currentthreshold - Check your current threshold\n"
        "/help - Show this help message"
    )

# Unknown command handler
def unknown_command(update: Update, context: CallbackContext):
    update.message.reply_text("Sorry, I didn't understand that command.")

# Main function to start the bot
def main():
    updater = Updater(token=TELEGRAM_BOT_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # Add command handlers
    dispatcher.add_handler(CommandHandler("cp", current_price))
    dispatcher.add_handler(CommandHandler("setthreshold", set_threshold))
    dispatcher.add_handler(CommandHandler("subscribe", subscribe))
    dispatcher.add_handler(CommandHandler("unsubscribe", unsubscribe))
    dispatcher.add_handler(CommandHandler("currentthreshold", current_threshold))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("unknown", unknown_command))

    # Scheduler to check for price drops
    scheduler = BackgroundScheduler(timezone=TIMEZONE_IST)
    # Pass the context to the job through a lambda function
    scheduler.add_job(notify_if_price_drops, 'interval', minutes=15, id='price_drop_check', kwargs={'context': updater.job_queue})
    scheduler.start()

    # Start the bot
    updater.start_polling()
    logging.info("Bot started.")
    updater.idle()

if __name__ == '__main__':
    main()

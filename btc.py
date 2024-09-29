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
ADMIN_IDS = [int(admin_id) for admin_id in os.getenv("ADMIN_IDS", "1318663278").split(',')]  # Admin IDs from environment variables
CHAT_ID = os.getenv("CHAT_ID", "-1001824360922")  # Chat ID where the bot sends notifications
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7066257336:AAHiASvtYMLHHTldyiFMVfOeAfBLRSudDhY")  # Telegram bot token from environment variables
BTC_THRESHOLD = float(os.getenv("BTC_THRESHOLD", -0.2))  # Default percentage threshold for BTC drop
TIMEZONE_IST = pytz.timezone('Asia/Kolkata')  # Indian Standard Time (IST)

# Database setup
conn = sqlite3.connect('user_settings.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    threshold REAL DEFAULT -0.2,
    subscribed INTEGER DEFAULT 0,
    coin TEXT DEFAULT 'bitcoin'
)
''')
conn.commit()

# Setup logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Fetch current price
def get_current_price(coin='bitcoin'):
    try:
        response = requests.get(f'https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd')
        data = response.json()
        return data[coin]['usd']
    except Exception as e:
        logging.error(f"Error fetching current price: {e}")
        return None

# Fetch 30-day price prediction
def get_price_prediction(coin='bitcoin'):
    try:
        response = requests.get(f'https://api.coingecko.com/api/v3/coins/{coin}/forecast')
        data = response.json()
        return data['market_data']['price_change_percentage_30d']  # This can be adjusted as per actual API response
    except Exception as e:
        logging.error(f"Error fetching price prediction: {e}")
        return None

# Notify if the price drops or rises above the threshold
def notify_price_change(context: CallbackContext):
    coin = 'bitcoin'  # You can change this to another coin as needed
    current_price = get_current_price(coin)
    if current_price is None:
        return  # Exit if unable to fetch price

    for user in get_subscribed_users():
        threshold = user[1]  # User-specific threshold
        last_price = context.job.last_run_time  # Previous price stored in job context

        if last_price is None:
            context.job.last_run_time = current_price  # Initialize the first run
            return

        price_change_percentage = ((current_price - last_price) / last_price) * 100
        if abs(price_change_percentage) >= abs(threshold):
            direction = "up" if price_change_percentage > 0 else "down"
            context.bot.send_message(chat_id=user[0], text=f"{coin.capitalize()} price has changed by {abs(price_change_percentage):.2f}% ({direction}) to ${current_price}.")

        context.job.last_run_time = current_price  # Update the last price for the next check

# Get users who are subscribed for price alerts
def get_subscribed_users():
    cursor.execute('SELECT user_id, threshold FROM users WHERE subscribed = 1')
    return cursor.fetchall()

# Command to get current price
def current_price(update: Update, context: CallbackContext):
    coin = 'bitcoin'  # Default coin
    price = get_current_price(coin)
    if price:
        update.message.reply_text(f"Current {coin.capitalize()} price: ${price}")
    else:
        update.message.reply_text("Error fetching current price.")

# Command to get 30-day price prediction
def price_prediction(update: Update, context: CallbackContext):
    coin = 'bitcoin'  # Default coin
    prediction = get_price_prediction(coin)
    if prediction is not None:
        update.message.reply_text(f"The 30-day price prediction change for {coin.capitalize()} is: {prediction}%")
    else:
        update.message.reply_text("Error fetching price prediction.")

# Command to set the price change threshold
def set_threshold(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    if update.effective_user.id in ADMIN_IDS:
        try:
            new_threshold = float(context.args[0])
            cursor.execute('INSERT OR REPLACE INTO users (user_id, threshold) VALUES (?, ?)', (user_id, new_threshold))
            conn.commit()
            update.message.reply_text(f"Price change threshold set to {new_threshold}%")
        except (IndexError, ValueError):
            update.message.reply_text("Usage: /setthreshold <percentage>")
    else:
        update.message.reply_text("You do not have permission to use this command.")

# Command to subscribe to price alerts
def subscribe(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    cursor.execute('INSERT OR REPLACE INTO users (user_id, subscribed) VALUES (?, ?)', (user_id, 1))
    conn.commit()
    update.message.reply_text("You have subscribed to price alerts.")

# Command to unsubscribe from price alerts
def unsubscribe(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    cursor.execute('INSERT OR REPLACE INTO users (user_id, subscribed) VALUES (?, ?)', (user_id, 0))
    conn.commit()
    update.message.reply_text("You have unsubscribed from price alerts.")

# Command to change the coin for price alerts
def set_coin(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    if update.effective_user.id in ADMIN_IDS:
        try:
            new_coin = context.args[0].lower()  # Change this to any coin available in the API
            cursor.execute('INSERT OR REPLACE INTO users (user_id, coin) VALUES (?, ?)', (user_id, new_coin))
            conn.commit()
            update.message.reply_text(f"Coin for price alerts set to {new_coin}.")
        except IndexError:
            update.message.reply_text("Usage: /setcoin <coin_name>")
    else:
        update.message.reply_text("You do not have permission to use this command.")

# Command to check current threshold
def current_threshold(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    cursor.execute('SELECT threshold FROM users WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()
    if result:
        update.message.reply_text(f"Your current price change threshold is set to {result[0]}%")
    else:
        update.message.reply_text("You do not have a threshold set.")

# Command to get help
def help_command(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Commands:\n"
        "/cp - Get current price\n"
        "/prediction - Get 30-day price prediction\n"
        "/setthreshold <percentage> - Set your price change threshold (Admin only)\n"
        "/subscribe - Subscribe to price alerts\n"
        "/unsubscribe - Unsubscribe from price alerts\n"
        "/setcoin <coin_name> - Change the coin for alerts (Admin only)\n"
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
    dispatcher.add_handler(CommandHandler("prediction", price_prediction))
    dispatcher.add_handler(CommandHandler("setthreshold", set_threshold))
    dispatcher.add_handler(CommandHandler("subscribe", subscribe))
    dispatcher.add_handler(CommandHandler("unsubscribe", unsubscribe))
    dispatcher.add_handler(CommandHandler("setcoin", set_coin))
    dispatcher.add_handler(CommandHandler("currentthreshold", current_threshold))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("unknown", unknown_command))

    # Scheduler to check for price changes
    scheduler = BackgroundScheduler(timezone=TIMEZONE_IST)
    scheduler.add_job(notify_price_change, 'interval', minutes=15, id='price_change_check')
    scheduler.start()

    # Start the bot
    updater.start_polling()
    logging.info("Bot started.")
    updater.idle()

if __name__ == '__main__':
    main()

import os
import requests
import logging
import pytz
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
    current_price = get_current_price()  # Fetch the current price
    if current_price is None:
        return  # Exit if unable to fetch price

    last_price = context.job.last_run_time  # Previous price stored in job context
    if last_price is None:
        context.job.last_run_time = current_price  # Initialize the first run
        return

    price_change_percentage = ((last_price - current_price) / last_price) * 100
    if price_change_percentage <= BTC_THRESHOLD:
        context.bot.send_message(chat_id=CHAT_ID, text=f"BTC price dropped by {abs(price_change_percentage):.2f}% to ${current_price}.")

    context.job.last_run_time = current_price  # Update the last price for the next check

# Command to get current price
def current_price(update: Update, context: CallbackContext):
    price = get_current_price()
    if price:
        update.message.reply_text(f"Current BTC price: ${price}")
    else:
        update.message.reply_text("Error fetching current BTC price.")

# Command to set the BTC drop threshold
def set_threshold(update: Update, context: CallbackContext):
    if update.effective_user.id in ADMIN_IDS:
        try:
            new_threshold = float(context.args[0])
            global BTC_THRESHOLD
            BTC_THRESHOLD = new_threshold
            update.message.reply_text(f"BTC drop threshold set to {BTC_THRESHOLD}%")
        except (IndexError, ValueError):
            update.message.reply_text("Usage: /setthreshold <percentage>")
    else:
        update.message.reply_text("You do not have permission to use this command.")

# Unknown command handler
def unknown_command(update: Update, context: CallbackContext):
    update.message.reply_text("Sorry, I didn't understand that command. Use /cp to get the current BTC price.")

# Main function to start the bot
def main():
    updater = Updater(token=TELEGRAM_BOT_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # Add command handlers
    dispatcher.add_handler(CommandHandler("cp", current_price))
    dispatcher.add_handler(CommandHandler("setthreshold", set_threshold))
    dispatcher.add_handler(CommandHandler("help", lambda update, context: update.message.reply_text("Use /cp to get current BTC price.")))
    dispatcher.add_handler(CommandHandler("unknown", unknown_command))

    # Scheduler to check for price drops
    scheduler = BackgroundScheduler(timezone=TIMEZONE_IST)
    scheduler.add_job(notify_if_price_drops, 'interval', minutes=15, args=[context])
    scheduler.start()

    # Start the bot
    updater.start_polling()
    logging.info("Bot started.")
    updater.idle()

if __name__ == '__main__':
    main()

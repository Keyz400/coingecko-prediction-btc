import os
import requests
import time
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

# Configurations
TELEGRAM_BOT_TOKEN = os.getenv("7066257336:AAHiASvtYMLHHTldyiFMVfOeAfBLRSudDhY")  # Your Telegram bot token
BTC_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"

previous_price = None

def get_btc_price():
    response = requests.get(BTC_URL)
    data = response.json()
    return data['bitcoin']['usd']

def notify_price_drop(context: CallbackContext):
    global previous_price
    current_price = get_btc_price()
    
    if previous_price is None:
        previous_price = current_price
        return
    
    # Calculate the drop percentage
    drop_percentage = ((previous_price - current_price) / previous_price) * 100
    
    if drop_percentage >= 0.2:
        context.bot.send_message(chat_id=context.job.context, text=f"BTC price dropped to ${current_price:.2f} ({drop_percentage:.2f}% decrease).")
    
    previous_price = current_price  # Update the previous price

def main():
    updater = Updater(TELEGRAM_BOT_TOKEN)
    job_queue = updater.job_queue
    job_queue.run_repeating(notify_price_drop, interval=60, first=0, context=updater.message.chat_id)  # Check every 1 minute
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

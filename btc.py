import os
import requests
import time
import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

# Configurations
TELEGRAM_BOT_TOKEN = "7066257336:AAHiASvtYMLHHTldyiFMVfOeAfBLRSudDhY"  # Your Telegram bot token
CHAT_ID = "-1001824360922"  # Replace with your Telegram channel ID or username
BTC_THRESHOLD = 0.2  # Percentage threshold for BTC drop

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Function to get the current BTC price
def get_btc_price():
    response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd")
    return response.json().get('bitcoin', {}).get('usd', 0)

# Function to send message to Telegram
def send_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message,
    }
    requests.post(url, data=data)

# Function to check price drop
def check_price_drop(previous_price):
    current_price = get_btc_price()
    price_change = ((current_price - previous_price) / previous_price) * 100

    if price_change <= -BTC_THRESHOLD:
        message = f"🚨 BTC Price Alert: Price dropped to ${current_price:.2f} (-{abs(price_change):.2f}%)"
        send_message(message)

    return current_price

def main():
    previous_price = get_btc_price()
    send_message(f"🔔 BTC Price Monitoring Started: Initial Price: ${previous_price:.2f}")

    while True:
        time.sleep(60)  # Check every 1 minute
        previous_price = check_price_drop(previous_price)

if __name__ == '__main__':
    main()

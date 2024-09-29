import os
import requests
import time
import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

# Configurations
TELEGRAM_BOT_TOKEN = "7066257336:AAHiASvtYMLHHTldyiFMVfOeAfBLRSudDhY"
CHAT_ID = "-1001824360922"
BTC_THRESHOLD_LOW = -0.02  # Default threshold for price drop
BTC_THRESHOLD_HIGH = 0.02  # Default threshold for price increase
COIN = "bitcoin"  # Default coin

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Function to get the current coin price
def get_coin_price(coin):
    response = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd")
    return response.json().get(coin, {}).get('usd', 0)

# Function to send message to Telegram
def send_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message,
    }
    requests.post(url, data=data)

# Function to check price drop or rise
def check_price_change(previous_price, coin):
    current_price = get_coin_price(coin)
    price_change = ((current_price - previous_price) / previous_price) * 100

    if price_change <= BTC_THRESHOLD_LOW:
        message = f"🚨 {coin.upper()} Price Alert: Price dropped to ${current_price:.2f} (-{abs(price_change):.2f}%)"
        send_message(message)
    elif price_change >= BTC_THRESHOLD_HIGH:
        message = f"🚨 {coin.upper()} Price Alert: Price increased to ${current_price:.2f} (+{abs(price_change):.2f}%)"
        send_message(message)

    return current_price

# Command to predict coin value
def predict(update: Update, context: CallbackContext):
    coin = COIN
    price = get_coin_price(coin)
    prediction = price * 1.1  # Simple prediction: +10%
    message = f"Prediction: {coin.upper()} might reach ${prediction:.2f} soon!"
    update.message.reply_text(message)

# Command to get current price
def current_price(update: Update, context: CallbackContext):
    coin = COIN
    price = get_coin_price(coin)
    message = f"Current {coin.upper()} Price: ${price:.2f}"
    update.message.reply_text(message)

# Command to set price thresholds
def set_threshold(update: Update, context: CallbackContext):
    global BTC_THRESHOLD_LOW, BTC_THRESHOLD_HIGH
    try:
        low, high = float(context.args[0]), float(context.args[1])
        BTC_THRESHOLD_LOW = low
        BTC_THRESHOLD_HIGH = high
        message = f"Thresholds updated: Low = {BTC_THRESHOLD_LOW}%, High = {BTC_THRESHOLD_HIGH}%"
    except (IndexError, ValueError):
        message = "Usage: /threshold <low> <high> (e.g., /threshold -2 2)"
    update.message.reply_text(message)

# Command to change the coin
def change_coin(update: Update, context: CallbackContext):
    global COIN
    try:
        new_coin = context.args[0].lower()
        COIN = new_coin
        message = f"Coin changed to {COIN.upper()}. Now tracking {COIN.upper()}."
    except IndexError:
        message = "Usage: /ch <coin_id> (e.g., /ch ethereum)"
    update.message.reply_text(message)

# Command to show help
def help_command(update: Update, context: CallbackContext):
    message = """
    Available Commands:
    /pre - Predict the value of the current coin
    /cp - Get the current price of the tracked coin
    /threshold <low> <high> - Set the low and high price change thresholds
    /ch <coin_id> - Change the coin to track (e.g., bitcoin, ethereum)
    /help - Show this help message
    """
    update.message.reply_text(message)

def main():
    previous_price = get_coin_price(COIN)
    send_message(f"🔔 {COIN.upper()} Price Monitoring Started: Initial Price: ${previous_price:.2f}")

    # Set up the bot
    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # Add command handlers
    dispatcher.add_handler(CommandHandler("pre", predict))
    dispatcher.add_handler(CommandHandler("cp", current_price))
    dispatcher.add_handler(CommandHandler("threshold", set_threshold))
    dispatcher.add_handler(CommandHandler("ch", change_coin))
    dispatcher.add_handler(CommandHandler("help", help_command))

    # Start the bot
    updater.start_polling()

    # Monitor price every 1 minute
    while True:
        time.sleep(60)
        previous_price = check_price_change(previous_price, COIN)

if __name__ == '__main__':
    main()

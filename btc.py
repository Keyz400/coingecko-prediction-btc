import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datetime
import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import Bot
from apscheduler.schedulers.background import BackgroundScheduler

# Telegram Bot API key
API_KEY = '7066257336:AAHiASvtYMLHHTldyiFMVfOeAfBLRSudDhY'
CHAT_ID = '-1001824360922'  # Set your chat ID for BTC drop notifications

# Fetch historical data from CoinGecko API for Bitcoin
def fetch_historical_data():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
    params = {'vs_currency': 'usd', 'days': '365'}
    response = requests.get(url, params=params)
    data = response.json()
    prices = data['prices']
    timestamps = [timestamp for timestamp, _ in prices]
    prices = [price for _, price in prices]
    return timestamps, prices

# Fetch current price of Bitcoin
def fetch_current_price():
    url = 'https://api.coingecko.com/api/v3/simple/price'
    params = {'ids': 'bitcoin', 'vs_currencies': 'usd'}
    response = requests.get(url, params=params)
    data = response.json()
    return data['bitcoin']['usd']

# Create features and labels for training
def create_features_and_labels(prices, window_size):
    X = []
    y = []
    for i in range(len(prices) - window_size):
        X.append(prices[i:i+window_size])
        y.append(prices[i+window_size])
    X = np.array(X)
    y = np.array(y)
    return X, y

# Start command handler
def start(update, context):
    update.message.reply_text('Welcome! You can use /pd to predict BTC price, /cp to get the current BTC price.')

# BTC prediction command handler
def predict_btc(update, context):
    try:
        # Fetch historical data
        timestamps, prices = fetch_historical_data()

        # Create a DataFrame from the fetched data
        data = {'Timestamp': timestamps, 'Price': prices}
        df = pd.DataFrame(data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df.set_index('Timestamp', inplace=True)

        # Normalize the data
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(df['Price'].values.reshape(-1, 1)).flatten()

        # Define the number of previous days to consider for prediction
        window_size = 7

        # Prepare the data for training
        X, y = create_features_and_labels(scaled_prices, window_size)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions for the next 7 days
        last_data = scaled_prices[-window_size:].reshape(1, -1)
        predictions_message = 'Predicted Prices for the Next 7 Days:\n'
        current_date = df.index[-1]

        # Loop to predict future prices day by day
        for i in range(7):
            next_prediction = model.predict(last_data)[0]
            predicted_price = scaler.inverse_transform([[next_prediction]])[0][0]

            current_date += datetime.timedelta(days=1)
            predictions_message += f'{current_date.date()}: ${round(predicted_price, 2)}\n'

            last_data = np.append(last_data[:, 1:], next_prediction).reshape(1, -1)

        update.message.reply_text(predictions_message)

    except Exception as e:
        update.message.reply_text(f'Error: {e}')

# BTC current price command handler
def current_price(update, context):
    try:
        current_price = fetch_current_price()
        update.message.reply_text(f'Current BTC Price: ${current_price}')
    except Exception as e:
        update.message.reply_text(f'Error: {e}')

# Background job to check BTC price drop by 2% and notify the user
def check_btc_price_drop():
    previous_price = fetch_current_price()
    scheduler = BackgroundScheduler()

    def notify_if_price_drops():
        current_price = fetch_current_price()
        price_drop_percentage = ((previous_price - current_price) / previous_price) * 100
        if price_drop_percentage >= 2:
            message = f'BTC Price Drop Alert! BTC has dropped by {price_drop_percentage:.2f}% to ${current_price}'
            bot.send_message(chat_id=CHAT_ID, text=message)

    # Schedule the task to run every 15 minutes
    scheduler.add_job(notify_if_price_drops, 'interval', minutes=15)
    scheduler.start()

# Error handler for unknown commands
def unknown(update, context):
    update.message.reply_text("Sorry, I didn't understand that command.")

# Main function to set up the bot
def main():
    global bot
    bot = Bot(API_KEY)
    
    # Set up logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    # Create the Updater and pass it your bot's token
    updater = Updater(API_KEY, use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # Add handler for /start command
    dp.add_handler(CommandHandler('start', start))

    # Add handler for /pd command (BTC price prediction)
    dp.add_handler(CommandHandler('pd', predict_btc))

    # Add handler for /cp command (current BTC price)
    dp.add_handler(CommandHandler('cp', current_price))

    # Add handler for unknown commands
    dp.add_handler(MessageHandler(Filters.command, unknown))

    # Start the BTC price drop monitoring
    check_btc_price_drop()

    # Start the bot
    updater.start_polling()

    # Run the bot until Ctrl-C is pressed
    updater.idle()

if __name__ == '__main__':
    main()

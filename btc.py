import pandas as pd
import requests
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datetime
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# Telegram Bot API key
API_KEY = '7066257336:AAHiASvtYMLHHTldyiFMVfOeAfBLRSudDhY'

# Fetch historical data from CoinGecko API
def fetch_historical_data():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
    params = {'vs_currency': 'usd', 'days': '365'}
    response = requests.get(url, params=params)
    data = response.json()
    prices = [price for _, price in data['prices']]
    timestamps = [timestamp for timestamp, _ in data['prices']]
    return timestamps, prices

# Create features and labels for training
def create_features_and_labels(prices, window_size):
    X, y = [], []
    for i in range(len(prices) - window_size):
        X.append(prices[i:i + window_size])
        y.append(prices[i + window_size])
    return np.array(X), np.array(y)

# Start command handler
def start(update, context):
    update.message.reply_text('Welcome! Enter the number of days for Bitcoin price predictions (e.g., 3, 5, 10).')

# Handle user input and prediction
def handle_query(update, context):
    try:
        next_days = int(update.message.text)
    except ValueError:
        update.message.reply_text('Please enter a valid number.')
        return

    update.message.reply_text(f'Predicting for the next {next_days} days...')

    # Fetch and process data
    timestamps, prices = fetch_historical_data()
    df = pd.DataFrame({'Timestamp': pd.to_datetime(timestamps, unit='ms'), 'Price': prices}).set_index('Timestamp')

    # Scale data
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(df['Price'].values.reshape(-1, 1)).flatten()

    # Prepare training data
    window_size = 7
    X, y = create_features_and_labels(scaled_prices, window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict future prices
    last_data = scaled_prices[-window_size:].reshape(1, -1)
    predictions_message = f'Predicted prices for the next {next_days} days:\n'
    current_date = df.index[-1]

    for _ in range(next_days):
        next_prediction = model.predict(last_data)[0]
        predicted_price = scaler.inverse_transform([[next_prediction]])[0][0]
        current_date += datetime.timedelta(days=1)
        predictions_message += f'{current_date.date()}: {round(predicted_price, 2)}\n'
        last_data = np.append(last_data[:, 1:], next_prediction).reshape(1, -1)

    update.message.reply_text(predictions_message)

# Handle unknown commands
def unknown(update, context):
    update.message.reply_text("Sorry, I didn't understand that command.")

# Main function to set up the bot
def main():
    updater = Updater(API_KEY, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_query))
    dp.add_handler(MessageHandler(Filters.command, unknown))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

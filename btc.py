import pandas as pd
import requests
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datetime
import matplotlib.pyplot as plt
import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import InputFile
import io

# Telegram Bot API key
API_KEY = '7066257336:AAHiASvtYMLHHTldyiFMVfOeAfBLRSudDhY'

# Supported coins and their CoinGecko IDs
SUPPORTED_COINS = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'BNB': 'binancecoin',
    'TON': 'toncoin',
    'MKR': 'maker',
    'ADA': 'cardano',
    'DOGE': 'dogecoin',
    'SOL': 'solana',
    'DOT': 'polkadot',
    'LTC': 'litecoin',
    'XRP': 'ripple'
}

# Fetch historical data from CoinGecko API for the selected coin
def fetch_historical_data(coin_id):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
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

# Helper function to split long messages
def split_message(message, max_length=4000):
    return [message[i:i + max_length] for i in range(0, len(message), max_length)]

# Start command handler
def start(update, context):
    update.message.reply_text(f'Welcome! Use /help to see the list of supported coins. Enter your query in the format: <coin> <days>. Example: BTC 5')

# Help command handler to list supported coins
def help_command(update, context):
    coins_list = ', '.join(SUPPORTED_COINS.keys())
    update.message.reply_text(f'Supported coins: {coins_list}\nUse the format <coin> <days> for predictions. Example: BTC 5')

# Generate and send plot image
def send_prediction_image(dates, predictions, update, context, coin):
    plt.figure(figsize=(10, 6))
    plt.plot(dates, predictions, marker='o', linestyle='-', color='b')
    plt.fill_between(dates, predictions, color='skyblue', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel(f'Price (USD)')
    plt.title(f'{coin} Price Predictions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    update.message.reply_photo(photo=InputFile(buf, filename=f'{coin}_predictions.png'))

# Handle user input and prediction
def handle_query(update, context):
    user_input = update.message.text.split()
    if len(user_input) != 2:
        update.message.reply_text('Please enter your query in the format: <coin> <days>. Example: BTC 5')
        return

    coin, days = user_input[0].upper(), user_input[1]

    # Check if the coin is supported
    if coin not in SUPPORTED_COINS:
        update.message.reply_text(f'Sorry, I don\'t support "{coin}". Use /help to see the list of supported coins.')
        return

    # Validate days input
    try:
        next_days = int(days)
    except ValueError:
        update.message.reply_text('Please enter a valid number of days.')
        return

    update.message.reply_text(f'Predicting {coin} prices for the next {next_days} days...')

    # Fetch and process data for the selected coin
    coin_id = SUPPORTED_COINS[coin]
    timestamps, prices = fetch_historical_data(coin_id)
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
    predictions_message = f'{coin} predicted prices for the next {next_days} days:\n'
    current_date = df.index[-1]
    predicted_dates = []
    predicted_prices = []

    for _ in range(next_days):
        next_prediction = model.predict(last_data)[0]
        predicted_price = scaler.inverse_transform([[next_prediction]])[0][0]
        current_date += datetime.timedelta(days=1)

        predictions_message += f'{current_date.date()}: {round(predicted_price, 2)} USD\n'
        predicted_dates.append(current_date.date())
        predicted_prices.append(predicted_price)

        last_data = np.append(last_data[:, 1:], next_prediction).reshape(1, -1)

    # Split and send the predictions message
    for message in split_message(predictions_message):
        update.message.reply_text(message)

    # Send prediction image
    send_prediction_image(predicted_dates, predicted_prices, update, context, coin)

# Handle unknown commands
def unknown(update, context):
    update.message.reply_text("Sorry, I didn't understand that command.")

# Main function to set up the bot
def main():
    updater = Updater(API_KEY, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CommandHandler('help', help_command))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_query))
    dp.add_handler(MessageHandler(Filters.command, unknown))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

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
import json

# Telegram Bot API key
API_KEY = '7066257336:AAHiASvtYMLHHTldyiFMVfOeAfBLRSudDhY'
ADMIN_ID = 1318663278  # Replace with the actual Telegram ID of the admin

# Supported coins (initial coins loaded from a JSON file)
COINS_FILE = 'coins.json'

def load_coins():
    try:
        with open(COINS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

SUPPORTED_COINS = load_coins()

# Fetch historical data from CoinGecko API for the selected coin
def fetch_historical_data(coin_id):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {'vs_currency': 'usd', 'days': '365'}
    response = requests.get(url, params=params)
    data = response.json()
    prices = [price for _, price in data['prices']]
    timestamps = [timestamp for timestamp, _ in data['prices']]
    return timestamps, prices

# Fetch current price of a coin
def fetch_current_price(coin_id):
    url = f'https://api.coingecko.com/api/v3/simple/price'
    params = {'ids': coin_id, 'vs_currencies': 'usd'}
    response = requests.get(url, params=params)
    data = response.json()
    return data.get(coin_id, {}).get('usd', None)

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
    plt.plot(dates, predictions, marker='o', linestyle='-', color='b', label=f'{coin} Predictions')
    plt.fill_between(dates, predictions, color='skyblue', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel(f'Price (USD)')
    plt.title(f'{coin} Price Predictions')
    plt.xticks(rotation=45)
    plt.legend()
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

# Fetch and send current price of a coin
def current_price(update, context):
    if len(context.args) != 1:
        update.message.reply_text("Please provide a valid coin symbol. Example: /price BTC")
        return

    coin = context.args[0].upper()
    if coin not in SUPPORTED_COINS:
        update.message.reply_text(f"I don't support {coin}. Use /help to see the list of supported coins.")
        return

    coin_id = SUPPORTED_COINS[coin]
    price = fetch_current_price(coin_id)
    if price:
        update.message.reply_text(f'The current price of {coin} is ${price} USD.')
    else:
        update.message.reply_text(f'Failed to fetch the current price of {coin}.')

# Admin command to add a new coin
def add_coin(update, context):
    if update.message.from_user.id != ADMIN_ID:
        update.message.reply_text('Only the admin can add new coins.')
        return

    if len(context.args) != 2:
        update.message.reply_text('Usage: /addcoin <symbol> <coingecko_id>. Example: /addcoin LINK chainlink')
        return

    coin_symbol, coin_id = context.args[0].upper(), context.args[1]
    SUPPORTED_COINS[coin_symbol] = coin_id
    with open(COINS_FILE, 'w') as f:
        json.dump(SUPPORTED_COINS, f)
    update.message.reply_text(f'Added {coin_symbol} to the supported coins list.')

# Admin command to remove a coin
def remove_coin(update, context):
    if update.message.from_user.id != ADMIN_ID:
        update.message.reply_text('Only the admin can remove coins.')
        return

    if len(context.args) != 1:
        update.message.reply_text('Usage: /removecoin <symbol>. Example: /removecoin BTC')
        return

    coin_symbol = context.args[0].upper()
    if coin_symbol in SUPPORTED_COINS:
        del SUPPORTED_COINS[coin_symbol]
        with open(COINS_FILE, 'w') as f:
            json.dump(SUPPORTED_COINS, f)
        update.message.reply_text(f'Removed {coin_symbol} from the supported coins list.')
    else:
        update.message.reply_text(f'{coin_symbol} is not in the supported coins list.')

# Admin command to broadcast a message to all users
def broadcast(update, context):
    if update.message.from_user.id != ADMIN_ID:
        update.message.reply_text('Only the admin can broadcast messages.')
        return

    if not context.args:
        update.message.reply_text('Usage: /broadcast <message>. Example: /broadcast Hello everyone!')
        return

    broadcast_message = ' '.join(context.args)

    # Fetch all chat_ids to broadcast the message
    # This should ideally be fetched from a database where all users are stored.
    # For this implementation, we assume we have a list of users.
    # You need to maintain chat IDs when users interact with the bot.
    users = [ADMIN_ID]  # Add actual user chat IDs here in a real implementation

    for user_id in users:
        try:
            context.bot.send_message(chat_id=user_id, text=broadcast_message)
        except Exception as e:
            logging.error(f"Failed to send message to {user_id}: {e}")

# Function to handle unknown commands
def unknown(update, context):
    update.message.reply_text("Sorry, I didn't understand that command. Use /help to see the available commands.")

# Main function to set up the bot
def main():
    # Set up logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    # Create the Updater and pass it your bot's token
    updater = Updater(API_KEY, use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # Add handler for /start command
    dp.add_handler(CommandHandler('start', start))

    # Add handler for /help command
    dp.add_handler(CommandHandler('help', help_command))

    # Add handler for price fetching
    dp.add_handler(CommandHandler('price', current_price))

    # Admin commands for adding/removing coins
    dp.add_handler(CommandHandler('addcoin', add_coin))
    dp.add_handler(CommandHandler('removecoin', remove_coin))
    
    # Admin broadcast command
    dp.add_handler(CommandHandler('broadcast', broadcast))

    # Handle user predictions query
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_query))

    # Add handler for unknown commands
    dp.add_handler(MessageHandler(Filters.command, unknown))

    # Start the bot
    updater.start_polling()

    # Run the bot until Ctrl-C is pressed or the process receives SIGINT, SIGTERM or SIGABRT
    updater.idle()

if __name__ == '__main__':
    main()

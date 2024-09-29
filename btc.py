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
import pytz

# -------------------- Configuration --------------------

# Replace these with your actual Telegram Bot API key and your chat ID
API_KEY = '7066257336:AAHiASvtYMLHHTldyiFMVfOeAfBLRSudDhY'
CHAT_ID = '-1001824360922'  # Replace with your Telegram chat ID for BTC drop notifications

# Set your timezone (e.g., 'UTC', 'America/New_York')
TIMEZONE = 'UTC'

# --------------------------------------------------------

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Initialize the bot
bot = Bot(API_KEY)

# -------------------- Helper Functions --------------------

def fetch_historical_data():
    """
    Fetches the past 365 days of Bitcoin price data from CoinGecko API.
    Returns timestamps and prices.
    """
    try:
        url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
        params = {'vs_currency': 'usd', 'days': '365'}
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'prices' not in data:
            logging.error("Key 'prices' not found in API response.")
            return None, None

        prices = data['prices']
        timestamps = [timestamp for timestamp, _ in prices]
        prices = [price for _, price in prices]
        return timestamps, prices
    except Exception as e:
        logging.error(f"Error fetching historical data: {e}")
        return None, None

def fetch_current_price():
    """
    Fetches the current price of Bitcoin in USD from CoinGecko API.
    Returns the current price.
    """
    try:
        url = 'https://api.coingecko.com/api/v3/simple/price'
        params = {'ids': 'bitcoin', 'vs_currencies': 'usd'}
        response = requests.get(url, params=params)
        data = response.json()
        return data['bitcoin']['usd']
    except Exception as e:
        logging.error(f"Error fetching current price: {e}")
        return None

def create_features_and_labels(prices, window_size):
    """
    Prepares the dataset for training by creating features and labels.
    """
    X = []
    y = []
    for i in range(len(prices) - window_size):
        X.append(prices[i:i+window_size])
        y.append(prices[i+window_size])
    X = np.array(X)
    y = np.array(y)
    return X, y

def split_message(message, max_length=4000):
    """
    Splits a long message into smaller chunks to comply with Telegram's message size limits.
    """
    return [message[i:i + max_length] for i in range(0, len(message), max_length)]

# -------------------- Command Handlers --------------------

def start(update, context):
    """
    Handles the /start command.
    """
    update.message.reply_text(
        'Welcome! You can use the following commands:\n'
        '/pd - Predict BTC price for the next 7 days.\n'
        '/cp - Get the current BTC price.'
    )

def predict_btc(update, context):
    """
    Handles the /pd command to predict BTC prices.
    """
    try:
        # Fetch historical data
        timestamps, prices = fetch_historical_data()
        if timestamps is None or prices is None:
            update.message.reply_text("Failed to fetch historical data for BTC.")
            return

        # Create DataFrame
        df = pd.DataFrame({
            'Timestamp': pd.to_datetime(timestamps, unit='ms'),
            'Price': prices
        })
        df.set_index('Timestamp', inplace=True)

        # Normalize data
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(df['Price'].values.reshape(-1, 1)).flatten()

        # Prepare training data
        window_size = 7
        X, y = create_features_and_labels(scaled_prices, window_size)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict next 7 days
        last_data = scaled_prices[-window_size:].reshape(1, -1)
        predictions = []
        predictions_message = '🔮 **Predicted BTC Prices for the Next 7 Days:**\n'

        current_date = df.index[-1]

        for _ in range(7):
            next_pred = model.predict(last_data)[0]
            predicted_price = scaler.inverse_transform([[next_pred]])[0][0]
            predictions.append(predicted_price)
            current_date += datetime.timedelta(days=1)
            predictions_message += f'{current_date.date()}: ${predicted_price:.2f}\n'
            last_data = np.append(last_data[:, 1:], next_pred).reshape(1, -1)

        # Send predictions in chunks if necessary
        for message in split_message(predictions_message):
            update.message.reply_text(message, parse_mode='Markdown')

    except Exception as e:
        logging.error(f"Error in predict_btc: {e}")
        update.message.reply_text("An error occurred while predicting BTC prices.")

def current_price_cmd(update, context):
    """
    Handles the /cp command to get the current BTC price.
    """
    try:
        price = fetch_current_price()
        if price is None:
            update.message.reply_text("Failed to fetch the current BTC price.")
            return
        update.message.reply_text(f'💰 **Current BTC Price:** ${price}')
    except Exception as e:
        logging.error(f"Error in current_price_cmd: {e}")
        update.message.reply_text("An error occurred while fetching the current BTC price.")

def unknown_command(update, context):
    """
    Handles unknown commands.
    """
    update.message.reply_text("Sorry, I didn't understand that command. Use /pd to predict BTC price or /cp to get the current BTC price.")

# -------------------- Price Drop Notification --------------------

def check_btc_price_drop():
    """
    Checks if BTC price has dropped by 2% and notifies the user.
    """
    previous_price = fetch_current_price()
    if previous_price is None:
        logging.error("Initial BTC price fetch failed. Cannot monitor price drops.")
        return

    def notify_if_price_drops():
        nonlocal previous_price
        current_price = fetch_current_price()
        if current_price is None:
            logging.error("Failed to fetch current BTC price during monitoring.")
            return

        price_drop_percentage = ((previous_price - current_price) / previous_price) * 100
        if price_drop_percentage >= 2:
            message = f'🚨 **BTC Price Drop Alert!**\nBTC has dropped by {price_drop_percentage:.2f}% to ${current_price:.2f}'
            try:
                bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')
                # Update previous_price to current to avoid repeated alerts
                previous_price = current_price
            except Exception as e:
                logging.error(f"Failed to send alert message: {e}")
        else:
            # Update previous_price to current for future comparisons
            previous_price = current_price

    # Initialize scheduler with timezone
    scheduler = BackgroundScheduler(timezone=pytz.timezone(TIMEZONE))
    scheduler.add_job(notify_if_price_drops, 'interval', minutes=15)
    scheduler.start()
    logging.info("Started BTC price drop monitoring.")

# -------------------- Main Function --------------------

def main():
    # Create the Updater and pass it your bot's token
    updater = Updater(API_KEY, use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # Register command handlers
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("pd", predict_btc))
    dp.add_handler(CommandHandler("cp", current_price_cmd))

    # Register unknown command handler
    dp.add_handler(MessageHandler(Filters.command, unknown_command))

    # Start the BTC price drop monitoring
    check_btc_price_drop()

    # Start the bot
    updater.start_polling()
    logging.info("Bot started.")

    # Run the bot until you press Ctrl-C or the process receives SIGINT, SIGTERM or SIGABRT
    updater.idle()

if __name__ == '__main__':
    main()

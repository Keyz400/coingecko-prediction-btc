import pandas as pd
import requests
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime
from sklearn.preprocessing import MinMaxScaler
import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler
import pyfiglet

# Telegram Bot Setup
API_KEY = '7066257336:AAHiASvtYMLHHTldyiFMVfOeAfBLRSudDhY'

# States for conversation
ENTER_DAYS = range(1)

# Fetch historical data from CoinGecko API
def fetch_historical_data():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': '365'
    }
    response = requests.get(url, params=params)
    data = response.json()
    prices = data['prices']
    timestamps = [timestamp for timestamp, _ in prices]
    prices = [price for _, price in prices]
    return timestamps, prices

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
    update.message.reply_text('Welcome! 😚 I am a Bitcoin price prediction bot. Enter the number of days you want predictions for: Eg:- 3,5,10')
    return ENTER_DAYS

# Function to handle user input for number of prediction days
def enter_days(update, context):
    try:
        next_days = int(update.message.text)
    except ValueError:
        update.message.reply_text('Please enter a valid number 🥺')
        return ENTER_DAYS
    
    update.message.reply_text(f'Fetching predictions for the next {next_days} days...⚡')

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

    # Make predictions for the next n days
    last_data = scaled_prices[-window_size:].reshape(1, -1)

    predictions_message = f'Predicted Prices for the Next {next_days} Days:\n'
    current_date = df.index[-1]

    # Loop to predict future prices day by day
    for i in range(next_days):
        # Predict the next price
        next_prediction = model.predict(last_data)[0]
        predicted_price = scaler.inverse_transform([[next_prediction]])[0][0]

        # Increment the date and append to the message
        current_date += datetime.timedelta(days=1)
        predictions_message += f'{current_date.date()}: {round(predicted_price, 2)}\n'

        # Update last_data by removing the oldest and adding the newest prediction
        last_data = np.append(last_data[:, 1:], next_prediction).reshape(1, -1)

    # Send the predictions back to the user
    figlet_text = pyfiglet.figlet_format("Black")
    update.message.reply_text(figlet_text)
    update.message.reply_text(predictions_message)

    return ConversationHandler.END

# Function to handle cancellation
def cancel(update, context):
    update.message.reply_text('Prediction process cancelled.')
    return ConversationHandler.END

# Main function to set up the bot
def main():
    # Set up logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    # Create the Updater and pass it your bot's token
    updater = Updater(API_KEY, use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # Add conversation handler with states
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            ENTER_DAYS: [MessageHandler(Filters.text & ~Filters.command, enter_days)],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    dp.add_handler(conv_handler)

    # Start the bot
    updater.start_polling()

    # Run the bot until Ctrl-C is pressed or the process receives SIGINT, SIGTERM or SIGABRT
    updater.idle()

if __name__ == '__main__':
    main()

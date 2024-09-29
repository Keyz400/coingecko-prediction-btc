import pandas as pd
import requests
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import ParseMode
import os

# Telegram Bot API key and admin user ID
API_KEY = '7066257336:AAHiASvtYMLHHTldyiFMVfOeAfBLRSudDhY'
ADMIN_ID = '1318663278'

# Global list of supported coins (can be modified by admin)
supported_coins = {
    'btc': 'bitcoin',
    'eth': 'ethereum',
    'bnb': 'binancecoin',
    'xrp': 'ripple',
    'ada': 'cardano',
    'sol': 'solana',
    'dot': 'polkadot',
    'mkr': 'maker',
    'ton': 'toncoin',
    'ltc': 'litecoin'
}

# Fetch historical data from CoinGecko API
def fetch_historical_data(coin_id):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {'vs_currency': 'usd', 'days': '365'}
    response = requests.get(url, params=params)
    data = response.json()
    if 'prices' in data:
        prices = data['prices']
        timestamps = [timestamp for timestamp, _ in prices]
        prices = [price for _, price in prices]
        return timestamps, prices
    else:
        return [], []

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

# Generate a graph for predictions
def generate_graph(timestamps, prices, predictions, coin):
    plt.figure(figsize=(10,6))
    sns.lineplot(x=timestamps, y=prices, label='Historical Prices')
    sns.lineplot(x=[timestamps[-1] + datetime.timedelta(days=i) for i in range(len(predictions))], y=predictions, label='Predicted Prices', color='orange')
    plt.fill_between([timestamps[-1] + datetime.timedelta(days=i) for i in range(len(predictions))], predictions, alpha=0.2, color='orange')
    plt.title(f'{coin.upper()} Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    graph_path = f"{coin}_prediction.png"
    plt.savefig(graph_path)
    plt.close()
    return graph_path

# Handle start command
def start(update, context):
    update.message.reply_text('Welcome! I am a cryptocurrency price prediction bot. Use /help to see available commands.')

# Help command to list all available coins
def help_command(update, context):
    coins_message = "Supported coins:
"
    for symbol, name in supported_coins.items():
        coins_message += f"{symbol.upper()} - {name.capitalize()}
"
    update.message.reply_text(coins_message)

# Handle user query for prediction
def handle_query(update, context):
    query = update.message.text.lower().strip()
    if query in supported_coins:
        coin_id = supported_coins[query]
        update.message.reply_text(f'Fetching predictions for {query.upper()}...')
        timestamps, prices = fetch_historical_data(coin_id)
        if not timestamps:
            update.message.reply_text(f"Error: Could not fetch data for {query.upper()}.")
            return
        
        df = pd.DataFrame({'Timestamp': pd.to_datetime(timestamps, unit='ms'), 'Price': prices})
        df.set_index('Timestamp', inplace=True)

        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(df['Price'].values.reshape(-1, 1)).flatten()

        window_size = 7
        X, y = create_features_and_labels(scaled_prices, window_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        last_data = scaled_prices[-window_size:].reshape(1, -1)
        predictions = []
        for _ in range(7):
            next_prediction = model.predict(last_data)[0]
            predicted_price = scaler.inverse_transform([[next_prediction]])[0][0]
            predictions.append(predicted_price)
            last_data = np.append(last_data[:, 1:], next_prediction).reshape(1, -1)

        graph_path = generate_graph(df.index, df['Price'], predictions, query)
        context.bot.send_photo(chat_id=update.message.chat_id, photo=open(graph_path, 'rb'))
        os.remove(graph_path)

        predictions_message = f'Predicted Prices for {query.upper()}:
'
        for i, price in enumerate(predictions, 1):
            predictions_message += f"Day {i}: ${round(price, 2)}
"
        update.message.reply_text(predictions_message)
    else:
        update.message.reply_text('Coin not supported. Use /help to see available coins.')

# Add new coin (Admin only)
def add_coin(update, context):
    if str(update.message.chat_id) != ADMIN_ID:
        update.message.reply_text("Unauthorized access.")
        return
    if len(context.args) != 2:
        update.message.reply_text("Usage: /addcoin <symbol> <coin_id>")
        return
    symbol, coin_id = context.args
    supported_coins[symbol] = coin_id
    update.message.reply_text(f"Coin {symbol.upper()} added successfully.")

# Remove coin (Admin only)
def remove_coin(update, context):
    if str(update.message.chat_id) != ADMIN_ID:
        update.message.reply_text("Unauthorized access.")
        return
    if len(context.args) != 1:
        update.message.reply_text("Usage: /removecoin <symbol>")
        return
    symbol = context.args[0]
    if symbol in supported_coins:
        del supported_coins[symbol]
        update.message.reply_text(f"Coin {symbol.upper()} removed successfully.")
    else:
        update.message.reply_text(f"Coin {symbol.upper()} not found.")

# Broadcast a message (Admin only)
def broadcast(update, context):
    if str(update.message.chat_id) != ADMIN_ID:
        update.message.reply_text("Unauthorized access.")
        return
    if not context.args:
        update.message.reply_text("Usage: /broadcast <message>")
        return
    message = ' '.join(context.args)
    for user_id in users:
        context.bot.send_message(chat_id=user_id, text=f"Broadcast: {message}")

# Fetch the current price of a coin
def current_price(update, context):
    if len(context.args) != 1:
        update.message.reply_text("Usage: /price <symbol>")
        return
    symbol = context.args[0].lower()
    if symbol not in supported_coins:
        update.message.reply_text("Coin not supported. Use /help to see available coins.")
        return
    coin_id = supported_coins[symbol]
    url = f'https://api.coingecko.com/api/v3/simple/price'
    params = {'ids': coin_id, 'vs_currencies': 'usd'}
    response = requests.get(url, params=params)
    data = response.json()
    if coin_id in data:
        price = data[coin_id]['usd']
        update.message.reply_text(f"The current price of {symbol.upper()} is ${price}")
    else:
        update.message.reply_text(f"Error: Could not fetch current price for {symbol.upper()}.")

# Main function to set up the bot
def main():
    updater = Updater(API_KEY, use_context=True)
    dp = updater.dispatcher

    # Add command handlers
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(CommandHandler("addcoin", add_coin, pass_args=True))
    dp.add_handler(CommandHandler("removecoin", remove_coin, pass_args=True))
    dp.add_handler(CommandHandler("broadcast", broadcast, pass_args=True))
    dp.add_handler(CommandHandler("price", current_price, pass_args=True))

    # Default message handler
    dp.add_handler(MessageHandler(Filters.text, handle_query))

    # Start the bot
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

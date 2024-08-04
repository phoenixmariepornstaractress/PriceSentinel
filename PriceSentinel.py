import time
import json
import smtplib
import logging
import sqlite3
import schedule
import requests
from email.mime.text import MIMEText
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from twilio.rest import Client
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import IsolationForest
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from keras_tuner import RandomSearch, HyperParameters
from textblob import TextBlob
import gym
from gym import spaces
from stable_baselines3 import PPO

# Load user configurations
def load_config():
    with open('config.json') as f:
        return json.load(f)

config = load_config()

# Configuration
DATABASE = 'price_tracker.db'
LOG_FILE = 'price_check_log.txt'
EMAIL_SENDER = config['email_sender']
EMAIL_RECIPIENT = config['email_recipient']
SMTP_SERVER = config['smtp_server']
SMTP_PORT = config['smtp_port']
EMAIL_PASSWORD = config['email_password']
TWILIO_ACCOUNT_SID = config['twilio_account_sid']
TWILIO_AUTH_TOKEN = config['twilio_auth_token']
TWILIO_PHONE_NUMBER = config['twilio_phone_number']
RECIPIENT_PHONE_NUMBER = config['recipient_phone_number']
SLACK_TOKEN = config['slack_token']
SLACK_CHANNEL = config['slack_channel']
PRICE_CHANGE_THRESHOLD = config['price_change_threshold']  # e.g., 0.05 for 5%

# Set up logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database functions
def create_table():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            url TEXT PRIMARY KEY,
            name TEXT,
            desired_price REAL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS price_history (
            url TEXT,
            date TEXT,
            price REAL,
            PRIMARY KEY (url, date)
        )
    ''')
    conn.commit()
    conn.close()

def insert_product(url, name, desired_price):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO products (url, name, desired_price) VALUES (?, ?, ?)
    ''', (url, name, desired_price))
    conn.commit()
    conn.close()

def get_products():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT url, name, desired_price FROM products')
    products = c.fetchall()
    conn.close()
    return products

def insert_price_history(url, date, price):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO price_history (url, date, price) VALUES (?, ?, ?)
    ''', (url, date, price))
    conn.commit()
    conn.close()

def get_price_history(url):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT date, price FROM price_history WHERE url = ? ORDER BY date', (url,))
    history = c.fetchall()
    conn.close()
    return history

# Web scraping and notification functions
def check_price(url):
    try:
        driver = webdriver.Chrome()
        driver.get(url)
        time.sleep(5)  # Wait for page to load
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()
        
        price_element = soup.find('span', {'class': 'price-tag'})  # Adjust this as needed
        if not price_element:
            logging.error(f'Price element not found for {url}')
            return None
        
        price_text = price_element.get_text(strip=True).replace('$', '')
        return float(price_text)
    except Exception as e:
        logging.error(f'Error checking price for {url}: {e}')
        return None

def send_email_notification(product_name, current_price, url):
    subject = 'Price Drop Alert'
    body = f'The price of {product_name} has dropped to ${current_price}.\nCheck it out here: {url}'

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECIPIENT

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())
            logging.info(f'Email notification sent for {product_name}. Current price is ${current_price}')
    except Exception as e:
        logging.error(f'Error sending email notification: {e}')

def send_sms_notification(product_name, current_price, url):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    message = f'The price of {product_name} has dropped to ${current_price}.\nCheck it out here: {url}'

    try:
        client.messages.create(body=message, from_=TWILIO_PHONE_NUMBER, to=RECIPIENT_PHONE_NUMBER)
        logging.info(f'SMS notification sent for {product_name}. Current price is ${current_price}')
    except Exception as e:
        logging.error(f'Error sending SMS notification: {e}')

def send_slack_notification(product_name, current_price, url):
    client = WebClient(token=SLACK_TOKEN)
    message = f'The price of {product_name} has dropped to ${current_price}.\nCheck it out here: {url}'

    try:
        client.chat_postMessage(channel=SLACK_CHANNEL, text=message)
        logging.info(f'Slack notification sent for {product_name}. Current price is ${current_price}')
    except SlackApiError as e:
        logging.error(f'Error sending Slack notification: {e}')

def send_webhook_notification(product_name, current_price, url):
    # Replace with your webhook URL
    webhook_url = config['webhook_url']
    payload = {
        'text': f'The price of {product_name} has dropped to ${current_price}.\nCheck it out here: {url}'
    }
    try:
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 200:
            logging.info(f'Webhook notification sent for {product_name}. Current price is ${current_price}')
        else:
            logging.error(f'Error sending webhook notification: {response.status_code}')
    except Exception as e:
        logging.error(f'Error sending webhook notification: {e}')

# Advanced LSTM Model with Hyperparameter Tuning
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), return_sequences=True, input_shape=(10, 1)))
    model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32)))
    model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=0.0001, max_value=0.01, step=0.0001)), loss=MeanSquaredError())
    return model

def train_lstm_model(url):
    history = get_price_history(url)
    if len(history) < 10:
        return None
    
    df = pd.DataFrame(history, columns=['date', 'price'])
    df['date'] = pd.to_datetime(df['date'])
    df['date_ordinal'] = df['date'].apply(lambda x: x.toordinal())
    
    df.set_index('date_ordinal', inplace=True)
    data = df[['price']].values

    train_data = data[:-10]
    test_data = data[-10:]
    
    def create_dataset(data, time_step=1):
        X, Y = [], []
        for i in range(len(data) - time_step - 1):
            a = data[i:(i + time_step), 0]
            X.append(a)
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)

    time_step = 10
    X_train, y_train = create_dataset(train_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='price_prediction'
    )

    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

    best_model = tuner.get_best_models(num_models=1)[0]

    X_test, y_test = create_dataset(test_data, time_step)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    predictions = best_model.predict(X_test)
    return predictions, y_test

# Price Change Anomaly Detection
def detect_anomalies(url):
    history = get_price_history(url)
    if len(history) < 10:
        return []

    df = pd.DataFrame(history, columns=['date', 'price'])
    df['date'] = pd.to_datetime(df['date'])
    df['date_ordinal'] = df['date'].apply(lambda x: x.toordinal())
    
    isolation_forest = IsolationForest(contamination=0.1)
    df['anomaly'] = isolation_forest.fit_predict(df[['date_ordinal', 'price']])

    anomalies = df[df['anomaly'] == -1]
    return anomalies

# Reinforcement Learning Environment for Optimal Price Tracking
class PriceTrackingEnv(gym.Env):
    def __init__(self, url):
        super(PriceTrackingEnv, self).__init__()
        self.url = url
        self.history = get_price_history(url)
        self.current_step = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return np.array([self.history[self.current_step][1]], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.history) - 1

        if action == 1:  # Buy action
            reward = -self.history[self.current_step][1]
        else:  # Hold action
            reward = 0

        obs = np.array([self.history[self.current_step][1]], dtype=np.float32)
        return obs, reward, done, {}

def train_reinforcement_learning_model(url):
    env = PriceTrackingEnv(url)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    return model

# Main price check and notification function
def price_check():
    products = get_products()
    for url, name, desired_price in products:
        current_price = check_price(url)
        if current_price is not None:
            insert_price_history(url, datetime.now().strftime('%Y-%m-%d'), current_price)
            if current_price <= desired_price:
                send_email_notification(name, current_price, url)
                send_sms_notification(name, current_price, url)
                send_slack_notification(name, current_price, url)
                send_webhook_notification(name, current_price, url)
            else:
                price_history = get_price_history(url)
                if len(price_history) >= 2:
                    previous_price = price_history[-2][1]
                    price_change = abs(current_price - previous_price) / previous_price
                    if price_change >= PRICE_CHANGE_THRESHOLD:
                        logging.info(f'Significant price change detected for {name}. Previous price: ${previous_price}, Current price: ${current_price}')

            anomalies = detect_anomalies(url)
            if not anomalies.empty:
                logging.info(f'Anomalies detected for {name}: {anomalies}')

            predictions, y_test = train_lstm_model(url)
            if predictions is not None:
                plt.plot(predictions, label='Predicted Prices')
                plt.plot(y_test, label='Actual Prices')
                plt.legend()
                plt.show()

            model = train_reinforcement_learning_model(url)
            logging.info(f'Reinforcement learning model trained for {name}')

# Schedule the price check function
schedule.every().hour.do(price_check)

# Create tables
create_table()

# Run the scheduler
while True:
    schedule.run_pending()
    time.sleep(1)

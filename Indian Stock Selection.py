#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Fetch Historical Data
tickers = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'KOTAKBANK.NS',
    'SBIN.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS', 'ITC.NS', 'HCLTECH.NS', 'AXISBANK.NS',
    'LT.NS', 'MARUTI.NS', 'HDFCLIFE.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS',
    'M&M.NS', 'BAJAJFINSV.NS', 'TECHM.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS',
    'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'GRASIM.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'ICICIGI.NS',
    'INDUSINDBK.NS', 'IOC.NS', 'JSWSTEEL.NS', 'POWERGRID.NS', 'SBILIFE.NS', 'SHREECEM.NS', 'TATASTEEL.NS', 'UPL.NS',
    'VEDL.NS', 'ADANIENT.NS', 'ADANITRANS.NS', 'ONGC.NS'
]
stock_data = yf.download(tickers, period="2y")['Close']

# Step 2: Prepare the Data 
window_size = 20
horizon = 1

def create_rolling_windows(data, window_size, horizon):
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

features, labels = {}, {}
for ticker in tickers:
    X, y = create_rolling_windows(stock_data[ticker].values, window_size, horizon)
    features[ticker] = X
    labels[ticker] = y

# Step 3: Define and Train the Model
def build_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

models = {}
for ticker in tickers:
    X_train, X_test, y_train, y_test = train_test_split(features[ticker], labels[ticker], test_size=0.2, random_state=42)
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    models[ticker] = model

# Step 4: Trading Strategy Simulation
initial_investment = 100000
investment_per_stock = initial_investment / len(tickers)
portfolio_value = initial_investment

for ticker in tickers:
    # Use the last window of data to make a prediction
    last_window = features[ticker][-1].reshape(1, -1)
    predicted_price = models[ticker].predict(last_window)[0][0]
    current_price = stock_data[ticker].iloc[-1]
    
    # Simulate a buy if the model predicts a higher future price, otherwise hold
    if predicted_price > current_price:
        estimated_shares = investment_per_stock / current_price
        # Simulate selling at the predicted price after a week
        portfolio_value += estimated_shares * (predicted_price - current_price)

total_return = (portfolio_value - initial_investment) / initial_investment * 100

print(f"Final Portfolio Value: ${portfolio_value:.2f}")
print(f"Total Return: {total_return:.2f}%")


# In[10]:


from sklearn.model_selection import train_test_split

##here we are preparing and for training and test sets

features = ['MA20', 'Close'] 
X = data[features]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:





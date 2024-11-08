#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Fetch Historical Data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
stock_data = yf.download(tickers, period="1y")['Adj Close']

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


# ##This Simple model does not give returns withch would be worthwhile given taxes and other considerations, therefore lets try with single stock

# In[22]:


# historical data
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')


# In[23]:


window_size = 20  #window size

# Calculate rolling window features  moving averages
data['MA20'] = data['Close'].rolling(window=window_size).mean()

# Calculate future returns and direction for classification
data['Future Close'] = data['Close'].shift(-1)
data['Return'] = (data['Future Close'] - data['Close']) / data['Close']
data['Target'] = (data['Return'] > 0).astype(int)

# Drop NaN values
data.dropna(inplace=True)


# In[ ]:


# Building the model


# In[24]:


from sklearn.model_selection import train_test_split

##here we are preparing and for training and test sets

features = ['MA20', 'Close'] 
X = data[features]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

##next define and train our neural network with ReLu and sigmoid activation functions.

model = Sequential([
    Dense(64, input_shape=(len(features),), activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


# In[40]:


#Retesting


# In[30]:


from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Data Collection
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')

# Step 2: Feature Engineering
# Calculate monthly returns
data['Monthly Return'] = data['Adj Close'].pct_change().resample('M').agg(lambda x: (x + 1).prod() - 1)

# Create moving averages as features
for window in [3, 6, 12]:  # 3, 6, and 12 months
    data[f'SMA_{window}M'] = data['Adj Close'].rolling(window=window*21).mean().resample('M').last()  # Approx. 21 trading days in a month

data.dropna(inplace=True)

# The target variable is the sign of the next month's return (1 for positive, 0 for negative)
data['Target'] = (data['Monthly Return'].shift(-1) > 0).astype(int)

# Step 3: Model Design
model = Sequential([
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Data Preparation
# Prepare the features and target
X = data[['SMA_3M', 'SMA_6M', 'SMA_12M']]
y = data['Target']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Model Training
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)

# Step 6: Evaluation
# Evaluate the model
performance = model.evaluate(X_test, y_test)

print(f"Test Loss: {performance[0]}, Test Accuracy: {performance[1]}")


# ##using one stock alone does not provide the ideal returns, therefore revert back to original strategy with 5 stocks within asm industry
# 

# In[67]:


tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
stock_data = yf.download(tickers, period="5y")['Adj Close']

# Step 2: Prepare the Data 
window_size = 5
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
initial_investment = 1000000
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


# In[ ]:


#test using stocks from different industry 


# In[83]:


tickers = ['F', 'TSLA', 'GM', 'STLA','TM']
stock_data = yf.download(tickers, period="1y")['Adj Close']

# --- Step 2: Prepare the Data ---
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

# --- Step 3: Define and Train the Model ---
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

# --- Step 4: Trading Strategy Simulation ---
initial_investment = 10000
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


# In[84]:


# Initialize investment tracking
investments = {ticker: investment_per_stock for ticker in tickers}
portfolio_value = initial_investment

for ticker in tickers:
    current_investment = investments[ticker]
    for week in range(weeks_in_month):
        # Assuming you have a way to select or aggregate data for weekly predictions
        # For simplicity, this example assumes you have weekly features ready
        weekly_window_index = -weeks_in_month + week
        if weekly_window_index >= len(features[ticker]):
            break
        
        weekly_window = features[ticker][weekly_window_index].reshape(1, -1)
        predicted_weekly_price = models[ticker].predict(weekly_window)[0][0]
        current_weekly_price = stock_data[ticker].iloc[weekly_window_index]  # Adjust for weekly
        
        # Make investment decision based on the model's weekly prediction
        if predicted_weekly_price > current_weekly_price:
            # The model predicts an increase in price, so we "buy" or hold
            estimated_shares = current_investment / current_weekly_price
            potential_returns = estimated_shares * (predicted_weekly_price - current_weekly_price)
            
            # Update current investment with potential returns
            current_investment += potential_returns
            
    # Update final investments after the month
    investments[ticker] = current_investment

# Calculate final portfolio value
final_portfolio_value = sum(investments.values())
total_return = (final_portfolio_value - initial_investment) / initial_investment * 100

print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
print(f"Total Return: {total_return:.2f}%")


# In[82]:


# Test with 5 stocks from different industry Pharma LLY, JNJ, MRK, PFE, SNY


# In[91]:


tickers = ['LLY','JNJ','MRK','PFE','SNY']
stock_data = yf.download(tickers, period="5y")['Adj Close']

# 5 YEAR DATA THIS TIME

# --- Step 2: Prepare the Data ---
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

# --- Step 3: Define and Train the Model ---
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

# --- Step 4: Trading Strategy Simulation ---
initial_investment = 10000
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

num_simulations = 5

# Function to run a single simulation
def run_simulation(seed):
    np.random.seed(seed)  # Ensure reproducibility for each simulation
    models = {}
    for ticker in tickers:
        X_train, X_test, y_train, y_test = train_test_split(features[ticker], labels[ticker], test_size=0.2, random_state=seed)
        model = build_model(X_train.shape[1])
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        models[ticker] = model

    portfolio_value = initial_investment
    for ticker in tickers:
        last_window = features[ticker][-1].reshape(1, -1)
        predicted_price = models[ticker].predict(last_window)[0][0]
        current_price = stock_data[ticker].iloc[-1]
        
        if predicted_price > current_price:
            estimated_shares = investment_per_stock / current_price
            portfolio_value += estimated_shares * (predicted_price - current_price)

    total_return = (portfolio_value - initial_investment) / initial_investment * 100
    return total_return

# Run multiple simulations and print results
for i in range(num_simulations):
    total_return = run_simulation(seed=i+42)  # Using different seeds for each simulation
    print(f"Simulation {i+1}: Final Portfolio Value: ${initial_investment + (initial_investment * total_return / 100):.2f}, Total Return: {total_return:.2f}%")


# In[90]:


# Next test with Bank industry, using JPM, BAC, WFC, C, SOFI using 5 year data, this time 10 simulation


# In[92]:


tickers = ['JPM','BAC','WFC','C','SOFI']
stock_data = yf.download(tickers, period="5y")['Adj Close']

# 5 YEAR DATA THIS TIME

# --- Step 2: Prepare the Data ---
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

# --- Step 3: Define and Train the Model ---
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

# --- Step 4: Trading Strategy Simulation ---
initial_investment = 10000
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

num_simulations = 5

# Function to run a single simulation
def run_simulation(seed):
    np.random.seed(seed)  # Ensure reproducibility for each simulation
    models = {}
    for ticker in tickers:
        X_train, X_test, y_train, y_test = train_test_split(features[ticker], labels[ticker], test_size=0.2, random_state=seed)
        model = build_model(X_train.shape[1])
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        models[ticker] = model

    portfolio_value = initial_investment
    for ticker in tickers:
        last_window = features[ticker][-1].reshape(1, -1)
        predicted_price = models[ticker].predict(last_window)[0][0]
        current_price = stock_data[ticker].iloc[-1]
        
        if predicted_price > current_price:
            estimated_shares = investment_per_stock / current_price
            portfolio_value += estimated_shares * (predicted_price - current_price)

    total_return = (portfolio_value - initial_investment) / initial_investment * 100
    return total_return

# Run multiple simulations and print results
for i in range(num_simulations):
    total_return = run_simulation(seed=i+42)  # Using different seeds for each simulation
    print(f"Simulation {i+1}: Final Portfolio Value: ${initial_investment + (initial_investment * total_return / 100):.2f}, Total Return: {total_return:.2f}%")


# In[93]:


#stimulate again using tech sector different stocks:PIN, Z, NIO, SPOT, SNAP


# In[95]:


tickers = ['PIN','Z','NIO','SPOT','SNAP']
stock_data = yf.download(tickers, period="1y")['Adj Close']

# 1 YEAR DATA THIS TIME

# --- Step 2: Prepare the Data ---
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

# --- Step 3: Define and Train the Model ---
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

# --- Step 4: Trading Strategy Simulation ---
initial_investment = 10000
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

num_simulations = 5

# Function to run a single simulation
def run_simulation(seed):
    np.random.seed(seed)  # Ensure reproducibility for each simulation
    models = {}
    for ticker in tickers:
        X_train, X_test, y_train, y_test = train_test_split(features[ticker], labels[ticker], test_size=0.2, random_state=seed)
        model = build_model(X_train.shape[1])
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        models[ticker] = model

    portfolio_value = initial_investment
    for ticker in tickers:
        last_window = features[ticker][-1].reshape(1, -1)
        predicted_price = models[ticker].predict(last_window)[0][0]
        current_price = stock_data[ticker].iloc[-1]
        
        if predicted_price > current_price:
            estimated_shares = investment_per_stock / current_price
            portfolio_value += estimated_shares * (predicted_price - current_price)

    total_return = (portfolio_value - initial_investment) / initial_investment * 100
    return total_return

# Run multiple simulations and print results
for i in range(num_simulations):
    total_return = run_simulation(seed=i+42)  # Using different seeds for each simulation
    print(f"Simulation {i+1}: Final Portfolio Value: ${initial_investment + (initial_investment * total_return / 100):.2f}, Total Return: {total_return:.2f}%")


# In[100]:


tickers = ['TSLA','NVDA','META']
stock_data = yf.download(tickers, period="5y")['Adj Close']

# 5 YEAR DATA THIS TIME

# --- Step 2: Prepare the Data ---
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

# --- Step 3: Define and Train the Model ---
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

# --- Step 4: Trading Strategy Simulation ---
initial_investment = 10000
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

##50 simulation
num_simulations = 50

# Function to run a single simulation
def run_simulation(seed):
    np.random.seed(seed)  # Ensure reproducibility for each simulation
    models = {}
    for ticker in tickers:
        X_train, X_test, y_train, y_test = train_test_split(features[ticker], labels[ticker], test_size=0.2, random_state=seed)
        model = build_model(X_train.shape[1])
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        models[ticker] = model

    portfolio_value = initial_investment
    for ticker in tickers:
        last_window = features[ticker][-1].reshape(1, -1)
        predicted_price = models[ticker].predict(last_window)[0][0]
        current_price = stock_data[ticker].iloc[-1]
        
        if predicted_price > current_price:
            estimated_shares = investment_per_stock / current_price
            portfolio_value += estimated_shares * (predicted_price - current_price)

    total_return = (portfolio_value - initial_investment) / initial_investment * 100
    return total_return

total_returns = []
final_portfolio_values = []

for i in range(num_simulations):
    total_return = run_simulation(seed=i+42)  # Using different seeds for each simulation
    final_portfolio_value = initial_investment + (initial_investment * total_return / 100)
    total_returns.append(total_return)
    final_portfolio_values.append(final_portfolio_value)
    print(f"Simulation {i+1}: Final Portfolio Value: ${final_portfolio_value:.2f}, Total Return: {total_return:.2f}%")

    # Plot Total Returns
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(total_returns, bins=5, alpha=0.7, color='blue')
plt.title('Distribution of Total Returns')
plt.xlabel('Total Return (%)')
plt.ylabel('Frequency')

# Plot Final Portfolio Values
plt.subplot(1, 2, 2)
plt.hist(final_portfolio_values, bins=5, alpha=0.7, color='green')
plt.title('Distribution of Final Portfolio Values')
plt.xlabel('Final Portfolio Value ($)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[102]:


## now test with tech stocks only as returns are most siginificant


# In[103]:


tickers = ['NVDA','META','PIN','Z','NIO','SPOT','SNAP']
stock_data = yf.download(tickers, period="5y")['Adj Close']

# 5 YEAR DATA THIS TIME

# --- Step 2: Prepare the Data ---
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

# --- Step 3: Define and Train the Model ---
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

# --- Step 4: Trading Strategy Simulation ---
initial_investment = 10000
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

##50 simulation
num_simulations = 50

# Function to run a single simulation
def run_simulation(seed):
    np.random.seed(seed)  # Ensure reproducibility for each simulation
    models = {}
    for ticker in tickers:
        X_train, X_test, y_train, y_test = train_test_split(features[ticker], labels[ticker], test_size=0.2, random_state=seed)
        model = build_model(X_train.shape[1])
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        models[ticker] = model

    portfolio_value = initial_investment
    for ticker in tickers:
        last_window = features[ticker][-1].reshape(1, -1)
        predicted_price = models[ticker].predict(last_window)[0][0]
        current_price = stock_data[ticker].iloc[-1]
        
        if predicted_price > current_price:
            estimated_shares = investment_per_stock / current_price
            portfolio_value += estimated_shares * (predicted_price - current_price)

    total_return = (portfolio_value - initial_investment) / initial_investment * 100
    return total_return

total_returns = []
final_portfolio_values = []

for i in range(num_simulations):
    total_return = run_simulation(seed=i+42)  # Using different seeds for each simulation
    final_portfolio_value = initial_investment + (initial_investment * total_return / 100)
    total_returns.append(total_return)
    final_portfolio_values.append(final_portfolio_value)
    print(f"Simulation {i+1}: Final Portfolio Value: ${final_portfolio_value:.2f}, Total Return: {total_return:.2f}%")

    # Plot Total Returns
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(total_returns, bins=5, alpha=0.7, color='blue')
plt.title('Distribution of Total Returns')
plt.xlabel('Total Return (%)')
plt.ylabel('Frequency')

# Plot Final Portfolio Values
plt.subplot(1, 2, 2)
plt.hist(final_portfolio_values, bins=5, alpha=0.7, color='green')
plt.title('Distribution of Final Portfolio Values')
plt.xlabel('Final Portfolio Value ($)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[104]:


## finally take a 5 stocks use 10 years of data, with 50 simulations to test the model


# In[106]:


tickers = ['NVDA','AAPL','TSLA','AMZN','JPM']
stock_data = yf.download(tickers, period="10y")['Adj Close']

# 10 YEAR DATA THIS TIME

# --- Step 2: Prepare the Data ---
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

# --- Step 3: Define and Train the Model ---
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

# --- Step 4: Trading Strategy Simulation ---
initial_investment = 10000
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

##50 simulation
num_simulations = 50

# Function to run a single simulation
def run_simulation(seed):
    np.random.seed(seed)  # Ensure reproducibility for each simulation
    models = {}
    for ticker in tickers:
        X_train, X_test, y_train, y_test = train_test_split(features[ticker], labels[ticker], test_size=0.2, random_state=seed)
        model = build_model(X_train.shape[1])
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        models[ticker] = model

    portfolio_value = initial_investment
    for ticker in tickers:
        last_window = features[ticker][-1].reshape(1, -1)
        predicted_price = models[ticker].predict(last_window)[0][0]
        current_price = stock_data[ticker].iloc[-1]
        
        if predicted_price > current_price:
            estimated_shares = investment_per_stock / current_price
            portfolio_value += estimated_shares * (predicted_price - current_price)

    total_return = (portfolio_value - initial_investment) / initial_investment * 100
    return total_return

total_returns = []
final_portfolio_values = []

for i in range(num_simulations):
    total_return = run_simulation(seed=i+42)  # Using different seeds for each simulation
    final_portfolio_value = initial_investment + (initial_investment * total_return / 100)
    total_returns.append(total_return)
    final_portfolio_values.append(final_portfolio_value)
    print(f"Simulation {i+1}: Final Portfolio Value: ${final_portfolio_value:.2f}, Total Return: {total_return:.2f}%")

    # Plot Total Returns
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(total_returns, bins=5, alpha=0.7, color='blue')
plt.title('Distribution of Total Returns')
plt.xlabel('Total Return (%)')
plt.ylabel('Frequency')

# Plot Final Portfolio Values
plt.subplot(1, 2, 2)
plt.hist(final_portfolio_values, bins=5, alpha=0.7, color='green')
plt.title('Distribution of Final Portfolio Values')
plt.xlabel('Final Portfolio Value ($)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[ ]:


##with out high return stocks, test if higher initial investment generates higher return, 100k 1y year history


# In[107]:


tickers = ['NVDA','META','PIN','Z','NIO','SPOT','SNAP']
stock_data = yf.download(tickers, period="1y")['Adj Close']

# 1 YEAR DATA THIS TIME

# --- Step 2: Prepare the Data ---
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

# --- Step 3: Define and Train the Model ---
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

# --- Step 4: Trading Strategy Simulation ---
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

##20 simulation
num_simulations = 20

# Function to run a single simulation
def run_simulation(seed):
    np.random.seed(seed)  # Ensure reproducibility for each simulation
    models = {}
    for ticker in tickers:
        X_train, X_test, y_train, y_test = train_test_split(features[ticker], labels[ticker], test_size=0.2, random_state=seed)
        model = build_model(X_train.shape[1])
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        models[ticker] = model

    portfolio_value = initial_investment
    for ticker in tickers:
        last_window = features[ticker][-1].reshape(1, -1)
        predicted_price = models[ticker].predict(last_window)[0][0]
        current_price = stock_data[ticker].iloc[-1]
        
        if predicted_price > current_price:
            estimated_shares = investment_per_stock / current_price
            portfolio_value += estimated_shares * (predicted_price - current_price)

    total_return = (portfolio_value - initial_investment) / initial_investment * 100
    return total_return

total_returns = []
final_portfolio_values = []

for i in range(num_simulations):
    total_return = run_simulation(seed=i+42)  # Using different seeds for each simulation
    final_portfolio_value = initial_investment + (initial_investment * total_return / 100)
    total_returns.append(total_return)
    final_portfolio_values.append(final_portfolio_value)
    print(f"Simulation {i+1}: Final Portfolio Value: ${final_portfolio_value:.2f}, Total Return: {total_return:.2f}%")

    # Plot Total Returns
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(total_returns, bins=5, alpha=0.7, color='blue')
plt.title('Distribution of Total Returns')
plt.xlabel('Total Return (%)')
plt.ylabel('Frequency')

# Plot Final Portfolio Values
plt.subplot(1, 2, 2)
plt.hist(final_portfolio_values, bins=5, alpha=0.7, color='green')
plt.title('Distribution of Final Portfolio Values')
plt.xlabel('Final Portfolio Value ($)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[ ]:





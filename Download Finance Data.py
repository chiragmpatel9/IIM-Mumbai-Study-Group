#!/usr/bin/env python
# coding: utf-8

# In[9]:


import yfinance as yf
import pandas as pd

# Function to download stock data and calculate returns
def download_stock_data(ticker, start_date, end_date, filename):
    # Download the stock data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Calculate daily returns based on the 'Adj Close' column
    data['Daily Return'] = data['Adj Close'].pct_change()
    
    
    data.to_csv(filename)
    
    return data

# Define the stock/index ticker, start and end dates, and the filename to save
ticker = 'TSLA'  # Example: 'AAPL' for Apple, or '^GSPC' for S&P 500
start_date = '2024-01-05'
end_date = '2024-11-05'
filename = 'stock_data_tsla.csv'

# Download the data and save it to a CSV
data = download_stock_data(ticker, start_date, end_date, filename)

# Display the first few rows of the data
print(data.head())


# In[7]:


def download_stock_data(ticker, start_date, end_date, filename):
    # Download the stock data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Calculate daily returns based on the 'Adj Close' column
    data['Daily Return'] = data['Adj Close'].pct_change()
    
    # Save the data as a CSV file
    #data.to_csv(filename)
    
    return data

# Define the stock/index ticker, start and end dates, and the filename to save
ticker = '^GSPC'  # Example: 'AAPL' for Apple, or '^GSPC' for S&P 500
start_date = '2024-09-01'
end_date = '2024-09-06'
#filename = 'stock_data.csv'

# Download the data and save it to a CSV
#data = download_stock_data(ticker, start_date, end_date, filename)

# Display the first few rows of the data
print(data.head())


# In[8]:


def download_stock_data(ticker, start_date, end_date, filename):
    # Download the stock data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Calculate daily returns based on the 'Adj Close' column
    data['Daily Return'] = data['Adj Close'].pct_change()
    
    # Save the data as a CSV file
    #data.to_csv(filename)
    
    return data

# Define the stock/index ticker, start and end dates, and the filename to save
ticker = '^GSPC'  # Example: 'AAPL' for Apple, or '^GSPC' for S&P 500
start_date = '2024-09-01'
end_date = '2024-09-06'
#filename = 'stock_data.csv'

# Download the data and save it to a CSV
#data = download_stock_data(ticker, start_date, end_date, filename)

# Display the first few rows of the data
print(data.head())


# In[ ]:





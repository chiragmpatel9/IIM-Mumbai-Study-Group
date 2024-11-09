#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import yfinance as yf

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

def get_sp500_tickers():
    # This uses a known URL to fetch S&P 500 company list
    table = pd.read_html(url)
    sp500_df = table[0]
    return sp500_df['Symbol'].tolist()

def snp500_names():
    table = pd.read_html(url)
    snp500_df = table[0]
    return snp500_df['Security'].tolist()

sp500_tickers = get_sp500_tickers()
snp500comp = snp500_names()

# Example tickers, replace with your actual buy signals
tickers = sp500_tickers


stock_data = download_data(sp500_tickers)


# In[5]:


import matplotlib.pyplot as plt

data = {}
for ticker in sp500_tickers:
    stock_data = yf.download(ticker, period='1d', interval='5m')
    if not stock_data.empty:
        data[ticker] = stock_data['Close']

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Calculate the percentage change from the previous day's close
previous_close = df.iloc[0]
percentage_change = (df - previous_close) / previous_close * 100



numb=5

# Get the top 5 performing stocks based on the percentage change
end_of_day_change = percentage_change.iloc[-1]
top_5_tickers = end_of_day_change.nlargest(numb).index



# Plot the line chart for the day's performance of top 5 stocks
plt.figure(figsize=(10, 6))
for ticker in top_5_tickers:
    plt.plot(percentage_change.index, percentage_change[ticker], label=ticker)

plt.title(f"Top {numb} Performing S&P 500 - Percentage Change from prior day")
plt.xlabel("Time")
plt.ylabel("Percentage Change (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


top_5_tickers, percentage_change


# In[ ]:





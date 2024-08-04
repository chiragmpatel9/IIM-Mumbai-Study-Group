#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preparation
data = {
    'Year': [2019, 2020, 2021, 2022, 2023],
    'Revenue': [386064, 469822, 513983, 574785, 604334],
    'Gross Profit': [152757, 197478, 225152, 270046, 290341],
    'Net Income': [21331, 33364, 30524, -2722, 44419],
    'Operating Expenses': [129858, 172599, 211804, 233194, 235965],
    'Free Cash Flow': [25924, -14726, -16893, 32217, 48340]
}

df = pd.DataFrame(data)

# Revenue and Gross Profit
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Revenue', data=df, marker='o', label='Revenue')
sns.lineplot(x='Year', y='Gross Profit', data=df, marker='o', label='Gross Profit')
plt.title('Amazon Revenue and Gross Profit (2019-2023)')
plt.xlabel('Year')
plt.ylabel('Millions USD')
plt.legend()
plt.grid(True)
plt.show()

# Net Income
plt.figure(figsize=(12, 6))
sns.barplot(x='Year', y='Net Income', data=df)
plt.title('Amazon Net Income (2019-2023)')
plt.xlabel('Year')
plt.ylabel('Millions USD')
plt.grid(True)
plt.show()

# Operating Expenses
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Operating Expenses', data=df, marker='o', label='Operating Expenses')
plt.title('Amazon Operating Expenses (2019-2023)')
plt.xlabel('Year')
plt.ylabel('Millions USD')
plt.legend()
plt.grid(True)
plt.show()

# Free Cash Flow
plt.figure(figsize=(12, 6))
sns.barplot(x='Year', y='Free Cash Flow', data=df)
plt.title('Amazon Free Cash Flow (2019-2023)')
plt.xlabel('Year')
plt.ylabel('Millions USD')
plt.grid(True)
plt.show()


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preparation
data = {
    'Year': [2019, 2020, 2021, 2022, 2023],
    'Revenue': [386064, 469822, 513983, 574785, 604334],
    'Gross Profit': [152757, 197478, 225152, 270046, 290341],
    'Net Income': [21331, 33364, 30524, -2722, 44419],
    'Operating Expenses': [129858, 172599, 211804, 233194, 235965],
    'Free Cash Flow': [25924, -14726, -16893, 32217, 48340],
    'EBITDA': [48079, 59312, 55269, 85515, 104049],
    'Gross Margin': [39.57, 42.03, 43.81, 46.98, 48.04],
    'Operating Margin': [5.93, 5.30, 2.60, 6.41, 9.00],
    'Profit Margin': [5.53, 7.10, -0.53, 5.29, 7.35]
}

df = pd.DataFrame(data)

# Enhanced Revenue and Gross Profit Plot
plt.figure(figsize=(14, 8))
sns.lineplot(x='Year', y='Revenue', data=df, marker='o', label='Revenue', color='blue')
sns.lineplot(x='Year', y='Gross Profit', data=df, marker='o', label='Gross Profit', color='green')
plt.title('Amazon Revenue and Gross Profit (2019-2023)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Millions USD', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Net Income with annotation for the negative year
plt.figure(figsize=(14, 8))
sns.barplot(x='Year', y='Net Income', data=df, palette='viridis')
plt.title('Amazon Net Income (2019-2023)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Millions USD', fontsize=14)
for index, value in enumerate(df['Net Income']):
    plt.text(index, value if value >= 0 else value - 4000, f'{value:.0f}', ha='center', color='black', fontsize=12)
plt.grid(True)
plt.show()

# Operating Expenses and EBITDA Plot
plt.figure(figsize=(14, 8))
sns.lineplot(x='Year', y='Operating Expenses', data=df, marker='o', label='Operating Expenses', color='red')
sns.lineplot(x='Year', y='EBITDA', data=df, marker='o', label='EBITDA', color='purple')
plt.title('Amazon Operating Expenses and EBITDA (2019-2023)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Millions USD', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Free Cash Flow with Heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(df.set_index('Year')[['Free Cash Flow']].T, annot=True, fmt='d', cmap='YlGnBu', cbar=False)
plt.title('Amazon Free Cash Flow (2019-2023)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Free Cash Flow', fontsize=14)
plt.show()

# Margins over Time
plt.figure(figsize=(14, 8))
sns.lineplot(x='Year', y='Gross Margin', data=df, marker='o', label='Gross Margin', color='orange')
sns.lineplot(x='Year', y='Operating Margin', data=df, marker='o', label='Operating Margin', color='pink')
sns.lineplot(x='Year', y='Profit Margin', data=df, marker='o', label='Profit Margin', color='cyan')
plt.title('Amazon Margins (2019-2023)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage (%)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()


# In[3]:


# Data Preparation
data = {
    'Year': [2019, 2020, 2021, 2022, 2023],
    'Revenue': [280522, 386064, 469822, 513983, 574785],
    'Gross_Profit': [114986, 152757, 197478, 225152, 270046],
    'Net_Income': [11588, 21331, 33364, -2722, 30425],
    'Operating_Income': [14541, 22899, 24879, 13348, 36852],
    'EBITDA': [36330, 48079, 59312, 55269, 85515],
    'Total_Assets': [225248, 321195, 420549, 462675, 527854],
    'Total_Equity': [62060, 93404, 138245, 146043, 201875],
    'Total_Debt': [77533, 104740, 139757, 169938, 161574],
    'Cash_Flow_Operations': [38514, 66064, 46327, 46752, 84946],
    'Capital_Expenditures': [16861, 40140, 61053, 63645, 52729]
}

df = pd.DataFrame(data)

# Financial Ratios Calculation
df['Current_Ratio'] = df['Total_Assets'] / df['Total_Equity']
df['Quick_Ratio'] = (df['Total_Assets'] - df['Capital_Expenditures']) / df['Total_Equity']
df['Debt_to_Equity'] = df['Total_Debt'] / df['Total_Equity']
df['EBITDA_Margin'] = df['EBITDA'] / df['Revenue']
df['Operating_Margin'] = df['Operating_Income'] / df['Revenue']
df['Net_Profit_Margin'] = df['Net_Income'] / df['Revenue']
df['Return_on_Assets'] = df['Net_Income'] / df['Total_Assets']
df['Return_on_Equity'] = df['Net_Income'] / df['Total_Equity']

# Advanced Visualizations
# Heatmap for Financial Ratios
plt.figure(figsize=(12, 8))
sns.heatmap(df[['Current_Ratio', 'Quick_Ratio', 'Debt_to_Equity', 'EBITDA_Margin', 'Operating_Margin', 'Net_Profit_Margin', 'Return_on_Assets', 'Return_on_Equity']].T, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Financial Ratios Heatmap (2019-2023)')
plt.show()

# EBITDA Margin and Operating Margin Over Time
plt.figure(figsize=(14, 8))
sns.lineplot(x='Year', y='EBITDA_Margin', data=df, marker='o', label='EBITDA Margin', color='blue')
sns.lineplot(x='Year', y='Operating_Margin', data=df, marker='o', label='Operating Margin', color='red')
plt.title('EBITDA Margin and Operating Margin Over Time (2019-2023)')
plt.xlabel('Year')
plt.ylabel('Margin (%)')
plt.legend()
plt.grid(True)
plt.show()

# Debt to Equity Ratio Over Time
plt.figure(figsize=(14, 8))
sns.barplot(x='Year', y='Debt_to_Equity', data=df, palette='viridis')
plt.title('Debt to Equity Ratio Over Time (2019-2023)')
plt.xlabel('Year')
plt.ylabel('Debt to Equity Ratio')
plt.grid(True)
plt.show()

# Cash Flow from Operations and Capital Expenditures Over Time
plt.figure(figsize=(14, 8))
sns.lineplot(x='Year', y='Cash_Flow_Operations', data=df, marker='o', label='Cash Flow from Operations', color='green')
sns.lineplot(x='Year', y='Capital_Expenditures', data=df, marker='o', label='Capital Expenditures', color='orange')
plt.title('Cash Flow from Operations and Capital Expenditures Over Time (2019-2023)')
plt.xlabel('Year')
plt.ylabel('Millions USD')
plt.legend()
plt.grid(True)
plt.show()

# Net Income and Revenue Over Time
plt.figure(figsize=(14, 8))
sns.lineplot(x='Year', y='Net_Income', data=df, marker='o', label='Net Income', color='purple')
sns.lineplot(x='Year', y='Revenue', data=df, marker='o', label='Revenue', color='cyan')
plt.title('Net Income and Revenue Over Time (2019-2023)')
plt.xlabel('Year')
plt.ylabel('Millions USD')
plt.legend()
plt.grid(True)
plt.show()


# In[4]:


import pandas as pd

# Data Preparation
data = {
    'Year': [2019, 2020, 2021, 2022, 2023],
    'Revenue': [280522, 386064, 469822, 513983, 574785],
    'Net_Income': [11588, 21331, 33364, -2722, 30425],
    'Total_Assets': [225248, 321195, 420549, 462675, 527854],
    'Total_Equity': [62060, 93404, 138245, 146043, 201875]
}

df = pd.DataFrame(data)


df['Profit_Margin'] = df['Net_Income'] / df['Revenue']
df['Asset_Turnover'] = df['Revenue'] / df['Total_Assets']
df['Financial_Leverage'] = df['Total_Assets'] / df['Total_Equity']
df['ROE'] = df['Profit_Margin'] * df['Asset_Turnover'] * df['Financial_Leverage']

# Print the DuPont analysis components and ROE
print(df[['Year', 'Profit_Margin', 'Asset_Turnover', 'Financial_Leverage', 'ROE']])


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plotting Profit Margin, Asset Turnover, and Financial Leverage
plt.figure(figsize=(14, 8))
sns.lineplot(x='Year', y='Profit_Margin', data=df, marker='o', label='Profit Margin', color='blue')
sns.lineplot(x='Year', y='Asset_Turnover', data=df, marker='o', label='Asset Turnover', color='green')
sns.lineplot(x='Year', y='Financial_Leverage', data=df, marker='o', label='Financial Leverage', color='red')
plt.title('DuPont Analysis Components (2019-2023)')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Plotting ROE
plt.figure(figsize=(14, 8))
sns.lineplot(x='Year', y='ROE', data=df, marker='o', label='ROE', color='purple')
plt.title('Return on Equity (ROE) (2019-2023)')
plt.xlabel('Year')
plt.ylabel('ROE')
plt.legend()
plt.grid(True)
plt.show()


# In[6]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data Preparation
data = {
    'Year': ['2019', '2020', '2021', '2022', '2023'],
    'Current_Ratio': [1.10, 1.05, 1.14, 0.95, 1.05],
    'Quick_Ratio': [0.86, 0.86, 0.91, 0.69, 0.81],
    'Debt_to_Equity': [1.25, 1.12, 1.01, 1.16, 0.80],
    'EBITDA_Margin': [0.13, 0.12, 0.13, 0.11, 0.15],
    'Operating_Margin': [0.05, 0.06, 0.05, 0.03, 0.06],
    'Net_Profit_Margin': [0.04, 0.05, 0.07, -0.01, 0.05],
    'Return_on_Assets': [0.05, 0.05, 0.07, 0.01, 0.06],
    'Return_on_Equity': [0.19, 0.23, 0.24, -0.02, 0.15]
}

df = pd.DataFrame(data)

# Creating the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.set_index('Year').T, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Financial Ratios Heatmap (2019-2023)')
plt.show()


# In[7]:


# Macroeconomics Assignment


# In[13]:


pip install mplfinance


# In[14]:


import mplfinance as mpf

# List of top 20 employers and their tickers
top_20_employers = {
    'Company': ['TCS', 'Infosys', 'Wipro', 'HCL Technologies', 'SBI', 'Reliance Industries', 'ICICI Bank', 
                'HDFC Bank', 'Bharti Airtel', 'L&T', 'Tech Mahindra', 'Axis Bank', 'Cognizant', 
                'Tata Steel', 'Mahindra & Mahindra', 'Maruti Suzuki', 'ONGC', 'Indian Oil', 'Coal India', 'NTPC'],
    'Ticker': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'SBIN.NS', 'RELIANCE.NS', 'ICICIBANK.NS', 
               'HDFCBANK.NS', 'BHARTIARTL.NS', 'LT.NS', 'TECHM.NS', 'AXISBANK.NS', 'CTSH', 
               'TATASTEEL.NS', 'M&M.NS', 'MARUTI.NS', 'ONGC.NS', 'IOC.NS', 'COALINDIA.NS', 'NTPC.NS']
}

# Fetch historical data for each company and create a candlestick chart
for company, ticker in zip(top_20_employers['Company'], top_20_employers['Ticker']):
    stock_data = yf.Ticker(ticker).history(period='10y')
    mpf.plot(stock_data, type='candle', title=f'10-Year Candlestick Chart of {company}', style='yahoo', volume=True)


# In[16]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# List of top 20 employers and their tickers
top_20_employers = {
    'Company': ['TCS', 'Infosys', 'Wipro', 'HCL Technologies', 'SBI', 'Reliance Industries', 'ICICI Bank', 
                'HDFC Bank', 'Bharti Airtel', 'L&T', 'Tech Mahindra', 'Axis Bank', 'Cognizant', 
                'Tata Steel', 'Mahindra & Mahindra', 'Maruti Suzuki', 'ONGC', 'Indian Oil', 'Coal India', 'NTPC'],
    'Ticker': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'SBIN.NS', 'RELIANCE.NS', 'ICICIBANK.NS', 
               'HDFCBANK.NS', 'BHARTIARTL.NS', 'LT.NS', 'TECHM.NS', 'AXISBANK.NS', 'CTSH', 
               'TATASTEEL.NS', 'M&M.NS', 'MARUTI.NS', 'ONGC.NS', 'IOC.NS', 'COALINDIA.NS', 'NTPC.NS']
}

# Initialize lists to hold market caps and returns
market_caps = []
returns = []

# Fetch market cap and 5-year percentage return for each company
for company, ticker in zip(top_20_employers['Company'], top_20_employers['Ticker']):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='5y')
    market_caps.append(stock.info['marketCap'])
    returns.append((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100)

# Create a DataFrame with market caps and returns
data = pd.DataFrame({
    'Company': top_20_employers['Company'],
    'Market Cap': market_caps,
    '5-Year Return': returns
})

# Normalize market cap for heatmap size representation
data['Market Cap Size'] = data['Market Cap'] / data['Market Cap'].max() * 100

# Create the heatmap
plt.figure(figsize=(12, 8))
heatmap_data = data.pivot('Company', 'Market Cap Size', '5-Year Return')
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='RdYlGn', linewidths=.5)

plt.title('Heatmap of Top 20 Employers by Market Cap Size and 5-Year Percentage Return')
plt.xlabel('Market Cap Size (Normalized)')
plt.ylabel('Company')
plt.show()


# In[17]:


import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# List of top 20 employers and their tickers
top_20_employers = {
    'Company': ['TCS', 'Infosys', 'Wipro', 'HCL Technologies', 'SBI', 'Reliance Industries', 'ICICI Bank', 
                'HDFC Bank', 'Bharti Airtel', 'L&T', 'Tech Mahindra', 'Axis Bank', 'Cognizant', 
                'Tata Steel', 'Mahindra & Mahindra', 'Maruti Suzuki', 'ONGC', 'Indian Oil', 'Coal India', 'NTPC'],
    'Ticker': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'SBIN.NS', 'RELIANCE.NS', 'ICICIBANK.NS', 
               'HDFCBANK.NS', 'BHARTIARTL.NS', 'LT.NS', 'TECHM.NS', 'AXISBANK.NS', 'CTSH', 
               'TATASTEEL.NS', 'M&M.NS', 'MARUTI.NS', 'ONGC.NS', 'IOC.NS', 'COALINDIA.NS', 'NTPC.NS']
}

# Manually provide market cap data for each company in billions of USD
market_caps = [150, 80, 30, 40, 50, 190, 70, 110, 40, 50, 35, 50, 40, 20, 35, 30, 25, 25, 15, 25]

# Initialize list to hold 5-year percentage returns
returns = []

# Fetch 5-year percentage return for each company
for company, ticker in zip(top_20_employers['Company'], top_20_employers['Ticker']):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='5y')
    returns.append((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100)

# Create a DataFrame with market caps and returns
data = pd.DataFrame({
    'Company': top_20_employers['Company'],
    'Market Cap': market_caps,
    '5-Year Return': returns
})

# Normalize market cap for heatmap size representation
data['Market Cap Size'] = data['Market Cap'] / data['Market Cap'].max() * 100

# Create the heatmap
plt.figure(figsize=(12, 8))
heatmap_data = data.pivot('Company', 'Market Cap Size', '5-Year Return')
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='RdYlGn', linewidths=.5)

plt.title('Heatmap of Top 20 Employers by Market Cap Size and 5-Year Percentage Return')
plt.xlabel('Market Cap Size (Normalized)')
plt.ylabel('Company')
plt.show()


# In[18]:


import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# List of top 20 employers and their tickers
top_20_employers = {
    'Company': ['TCS', 'Infosys', 'Wipro', 'HCL Technologies', 'SBI', 'Reliance Industries', 'ICICI Bank', 
                'HDFC Bank', 'Bharti Airtel', 'L&T', 'Tech Mahindra', 'Axis Bank', 'Cognizant', 
                'Tata Steel', 'Mahindra & Mahindra', 'Maruti Suzuki', 'ONGC', 'Indian Oil', 'Coal India', 'NTPC'],
    'Ticker': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'SBIN.NS', 'RELIANCE.NS', 'ICICIBANK.NS', 
               'HDFCBANK.NS', 'BHARTIARTL.NS', 'LT.NS', 'TECHM.NS', 'AXISBANK.NS', 'CTSH', 
               'TATASTEEL.NS', 'M&M.NS', 'MARUTI.NS', 'ONGC.NS', 'IOC.NS', 'COALINDIA.NS', 'NTPC.NS']
}

# Manually provide market cap data for each company in billions of USD
market_caps = [150, 80, 30, 40, 50, 190, 70, 110, 40, 50, 35, 50, 40, 20, 35, 30, 25, 25, 15, 25]

# Manually provide number of employees for each company in thousands
num_employees = [500, 300, 180, 150, 250, 195, 100, 120, 90, 80, 125, 95, 70, 60, 70, 55, 40, 35, 25, 30]

# Initialize list to hold 5-year percentage returns
returns = []

# Fetch 5-year percentage return for each company
for company, ticker in zip(top_20_employers['Company'], top_20_employers['Ticker']):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='5y')
    returns.append((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100)

# Create a DataFrame with the data
data = pd.DataFrame({
    'Company': top_20_employers['Company'],
    'Market Cap': market_caps,
    '5-Year Return': returns,
    'Employees': num_employees
})

# Plotting the data
fig, ax1 = plt.subplots(figsize=(14, 8))

# Bar chart for Market Cap
ax1.bar(data['Company'], data['Market Cap'], color='b', alpha=0.6, label='Market Cap (Billion USD)')
ax1.set_ylabel('Market Cap (Billion USD)', color='b')
ax1.set_xticklabels(data['Company'], rotation=90)
ax1.tick_params(axis='y', labelcolor='b')

# Create a second y-axis to plot the 5-Year Return and number of employees
ax2 = ax1.twinx()
ax2.plot(data['Company'], data['5-Year Return'], color='r', marker='o', linestyle='-', label='5-Year Return (%)')
ax2.set_ylabel('5-Year Return (%)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Scatter plot for number of employees
ax2.scatter(data['Company'], data['Employees'], color='g', s=100, label='Employees (Thousands)', alpha=0.6)

# Adding titles and legend
plt.title('Comparative Chart: Market Cap, 5-Year Return, and Number of Employees')
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
plt.show()


# In[28]:


import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# List of top 20 employers and their tickers
top_20_employers = {
    'Company': ['TCS', 'Infosys', 'Wipro', 'HCL Technologies', 'SBI', 'Reliance Industries', 'ICICI Bank', 
                'HDFC Bank', 'Bharti Airtel', 'L&T', 'Tech Mahindra', 'Axis Bank', 'Cognizant', 'Mahindra & Mahindra', 'Maruti Suzuki', 'ONGC', 'Indian Oil', 'Coal India', 'NTPC','Quess Corp'],
    'Ticker': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'SBIN.NS', 'RELIANCE.NS', 'ICICIBANK.NS', 
               'HDFCBANK.NS', 'BHARTIARTL.NS', 'LT.NS', 'TECHM.NS', 'AXISBANK.NS', 'CTSH', 'M&M.NS', 'MARUTI.NS', 'ONGC.NS', 'IOC.NS', 'COALINDIA.NS', 'NTPC.NS','QUESS.NS']
}

# Manually provide market cap data for each company in billions of USD
market_caps = [150, 80, 30, 40, 50, 190, 70, 110, 40, 50, 35, 50, 40, 35, 30, 25, 25, 15, 25,102]

# Manually provide number of employees for each company in thousands
num_employees = [500, 300, 180, 150, 250, 195, 100, 120, 90, 80, 125, 95, 70, 70, 55, 40, 35, 25, 30, 500]

# Initialize list to hold 5-year percentage returns
returns = []

# Fetch 5-year percentage return for each company
for company, ticker in zip(top_20_employers['Company'], top_20_employers['Ticker']):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='5y')
    returns.append((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100)

# Create a DataFrame with the data
data = pd.DataFrame({
    'Company': top_20_employers['Company'],
    'Market Cap': market_caps,
    '5-Year Return': returns,
    'Employees': num_employees
})

# Plotting the data
fig, ax1 = plt.subplots(figsize=(14, 8))

# Bar chart for Market Cap and number of employees
width = 0.4
ax1.bar(data['Company'], data['Market Cap'], width, label='Market Cap (Billion USD)', color='b', alpha=0.6)
ax1.bar(data['Company'], data['Employees'], width, bottom=data['Market Cap'], label='Employees (Thousands)', color='g', alpha=0.6)
ax1.set_ylabel('Market Cap (Billion USD) and Employees (Thousands)')
ax1.set_xticklabels(data['Company'], rotation=90)

# Create a second y-axis to plot the 5-Year Return
ax2 = ax1.twinx()
ax2.plot(data['Company'], data['5-Year Return'], color='dark red', marker='o', linestyle='-', label='5-Year Return (%)')
ax2.set_ylabel('5-Year Return (%)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Adding titles and legend
plt.title('Comparative Chart: Market Cap, 5-Year Return, and Number of Employees')
fig.tight_layout()  # Adjust layout to prevent overlap
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
plt.show()


# In[1]:





# In[7]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Load unemployment data
unemployment_data = pd.DataFrame({
    'Year': range(1991, 2024),
    'Unemployment Rate': [
        6.85, 6.853, 6.859, 6.829, 6.99, 7.147, 7.335, 7.517, 7.682, 7.856, 8.039, 8.248,
        8.397, 8.551, 8.697, 8.614, 8.534, 8.486, 8.406, 8.318, 8.222, 8.156, 8.088,
        7.992, 7.894, 7.8, 7.723, 7.652, 6.51, 7.859, 6.38, 4.172, 4.822
    ]
})

# Fetch historical data for SENSEX and Nifty 50
sensex_data = yf.Ticker("^BSESN").history(start="1991-01-01", end="2023-12-31", interval="1mo")
nifty_data = yf.Ticker("^NSEI").history(start="1991-01-01", end="2023-12-31", interval="1mo")

# Calculate yearly percentage change for SENSEX and Nifty 50
sensex_pct_change = sensex_data['Close'].resample('Y').last().pct_change() * 100
nifty_pct_change = nifty_data['Close'].resample('Y').last().pct_change() * 100

# Ensure the unemployment DataFrame is properly indexed by year for alignment
unemployment_rate_series = unemployment_data.set_index('Year')['Unemployment Rate']

# Align the indices properly and convert to a DataFrame directly
aligned_data = pd.DataFrame({
    'SENSEX % Change': sensex_pct_change,
    'Nifty 50 % Change': nifty_pct_change
}).dropna()  # Drop NaN values to ensure clean data

aligned_data['Year'] = aligned_data.index.year  # Extract the year from the index

# Merge with unemployment data
combined_data = pd.merge(unemployment_data, aligned_data, on='Year', how='inner')

# Plotting the data
plt.figure(figsize=(14, 8))
plt.plot(combined_data['Year'], combined_data['Unemployment Rate'], label='Unemployment Rate', marker='o')
plt.plot(combined_data['Year'], combined_data['SENSEX % Change'], label='SENSEX % Change', marker='o')
plt.plot(combined_data['Year'], combined_data['Nifty 50 % Change'], label='Nifty 50 % Change', marker='o')
plt.title('Comparison of Unemployment Rate with SENSEX and Nifty 50 Performance (2008-2023)')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.legend()
plt.grid(True)
plt.show()


# In[9]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Load unemployment data
unemployment_data = pd.DataFrame({
    'Year': range(1991, 2024),
    'Unemployment Rate': [
        6.85, 6.853, 6.859, 6.829, 6.99, 7.147, 7.335, 7.517, 7.682, 7.856, 8.039, 8.248,
        8.397, 8.551, 8.697, 8.614, 8.534, 8.486, 8.406, 8.318, 8.222, 8.156, 8.088,
        7.992, 7.894, 7.8, 7.723, 7.652, 6.51, 7.859, 6.38, 4.172, 4.822
    ]
})

# Fetch historical data for SENSEX and Nifty 50
sensex_data = yf.Ticker("^BSESN").history(start="1991-01-01", end="2023-12-31", interval="1mo")
nifty_data = yf.Ticker("^NSEI").history(start="1991-01-01", end="2023-12-31", interval="1mo")

# Calculate yearly percentage change for SENSEX and Nifty 50
sensex_pct_change = sensex_data['Close'].resample('Y').last().pct_change() * 100
nifty_pct_change = nifty_data['Close'].resample('Y').last().pct_change() * 100

# Ensure the unemployment DataFrame is properly indexed by year for alignment
aligned_data = pd.DataFrame({
    'SENSEX % Change': sensex_pct_change,
    'Nifty 50 % Change': nifty_pct_change
}).dropna()  # Drop NaN values to ensure clean data

aligned_data['Year'] = aligned_data.index.year  # Extract the year from the index

# Merge with unemployment data
combined_data = pd.merge(unemployment_data, aligned_data, on='Year', how='inner')

# Plotting the data with dual axes
fig, ax1 = plt.subplots(figsize=(14, 8))

# Market performance on the left y-axis
color = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('Market Performance (%)', color=color)
ax1.plot(combined_data['Year'], combined_data['SENSEX % Change'], label='SENSEX % Change', color='tab:orange', marker='o')
ax1.plot(combined_data['Year'], combined_data['Nifty 50 % Change'], label='Nifty 50 % Change', color='tab:blue', marker='o')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

# Unemployment rate on the right y-axis
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('Unemployment Rate (%)', color=color)
ax2.plot(combined_data['Year'], combined_data['Unemployment Rate'], label='Unemployment Rate', color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

plt.title('Comparison of Unemployment Rate with SENSEX and Nifty 50 Performance (2008-2022)')
plt.grid(True)
plt.show()


# In[16]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define the period for data extraction
start_date = "2008-01-01"
end_date = "2023-12-31"

# Load unemployment data
unemployment_data = pd.DataFrame({
    'Year': range(1991, 2024),
    'Unemployment Rate': [
        6.85, 6.853, 6.859, 6.829, 6.99, 7.147, 7.335, 7.517, 7.682, 7.856, 8.039, 8.248,
        8.397, 8.551, 8.697, 8.614, 8.534, 8.486, 8.406, 8.318, 8.222, 8.156, 8.088,
        7.992, 7.894, 7.8, 7.723, 7.65, 8.00, 7.85, 7.33, 8.0, 9.2
    ]
}).set_index('Year')

# Fetch historical data for SENSEX and Nifty 50
sensex_data = yf.Ticker("^BSESN").history(start=start_date, end=end_date, interval="1mo")
nifty_data = yf.Ticker("^NSEI").history(start=start_date, end=end_date, interval="1mo")

# Calculate normalized cumulative returns for SENSEX and Nifty 50
sensex_cum_return = (sensex_data['Close'].pct_change() + 1).cumprod() - 1
nifty_cum_return = (nifty_data['Close'].pct_change() + 1).cumprod() - 1

# Normalize the cumulative returns starting from 100
sensex_cum_return = 100 * (1 + sensex_cum_return)
nifty_cum_return = 100 * (1 + nifty_cum_return)

# Create annual data from the cumulative return data
sensex_annual = sensex_cum_return.resample('Y').last()
nifty_annual = nifty_cum_return.resample('Y').last()

# Align and clean data
data_to_merge = {
    'SENSEX Cumulative Return': sensex_annual,
    'Nifty 50 Cumulative Return': nifty_annual
}
market_data = pd.concat(data_to_merge, axis=1)
market_data.index = market_data.index.year  # Convert DatetimeIndex to plain years

# Merge with unemployment data
combined_data = market_data.join(unemployment_data, how='inner')

# Plotting the data with dual axes
fig, ax1 = plt.subplots(figsize=(14, 8))

# Market performance on the left y-axis
ax1.set_xlabel('Year')
ax1.set_ylabel('Cumulative Return (Normalized to 100)', color='tab:blue')
ax1.plot(combined_data.index, combined_data['SENSEX Cumulative Return'], label='SENSEX Cumulative Return', color='tab:orange', marker='o')
ax1.plot(combined_data.index, combined_data['Nifty 50 Cumulative Return'], label='Nifty 50 Cumulative Return', color='tab:blue', marker='o')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')

# Unemployment rate on the right y-axis
ax2 = ax1.twinx()  
ax2.set_ylabel('Unemployment Rate (%)', color='tab:red')
ax2.plot(combined_data.index, combined_data['Unemployment Rate'], label='Unemployment Rate', color='tab:red', marker='o')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc='upper right')

plt.title('Comparison of Unemployment Rate with Normalized Cumulative Returns of SENSEX and Nifty 50')
plt.grid(True)
plt.show()


# In[20]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the period for data extraction
start_date = "2008-01-01"
end_date = "2023-12-31"

# Load unemployment data
unemployment_data = pd.DataFrame({
    'Year': range(1991, 2024),
    'Unemployment Rate': [
        6.85, 6.853, 6.859, 6.829, 6.99, 7.147, 7.335, 7.517, 7.682, 7.856, 8.039, 8.248,
        8.397, 8.551, 8.697, 8.614, 8.534, 8.486, 8.406, 8.318, 8.222, 8.156, 8.088,
        7.992, 7.894, 7.8, 7.723, 7.65, 8.00, 7.85, 7.33, 8.0, 9.2
    ]
}).set_index('Year')

# Fetch historical data for SENSEX and Nifty 50
sensex_data = yf.Ticker("^BSESN").history(start=start_date, end=end_date, interval="1d")
nifty_data = yf.Ticker("^NSEI").history(start=start_date, end=end_date, interval="1d")

# Calculate normalized cumulative returns for SENSEX and Nifty 50
sensex_cum_return = (sensex_data['Close'].pct_change() + 1).cumprod() - 1
nifty_cum_return = (nifty_data['Close'].pct_change() + 1).cumprod() - 1

# Normalize the cumulative returns starting from 100
sensex_cum_return = 100 * (1 + sensex_cum_return)
nifty_cum_return = 100 * (1 + nifty_cum_return)

# Create annual data from the cumulative return data and calculate log returns
sensex_annual = sensex_cum_return.resample('Y').last()
nifty_annual = nifty_cum_return.resample('Y').last()
sensex_log_change = np.log(sensex_annual).diff()
nifty_log_change = np.log(nifty_annual).diff()

# Merge data
data_to_merge = {
    'SENSEX Log Change': sensex_log_change,
    'Nifty 50 Log Change': nifty_log_change
}
market_data = pd.concat(data_to_merge, axis=1)
market_data.index = market_data.index.year  # Convert DatetimeIndex to plain years

# Merge with unemployment data
combined_data = market_data.join(unemployment_data, how='inner')

# Plotting the data with dual axes
fig, ax1 = plt.subplots(figsize=(14, 8))

# Logarithmic changes on the left y-axis
ax1.set_xlabel('Year')
ax1.set_ylabel('Logarithmic Change (YoY)', color='tab:blue')
ax1.plot(combined_data.index, combined_data['SENSEX Log Change'], label='SENSEX Log Change', color='tab:orange', marker='o')
ax1.plot(combined_data.index, combined_data['Nifty 50 Log Change'], label='Nifty 50 Log Change', color='tab:blue', marker='o')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')

# Unemployment rate on the right y-axis
ax2 = ax1.twinx()  
ax2.set_ylabel('Unemployment Rate (%)', color='tab:red')
ax2.plot(combined_data.index, combined_data['Unemployment Rate'], label='Unemployment Rate', color='tab:red', marker='o')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc='upper right')

plt.title('Logarithmic Year-Over-Year Changes vs. Unemployment Rate')
plt.grid(True)
plt.show()


# In[23]:


# Define the period for data extraction
start_date = "2008-01-01"
end_date = "2023-12-31"

# Load unemployment data
unemployment_data = pd.DataFrame({
    'Year': range(1991, 2024),
    'Unemployment Rate': [
        6.85, 6.853, 6.859, 6.829, 6.99, 7.147, 7.335, 7.517, 7.682, 7.856, 8.039, 8.248,
        8.397, 8.551, 8.697, 8.614, 8.534, 8.486, 8.406, 8.318, 8.222, 8.156, 8.088,
        7.992, 7.894, 7.8, 7.723, 7.65, 8.00, 7.85, 7.33, 8.0, 9.2
    ]
}).set_index('Year')


# Fetch historical data for SENSEX and Nifty 50
sensex_data = yf.Ticker("^BSESN").history(start=start_date, end=end_date, interval="1mo")
nifty_data = yf.Ticker("^NSEI").history(start=start_date, end=end_date, interval="1mo")

# Calculate yearly percentage change for SENSEX and Nifty 50
sensex_pct_change = sensex_data['Close'].resample('Y').last().pct_change() * 100
nifty_pct_change = nifty_data['Close'].resample('Y').last().pct_change() * 100

# Align and clean data
data_to_merge = {
    'SENSEX % Change': sensex_pct_change,
    'Nifty 50 % Change': nifty_pct_change
}
market_data = pd.concat(data_to_merge, axis=1)
market_data.index = market_data.index.year  # Convert DatetimeIndex to plain years

# Merge with unemployment data
combined_data = market_data.join(unemployment_data, how='inner')

# Plotting the data with dual axes
fig, ax1 = plt.subplots(figsize=(14, 8))

# Percentage changes on the left y-axis
ax1.set_xlabel('Year')
ax1.set_ylabel('Percentage Change (YoY)', color='tab:blue')
ax1.plot(combined_data.index, combined_data['SENSEX % Change'], label='SENSEX % Change', color='tab:orange', marker='o')
ax1.plot(combined_data.index, combined_data['Nifty 50 % Change'], label='Nifty 50 % Change', color='tab:blue', marker='o')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')

# Unemployment rate on the right y-axis
ax2 = ax1.twinx()  
ax2.set_ylabel('Unemployment Rate (%)', color='tab:red')
ax2.plot(combined_data.index, combined_data['Unemployment Rate'], label='Unemployment Rate', color='tab:red', marker='o')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc='upper right')

plt.title('Year-Over-Year Percentage Changes vs. Unemployment Rate')
plt.grid(True)
plt.show()


# In[29]:


# Define the period for data extraction
start_date = "2008-01-01"
end_date = "2023-12-31"

# Load unemployment data
unemployment_data = pd.DataFrame({
    'Year': range(1991, 2024),
    'Unemployment Rate': [
        6.85, 6.853, 6.859, 6.829, 6.99, 7.147, 7.335, 7.517, 7.682, 7.856, 8.039, 8.248,
        8.397, 8.551, 8.697, 8.614, 8.534, 8.486, 8.406, 8.318, 8.222, 8.156, 8.088,
        7.992, 7.894, 7.8, 7.723, 7.65, 8.00, 7.85, 7.33, 8.0, 9.2
    ]
}).set_index('Year')


# Fetch historical data for SENSEX and Nifty 50
sensex_data = yf.Ticker("^BSESN").history(start=start_date, end=end_date, interval="1mo")
nifty_data = yf.Ticker("^NSEI").history(start=start_date, end=end_date, interval="1mo")

# Calculate yearly percentage change for SENSEX and Nifty 50
sensex_pct_change = sensex_data['Close'].resample('Y').last().pct_change() * 100
nifty_pct_change = nifty_data['Close'].resample('Y').last().pct_change() * 100

# Align and clean data
data_to_merge = {
    'SENSEX % Change': sensex_pct_change,
    'Nifty 50 % Change': nifty_pct_change
}
market_data = pd.concat(data_to_merge, axis=1)
market_data.index = market_data.index.year  # Convert DatetimeIndex to plain years

# Merge with unemployment data
combined_data = market_data.join(unemployment_data, how='inner')

# Plotting the data with dual axes
fig, ax1 = plt.subplots(figsize=(14, 8))

# Percentage changes on the left y-axis (Bar Chart)
ax1.set_xlabel('Year')
ax1.set_ylabel('Percentage Change (YoY)', color='tab:blue')
ax1.bar(combined_data.index - 0.2, combined_data['SENSEX % Change'], width=0.4, label='SENSEX % Change', color='tab:orange', align='center')
ax1.bar(combined_data.index + 0.2, combined_data['Nifty 50 % Change'], width=0.4, label='Nifty 50 % Change', color='tab:blue', align='center')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')

# Unemployment rate on the right y-axis (Line Chart)
ax2 = ax1.twinx()  
ax2.set_ylabel('Unemployment Rate (%)', color='tab:red')
ax2.plot(combined_data.index, combined_data['Unemployment Rate'], label='Unemployment Rate', color='tab:red', marker='o', linestyle='-')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc='upper right')

plt.title('Year-Over-Year Percentage Changes vs. Unemployment Rate')
plt.grid(True)
plt.xticks(combined_data.index)  # Ensure all years are labeled
plt.show()


# In[ ]:





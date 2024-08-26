import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
prices_df = pd.read_csv('prices-split-adjusted.csv')

# Convert date column to datetime format
prices_df['date'] = pd.to_datetime(prices_df['date'])

# Summary statistics
print("\nSummary Statistics:")
print(prices_df.describe())

# Line plot of closing prices for a specific symbol
symbol_of_interest = 'WLTW'
symbol_data = prices_df[prices_df['symbol'] == symbol_of_interest]
plt.plot(symbol_data['date'], symbol_data['close'])
plt.title(f'Closing Prices for {symbol_of_interest}')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.xticks(rotation=45)
plt.show()

# Histogram of trading volumes
plt.hist(prices_df['volume'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Trading Volumes')
plt.xlabel('Volume')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix
numeric_columns = prices_df.select_dtypes(include=[float, int]).columns
correlation_matrix = prices_df[numeric_columns].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

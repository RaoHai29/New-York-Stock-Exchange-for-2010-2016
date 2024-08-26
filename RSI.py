import pandas as pd
import matplotlib.pyplot as plt
import ta

# Load the CSV file
df = pd.read_csv('prices.csv')

# Filter data for the symbol 'WLTW'
df_wltw = df[df['symbol'] == 'WLTW']

# Convert the 'date' column to datetime
df_wltw['date'] = pd.to_datetime(df_wltw['date'])

# Set 'date' as the index
df_wltw.set_index('date', inplace=True)

# Calculate the RSI using the 'close' prices
df_wltw['RSI'] = ta.momentum.rsi(df_wltw['close'], window=14)

# Display the RSI values
print(df_wltw[['close', 'RSI']])

# Plot the RSI
def plot_rsi(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['RSI'], label='RSI', color='blue')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title('Relative Strength Index (RSI) for WLTW')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.show()

plot_rsi(df_wltw)

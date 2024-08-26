import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import ta

# Load the data
df = pd.read_csv('prices-split-adjusted.csv')

# Convert date column to datetime, handling different formats
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Drop rows with invalid dates
df.dropna(subset=['date'], inplace=True)

# Remove duplicate dates by taking the first occurrence
df = df.drop_duplicates(subset=['date'], keep='first')

# Ensure the date index is sorted
df.sort_values('date', inplace=True)

# Set date as index
df.set_index('date', inplace=True)

# Set frequency to business days
df = df.asfreq('B')

# Exploratory Data Analysis
def plot_close_price(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df['close'], label='Close Price')
    plt.title('Close Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

# Plot the closing price
plot_close_price(df)

# Calculate and plot moving averages
df['SMA_20'] = df['close'].rolling(window=20).mean()
df['SMA_50'] = df['close'].rolling(window=50).mean()

def plot_moving_averages(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df['close'], label='Close Price')
    plt.plot(df['SMA_20'], label='20-Day SMA')
    plt.plot(df['SMA_50'], label='50-Day SMA')
    plt.title('Close Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

plot_moving_averages(df)

# Technical Analysis
# Calculate RSI
df['RSI'] = ta.momentum.rsi(df['close'], window=14)

# Calculate Bollinger Bands
bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
df['BB_High'] = bb_indicator.bollinger_hband()
df['BB_Low'] = bb_indicator.bollinger_lband()

def plot_bollinger_bands(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df['close'], label='Close Price')
    plt.plot(df['BB_High'], label='Bollinger High Band')
    plt.plot(df['BB_Low'], label='Bollinger Low Band')
    plt.fill_between(df.index, df['BB_Low'], df['BB_High'], color='gray', alpha=0.3)
    plt.title('Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

plot_bollinger_bands(df)

# Predictive Modeling using ARIMA
def arima_forecast(df):
    # Fit ARIMA model
    model = ARIMA(df['close'], order=(5, 1, 0))
    model_fit = model.fit()
    print(model_fit.summary())

    # Forecast
    forecast = model_fit.forecast(steps=30)
    plt.figure(figsize=(14, 7))
    plt.plot(df['close'], label='Actual')
    plt.plot(forecast, label='Forecast', color='red')
    plt.title('ARIMA Forecast')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

arima_forecast(df)

# Volatility Modeling using GARCH
def garch_forecast(df):
    # Rescale returns
    returns = df['close'].pct_change().dropna()
    scaled_returns = returns * 100  # Scaling returns

    # Fit GARCH model
    garch = arch_model(scaled_returns, vol='Garch', p=1, q=1)
    garch_fit = garch.fit(disp='off')
    print(garch_fit.summary())



garch_forecast(df)

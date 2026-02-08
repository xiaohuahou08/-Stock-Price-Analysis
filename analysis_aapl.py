import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import math

# Parameters
symbol = 'AAPL'
end = datetime.utcnow().date()
start = end - timedelta(days=365)
period1 = int(time.mktime(start.timetuple()))
period2 = int(time.mktime((end + timedelta(days=1)).timetuple()))
# Try scraping the history page table (avoids CSV download limits)
url = f"https://finance.yahoo.com/quote/{symbol}/history?p={symbol}"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}
resp = requests.get(url, headers=headers)
if resp.status_code != 200:
    print('Failed to fetch history page:', resp.status_code)
    raise SystemExit(1)

# pandas can extract the visible historical-prices table from the HTML
tables = pd.read_html(resp.text)
# find table that contains a 'Close' column
df = None
for t in tables:
    if 'Close' in t.columns:
        df = t.copy()
        break
if df is None:
    print('Historical prices table not found on page')
    raise SystemExit(1)
# Drop rows that are not numeric (e.g., 'Dividend' rows)
df = df[pd.to_numeric(df['Close'], errors='coerce').notnull()]

df = df.sort_values('Date').reset_index(drop=True)
close = df['Close'].astype(float)
volume = df['Volume'].astype(float)
last_price = close.iloc[-1]

# Descriptive stats
mean_price = close.mean()
median_price = close.median()
std_price = close.std()

# Moving averages
ma30 = close.rolling(window=30, min_periods=1).mean()
ma90 = close.rolling(window=90, min_periods=1).mean()
ma30_last = ma30.iloc[-1]
ma90_last = ma90.iloc[-1]

# Trendline (linear regression on price vs day index)
x = np.arange(len(close))
coef = np.polyfit(x, close.values, 1)
slope_per_day = coef[0]
intercept = coef[1]
# extend 30 days
slope_30day_change = slope_per_day * 30
trend_ext_price = last_price + slope_30day_change

# Volatility and 30-day projection range using daily log returns
returns = np.log(close / close.shift(1)).dropna()
daily_return_std = returns.std()
# 30-day std (log-return)
std_30 = daily_return_std * math.sqrt(30)
# Projected 30-day price range using lognormal approx (1-sigma)
proj_low = last_price * math.exp(-std_30)
proj_high = last_price * math.exp(std_30)

# Output results
print(f"Symbol: {symbol}")
print(f"Date range: {start} to {end}")
print(f"Last price: {last_price:.2f}")
print(f"Mean close: {mean_price:.2f}")
print(f"Median close: {median_price:.2f}")
print(f"Std dev (price): {std_price:.2f}")
print(f"30d MA (last): {ma30_last:.2f}")
print(f"90d MA (last): {ma90_last:.2f}")
print(f"Trend slope (price/day): {slope_per_day:.6f}")
print(f"Trend 30-day extension (price): {trend_ext_price:.2f}")
print(f"Daily return std: {daily_return_std:.6f}")
print(f"Projected 30-day price range (approx 1σ): {proj_low:.2f} - {proj_high:.2f}")

# Save summary to file
with open('aapl_analysis_summary.txt', 'w') as f:
    f.write(f"Symbol: {symbol}\n")
    f.write(f"Date range: {start} to {end}\n")
    f.write(f"Last price: {last_price:.2f}\n")
    f.write(f"Mean close: {mean_price:.2f}\n")
    f.write(f"Median close: {median_price:.2f}\n")
    f.write(f"Std dev (price): {std_price:.2f}\n")
    f.write(f"30d MA (last): {ma30_last:.2f}\n")
    f.write(f"90d MA (last): {ma90_last:.2f}\n")
    f.write(f"Trend slope (price/day): {slope_per_day:.6f}\n")
    f.write(f"Trend 30-day extension (price): {trend_ext_price:.2f}\n")
    f.write(f"Projected 30-day price range (approx 1σ): {proj_low:.2f} - {proj_high:.2f}\n")

print('\nSummary written to aapl_analysis_summary.txt')

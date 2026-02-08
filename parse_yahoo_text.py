import re
import pandas as pd
import numpy as np
import math
from datetime import datetime

lines = open('yahoo_history_raw.txt','r',encoding='utf-8').read().splitlines()

pattern = re.compile(r"\|\s*([A-Za-z]{3} \d{1,2}, \d{4})\s*\|\s*([0-9.,]+)\s*\|\s*([0-9.,]+)\s*\|\s*([0-9.,]+)\s*\|\s*([0-9.,]+)\s*\|\s*([0-9.,]+)\s*\|\s*([0-9,]+)\s*\|")
rows = []
for L in lines:
    m = pattern.search(L)
    if m:
        date_s = m.group(1)
        open_p = float(m.group(2).replace(',',''))
        high = float(m.group(3).replace(',',''))
        low = float(m.group(4).replace(',',''))
        close = float(m.group(5).replace(',',''))
        adj_close = float(m.group(6).replace(',',''))
        volume = int(m.group(7).replace(',',''))
        # parse date
        dt = datetime.strptime(date_s, '%b %d, %Y').date()
        rows.append((dt, open_p, high, low, close, adj_close, volume))

if not rows:
    print('No rows parsed')
    raise SystemExit(1)

# Build DataFrame sorted by date ascending
rows = sorted(rows, key=lambda x: x[0])
df = pd.DataFrame(rows, columns=['Date','Open','High','Low','Close','Adj Close','Volume'])

close = df['Close']
last_price = close.iloc[-1]
mean_price = close.mean()
median_price = close.median()
std_price = close.std()
ma30 = close.rolling(window=30,min_periods=1).mean().iloc[-1]
ma90 = close.rolling(window=90,min_periods=1).mean().iloc[-1]

# Trendline
x = np.arange(len(close))
coef = np.polyfit(x, close.values, 1)
slope_per_day = coef[0]
trend_ext_price = last_price + slope_per_day*30

# Volatility
returns = np.log(close / close.shift(1)).dropna()
daily_return_std = returns.std()
std_30 = daily_return_std * math.sqrt(30)
proj_low = last_price * math.exp(-std_30)
proj_high = last_price * math.exp(std_30)

print(f"Parsed {len(df)} rows")
print(f"Last price: {last_price:.2f}")
print(f"Mean: {mean_price:.2f}")
print(f"Median: {median_price:.2f}")
print(f"Std dev (price): {std_price:.2f}")
print(f"30d MA: {ma30:.2f}")
print(f"90d MA: {ma90:.2f}")
print(f"Trend slope (price/day): {slope_per_day:.6f}")
print(f"Trend 30-day extension price: {trend_ext_price:.2f}")
print(f"Daily return std: {daily_return_std:.6f}")
print(f"Projected 30-day range (1σ): {proj_low:.2f} - {proj_high:.2f}")

# Save summary
with open('aapl_parsed_summary.txt','w') as f:
    f.write(f"Rows: {len(df)}\n")
    f.write(f"Last price: {last_price:.2f}\n")
    f.write(f"Mean: {mean_price:.2f}\n")
    f.write(f"Median: {median_price:.2f}\n")
    f.write(f"Std dev (price): {std_price:.2f}\n")
    f.write(f"30d MA: {ma30:.2f}\n")
    f.write(f"90d MA: {ma90:.2f}\n")
    f.write(f"Trend slope (price/day): {slope_per_day:.6f}\n")
    f.write(f"Trend 30-day extension price: {trend_ext_price:.2f}\n")
    f.write(f"Projected 30-day range (1σ): {proj_low:.2f} - {proj_high:.2f}\n")

print('\nSummary saved to aapl_parsed_summary.txt')

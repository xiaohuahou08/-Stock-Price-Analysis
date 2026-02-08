import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from datetime import datetime, timedelta
import textwrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def parse_raw(path='yahoo_history_raw.txt'):
    pattern = re.compile(r"\|\s*([A-Za-z]{3} \d{1,2}, \d{4})\s*\|\s*([0-9.,]+)\s*\|\s*([0-9.,]+)\s*\|\s*([0-9.,]+)\s*\|\s*([0-9.,]+)\s*\|\s*([0-9.,]+)\s*\|\s*([0-9,]+)\s*\|")
    lines = open(path, 'r', encoding='utf-8').read().splitlines()
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
            dt = datetime.strptime(date_s, '%b %d, %Y').date()
            rows.append((dt, open_p, high, low, close, adj_close, volume))
    if not rows:
        raise RuntimeError('No rows parsed from raw file')
    rows = sorted(rows, key=lambda x: x[0])
    df = pd.DataFrame(rows, columns=['Date','Open','High','Low','Close','Adj Close','Volume'])
    return df


def compute_metrics(df):
    close = df['Close']
    last_price = close.iloc[-1]
    mean_price = close.mean()
    median_price = close.median()
    std_price = close.std()
    ma30 = close.rolling(window=30, min_periods=1).mean()
    ma90 = close.rolling(window=90, min_periods=1).mean()
    x = np.arange(len(close))
    coef = np.polyfit(x, close.values, 1)
    slope_per_day = coef[0]
    intercept = coef[1]
    # volatility and projection
    returns = np.log(close / close.shift(1)).dropna()
    daily_return_std = returns.std()
    std_30 = daily_return_std * math.sqrt(30)
    proj_low = last_price * math.exp(-std_30)
    proj_high = last_price * math.exp(std_30)
    return {
        'last_price': last_price,
        'mean': mean_price,
        'median': median_price,
        'std': std_price,
        'ma30': ma30,
        'ma90': ma90,
        'slope': slope_per_day,
        'intercept': intercept,
        'daily_return_std': daily_return_std,
        'proj_low': proj_low,
        'proj_high': proj_high
    }


def make_report(df, metrics, out_pdf='AAPL_report_improved.pdf'):
    # Typography defaults
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 10,
    })

    dates = pd.to_datetime(df['Date'])
    close = df['Close']
    ma30 = metrics['ma30']
    ma90 = metrics['ma90']
    slope = metrics['slope']
    intercept = metrics['intercept']
    last_date = dates.iloc[-1]

    # future business days for 30 trading days
    future_dates = pd.bdate_range(last_date + timedelta(days=1), periods=30)
    # indices for trendline extension
    x_hist = np.arange(len(close))
    x_future = np.arange(len(close), len(close) + len(future_dates))
    trend_hist = slope * x_hist + intercept
    trend_future = slope * x_future + intercept

    # projection band (constant multiplicative band based on std_30)
    proj_low = metrics['proj_low']
    proj_high = metrics['proj_high']

    # Prepare figures in memory with titles and descriptions for TOC
    pages = []  # list of tuples (fig, short_title, description)

    # (Removed redundant standalone summary page; analysis intro provides the summary)

    # Close price with moving averages
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(dates, close, label='Close', color='tab:blue', linewidth=1.2)
    ax.plot(dates, ma30, label='30d MA', color='tab:orange')
    ax.plot(dates, ma90, label='90d MA', color='tab:green')
    ax.set_title('Close with 30d & 90d Moving Averages')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    desc = "This chart shows Apple's daily closing price over the past year, with 30-day and 90-day moving averages. The moving averages help visualize medium- and long-term price trends, smoothing out short-term fluctuations."
    pages.append((fig, 'Moving Averages', desc))

    # Trendline with extension
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(dates, close, label='Close', color='tab:blue', linewidth=1.0)
    ax.plot(dates, trend_hist, '--', color='gray', label='Trend (hist)')
    ax.plot(future_dates, trend_future, '--', color='gray', label='Trend (30d ext)')
    ax.axvline(dates.iloc[-1], color='k', linewidth=0.6, linestyle=':')
    ax.set_title(f'Trendline and 30‑day Extension (slope={slope:.4f} USD/day)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    desc = "A linear trendline is fitted to Apple's historical closing prices and extended 30 trading days into the future. This illustrates the average price drift, but does not account for volatility or unexpected events."
    pages.append((fig, 'Trend Extension', desc))

    # Daily returns histogram
    returns = np.log(close / close.shift(1)).dropna()
    mu = returns.mean()
    sigma = returns.std()
    fig, ax = plt.subplots(figsize=(8, 6))
    n, bins, patches = ax.hist(returns, bins=30, density=True, alpha=0.6, color='tab:blue')
    from scipy.stats import norm
    x_vals = np.linspace(returns.min(), returns.max(), 200)
    ax.plot(x_vals, norm.pdf(x_vals, mu, sigma), 'r--', label='Normal PDF')
    ax.set_title('Daily Log Returns Histogram')
    ax.set_xlabel('Log Return')
    ax.set_ylabel('Density')
    ax.legend()
    fig.tight_layout()
    desc = "This histogram shows the distribution of daily log returns for Apple stock. The red dashed line is a normal distribution fit, allowing comparison of actual return behavior to theoretical expectations (skewness, fat tails)."
    pages.append((fig, 'Returns Histogram', desc))

    # 30-day rolling volatility (annualized)
    roll_vol = returns.rolling(window=30, min_periods=1).std() * math.sqrt(252)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dates[1:], roll_vol, color='tab:purple', label='30d Rolling Vol (annualized)')
    ax.set_title('30‑day Rolling Volatility (annualized)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility')
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    desc = "The 30-day rolling volatility (annualized) measures the time-varying risk of Apple stock. Higher volatility periods indicate greater uncertainty and larger price swings, often coinciding with major news or events."
    pages.append((fig, 'Rolling Volatility', desc))

    # Volume chart
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(dates, df['Volume'], color='tab:gray')
    ax.set_title('Daily Trading Volume')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.grid(alpha=0.12)
    fig.tight_layout()
    desc = "This chart displays Apple's daily trading volume. Spikes in volume often signal major price moves, news releases, or shifts in investor sentiment, and help assess liquidity."
    pages.append((fig, 'Volume', desc))

    # Now build final PDF: title page, TOC, then pages with headers/footers
    title_fig = plt.figure(figsize=(8.27, 11.69))
    title_fig.suptitle('Apple Inc. (AAPL)', fontsize=28, y=0.7)
    plt.axis('off')
    plt.text(0.5, 0.55, '1‑Year Price Analysis', ha='center', va='center', fontsize=18)
    plt.text(0.5, 0.45, f'Generated: {datetime.utcnow().date()}', ha='center', va='center', fontsize=10)

    # Build sections: merge Stock Selection + Data Collection, add Events, then Data Analysis
    rows_count = len(df)
    date_range = f"{df['Date'].min()} to {df['Date'].max()}"

    # Combined Selection & Data Collection page
    combined_fig = plt.figure(figsize=(8.27, 11.69))
    combined_fig.suptitle('Section 1 — Selection & Data Collection', fontsize=18, y=0.95)
    plt.axis('off')
    combined_paragraphs = [
        "Selection: Apple Inc. (AAPL) — large-cap U.S. equity chosen for its liquidity and representative exposure to consumer technology.",
        "Rationale: chosen for liquidity, availability of a full year of daily data, and relevance to sector-level analysis (devices, services, and semiconductors).",
        "Data collection: Yahoo Finance historical prices table was downloaded on the report run date and saved verbatim to `yahoo_history_raw.txt` to preserve the exact rows used.",
        f"Parsing pipeline: `parse_raw()` regex-extracts the table into numeric Open/High/Low/Close/Adj Close/Volume fields, sorts ascending by date, drops non-price marker rows (e.g., dividends/splits), and coerces volumes to integers. Rows parsed: {rows_count}. Date range: {date_range}.",
        "Reproducibility: rerun `analysis_aapl.py` or `generate_aapl_report.py` to regenerate the raw text and metrics using the same steps; volumes remain unadjusted share counts while prices use Yahoo's adjusted-close column for comparability.",
    ]
    y0 = 0.88
    for p in combined_paragraphs:
        wrapped = textwrap.fill(p, width=96)
        combined_fig.text(0.07, y0, wrapped, va='top', ha='left', fontsize=11)
        y0 -= 0.07

    # Events and drivers page
    events_fig = plt.figure(figsize=(8.27, 11.69))
    events_fig.suptitle('Section 2 — Major Events & Drivers', fontsize=18, y=0.95)
    plt.axis('off')
    events_paragraphs = [
        "Corporate: Quarterly earnings (typically late Jan/Apr/Jul/Oct) and forward guidance, buyback/dividend updates, and major launch events (WWDC in June, iPhone/Watch refresh in September) often trigger sharp repricing of growth and margin expectations.",
        "Industry: Semiconductor supply, foundry pricing (e.g., TSMC), and competitor launches in smartphones, PCs, and wearables shift share assumptions; channel checks or disruptions in Greater China/Europe can swing demand outlooks.",
        "Economic: FOMC rate decisions, CPI/PCE inflation releases, and labor data reset discount rates and consumer spending expectations; USD strength/weakness and tariff or export-control actions move reported revenue and cost structures.",
        "Sentiment & Events: Regulatory actions (App Store, antitrust), high-profile litigation, cybersecurity incidents, or macro shocks (banking stress, pandemics, geopolitical conflict) can cause volatility spikes or regime shifts.",
        "How they impact price: Positive surprises to revenue growth, services margins, or cost control widen valuation multiples; negative guidance, supply constraints, or macro tightening compress multiples and drive gap-downs, usually alongside elevated volume.",
    ]
    y1 = 0.88
    for p in events_paragraphs:
        wrapped = textwrap.fill(p, width=96)
        events_fig.text(0.07, y1, wrapped, va='top', ha='left', fontsize=11)
        y1 -= 0.08

    # Data Analysis section intro page (summary text recreated here)
    summary_text = (
        f"Last close: {metrics['last_price']:.2f}\n"
        f"Mean: {metrics['mean']:.2f}   Median: {metrics['median']:.2f}   Std: {metrics['std']:.2f}\n"
        f"30d MA (latest): {metrics['ma30'].iloc[-1]:.2f}   90d MA (latest): {metrics['ma90'].iloc[-1]:.2f}\n"
        f"Trend slope: {metrics['slope']:.4f} USD/day   30‑day trend ext: {metrics['last_price'] + metrics['slope']*30:.2f}\n"
        f"30‑day projected 1σ range: {proj_low:.2f} — {proj_high:.2f}\n\n"
        "Event drivers (summary): dividends and earnings, product cycles, macro (rates/inflation),\n"
        "industry supply/competitive events. Limitations: uses historical realized volatility, linear trend,\n"
        "and does not capture intraday or options-implied signals or future structural breaks.\n"
    )
    analysis_fig = plt.figure(figsize=(8.27, 11.69))
    analysis_fig.suptitle('Section 3 — Data Analysis', fontsize=18, y=0.95)
    plt.axis('off')
    plt.text(0.07, 0.85, summary_text, va='top', ha='left', fontsize=11)

    # Save all to PDF in order: title, stock selection, data collection, analysis intro, then chart pages
    with PdfPages(out_pdf) as pdf:
        page_no = 1
        # title
        pdf.savefig(title_fig)
        plt.close(title_fig)
        page_no += 1

        # Section 1: Selection & Data Collection (combined)
        pdf.savefig(combined_fig)
        plt.close(combined_fig)
        page_no += 1

        # Section 2: Major Events & Drivers
        pdf.savefig(events_fig)
        plt.close(events_fig)
        page_no += 1

        # Section 3: Data Analysis (intro)
        pdf.savefig(analysis_fig)
        plt.close(analysis_fig)
        page_no += 1

        # content pages (charts)
        gen_date = datetime.utcnow().date().isoformat()
        for fig, short_title, description in pages:
            # add consistent minimal header/footer
            fig.text(0.02, 0.98, 'Apple Inc. (AAPL)', ha='left', va='top', fontsize=8)
            fig.text(0.5, 0.93, description, ha='center', va='center', fontsize=9, color='gray')
            fig.text(0.98, 0.02, f'Page {page_no}', ha='right', va='bottom', fontsize=8)
            fig.text(0.02, 0.02, f'Generated: {gen_date}', ha='left', va='bottom', fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)
            page_no += 1

    print(f'Report saved to {out_pdf}')

    print(f'Report saved to {out_pdf}')


if __name__ == '__main__':
    df = parse_raw('yahoo_history_raw.txt')
    metrics = compute_metrics(df)
    make_report(df, metrics, out_pdf='AAPL_report.pdf')

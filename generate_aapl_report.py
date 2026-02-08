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

    # Figure sizes (in inches) optimized for PDF layout to keep captions readable without overflowing.
    wide_chart_figsize = (10.0, 5.8)
    medium_chart_figsize = (10.0, 5.0)
    compact_chart_figsize = (10.0, 4.3)
    header_top_adjust = 0.9
    footer_bottom_adjust = 0.32
    caption_y = 0.23
    description_y_position = 0.17
    description_wrap_width = 105
    description_line_spacing = 1.15
    header_x = 0.02
    footer_x = 0.02

    dates = pd.to_datetime(df['Date'])
    close = df['Close']
    ma30 = metrics['ma30']
    ma90 = metrics['ma90']
    slope = metrics['slope']
    intercept = metrics['intercept']
    last_date = dates.iloc[-1]
    last_price = metrics['last_price']
    mean_price = metrics['mean']
    median_price = metrics['median']
    std_price = metrics['std']
    ma30_last = ma30.iloc[-1]
    ma90_last = ma90.iloc[-1]
    trend_30_ext = last_price + slope * 30

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
    fig, ax = plt.subplots(figsize=wide_chart_figsize)
    ax.plot(dates, close, label='Close', color='tab:blue', linewidth=1.2)
    ax.plot(dates, ma30, label='30d MA', color='tab:orange')
    ax.plot(dates, ma90, label='90d MA', color='tab:green')
    ax.set_title('Close with 30d & 90d Moving Averages')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(alpha=0.2)
    stats_text = (
        f"Last close: {last_price:.2f}\n"
        f"Mean/Median: {mean_price:.2f}/{median_price:.2f}\n"
        f"Std dev: {std_price:.2f}\n"
        f"30d MA: {ma30_last:.2f}\n"
        f"90d MA: {ma90_last:.2f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.85),
    )
    fig.tight_layout()
    fig.subplots_adjust(top=header_top_adjust, bottom=footer_bottom_adjust)
    caption = "Daily close with 30- and 90-day moving averages."
    desc = (
        "This chart plots Apple's daily closing price for the past year and overlays 30-day and 90-day moving averages. "
        f"The latest close is {last_price:.2f}, with a mean of {mean_price:.2f}, a median of {median_price:.2f}, and a "
        f"standard deviation of {std_price:.2f} across the period. The most recent moving averages are {ma30_last:.2f} "
        f"(30d) and {ma90_last:.2f} (90d), smoothing short-term noise to reveal the underlying trend direction."
    )
    pages.append((fig, 'Moving Averages', caption, desc))

    # Trendline with extension
    fig, ax = plt.subplots(figsize=wide_chart_figsize)
    ax.plot(dates, close, label='Close', color='tab:blue', linewidth=1.0)
    ax.plot(dates, trend_hist, '--', color='gray', label='Trend (hist)')
    ax.plot(future_dates, trend_future, '--', color='gray', label='Trend (30d ext)')
    ax.axvline(dates.iloc[-1], color='k', linewidth=0.6, linestyle=':')
    ax.set_title(f'Trendline and 30‑day Extension (slope={slope:.4f} USD/day)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(alpha=0.2)
    trend_text = f"Slope: {slope:.4f} USD/day\n30d extension: {trend_30_ext:.2f}"
    ax.text(
        0.02,
        0.98,
        trend_text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.85),
    )
    fig.tight_layout()
    fig.subplots_adjust(top=header_top_adjust, bottom=footer_bottom_adjust)
    caption = "Linear trend fitted to closing prices with a 30-trading-day extension."
    desc = (
        "A linear trendline is fitted to Apple's historical closing prices and extended 30 trading days into the future. "
        f"The slope is {slope:.4f} USD/day, implying a 30-day extension near {trend_30_ext:.2f} if the linear drift persists. "
        "The dotted vertical line marks the last observed date to separate history from the simple projection. This extension "
        "is a baseline reference only and does not capture volatility or event-driven shocks."
    )
    pages.append((fig, 'Trend Extension', caption, desc))

    # Daily returns histogram
    returns = np.log(close / close.shift(1)).dropna()
    mu = returns.mean()
    sigma = returns.std()
    fig, ax = plt.subplots(figsize=medium_chart_figsize)
    n, bins, patches = ax.hist(returns, bins=30, density=True, alpha=0.6, color='tab:blue')
    x_vals = np.linspace(returns.min(), returns.max(), 200)
    if sigma > 0:
        # Manual normal PDF to avoid scipy dependency: f(x) = (1/(sigma*sqrt(2*pi))) * exp(-0.5*((x-mu)/sigma)**2)
        normal_pdf_values = (1 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x_vals - mu) / sigma) ** 2)
        ax.plot(x_vals, normal_pdf_values, 'r--', label='Normal PDF')
    ax.set_title('Daily Log Returns Histogram')
    ax.set_xlabel('Log Return')
    ax.set_ylabel('Density')
    ax.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=header_top_adjust, bottom=footer_bottom_adjust)
    caption = "Distribution of daily log returns with a normal fit overlay."
    desc = "This histogram shows the distribution of Apple's daily log returns, centered around zero for typical daily moves. The red dashed curve is a normal distribution fit, letting you compare the real return shape to the theoretical bell curve. It also helps spot asymmetry or heavier tails that indicate more extreme moves than normal."
    pages.append((fig, 'Returns Histogram', caption, desc))

    # 30-day rolling volatility (annualized)
    roll_vol = returns.rolling(window=30, min_periods=1).std() * math.sqrt(252)
    fig, ax = plt.subplots(figsize=compact_chart_figsize)
    ax.plot(dates[1:], roll_vol, color='tab:purple', label='30d Rolling Vol (annualized)')
    ax.set_title('30‑day Rolling Volatility (annualized)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility')
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=header_top_adjust, bottom=footer_bottom_adjust)
    caption = "Rolling 30-day (annualized) volatility of daily log returns."
    desc = "The 30-day rolling volatility (annualized) tracks how Apple's risk changes over time by measuring the variability of recent daily returns. Spikes reflect periods of heightened uncertainty and larger price swings, while flatter stretches indicate calmer trading conditions."
    pages.append((fig, 'Rolling Volatility', caption, desc))

    # 30-day projected price range
    proj_low_series = np.full(len(future_dates), proj_low)
    proj_high_series = np.full(len(future_dates), proj_high)
    last_price_series = np.full(len(future_dates), metrics['last_price'])
    fig, ax = plt.subplots(figsize=compact_chart_figsize)
    ax.plot(dates, close, label='Close', color='tab:blue', linewidth=1.0)
    ax.axvline(dates.iloc[-1], color='k', linewidth=0.6, linestyle=':')
    ax.fill_between(future_dates, proj_low_series, proj_high_series, alpha=0.25, color='tab:red', label='30d 1σ range')
    ax.plot(future_dates, last_price_series, '--', color='tab:orange', label='Last close')
    ax.set_title('30‑day Projected Price Range')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.grid(alpha=0.2)
    ax.legend()
    range_text = f"30d 1σ range:\n{proj_low:.2f} to {proj_high:.2f}"
    ax.text(
        0.02,
        0.98,
        range_text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.85),
    )
    fig.tight_layout()
    fig.subplots_adjust(top=header_top_adjust, bottom=footer_bottom_adjust)
    caption = "Projects a 30‑day price band using realized volatility (one standard deviation over 30 days, 1σ30)."
    desc = (
        f"The shaded band projects a 30-day price range of {proj_low:.2f} to {proj_high:.2f} using the latest close "
        f"({last_price:.2f}) and historical daily log-return volatility. Daily volatility (σ) is scaled by √30 to estimate "
        "σ30, and the band uses last_price × exp(±σ30) for the upper and lower bounds. It represents a probabilistic 1σ "
        "range if recent volatility persists, not a guaranteed forecast."
    )
    pages.append((fig, '30‑day Projected Range', caption, desc))

    # Volume chart
    fig, ax = plt.subplots(figsize=compact_chart_figsize)
    ax.bar(dates, df['Volume'], color='tab:gray')
    ax.set_title('Daily Trading Volume')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.grid(alpha=0.12)
    fig.tight_layout()
    fig.subplots_adjust(top=header_top_adjust, bottom=footer_bottom_adjust)
    caption = "Shares traded per day based on Yahoo Finance volume."
    desc = (
        "This chart displays Apple's daily trading volume, highlighting how much liquidity traded each session. "
        "Noticeable spikes often accompany earnings releases, macro headlines, or large price moves, and elevated "
        "volume during trend changes can signal stronger conviction from market participants."
    )
    pages.append((fig, 'Volume', caption, desc))

    # Now build final PDF: title page, TOC, then pages with headers/footers
    title_fig = plt.figure(figsize=(8.27, 11.69))
    title_fig.suptitle('Apple Inc. (AAPL)', fontsize=28, y=0.7)
    plt.axis('off')
    plt.text(0.5, 0.55, '1‑Year Price Analysis', ha='center', va='center', fontsize=18)
    plt.text(0.5, 0.45, f'Generated: {datetime.utcnow().date()}', ha='center', va='center', fontsize=10)

    # Build sections: combine selection, data collection, and major events, then data analysis
    date_range = f"{df['Date'].min()} to {df['Date'].max()}"

    # Combined Selection, Data Collection, and Major Events page
    combined_fig = plt.figure(figsize=(8.27, 11.69))
    combined_fig.suptitle('Section 1 — Selection, Data Collection & Major Events', fontsize=18, y=0.95)
    plt.axis('off')
    combined_paragraphs = [
        (
            "Apple Inc. (AAPL) was chosen because it is a widely held, highly liquid company that represents the consumer technology sector. "
            f"The review uses a full year of daily historical prices from Yahoo Finance, covering {date_range}, to capture recent market behavior. "
            "Major drivers during this period typically include quarterly earnings updates, product launch cycles, shifts in semiconductor supply and "
            "competition, and broader economic news such as interest-rate and inflation announcements that influence investor sentiment and trading volume."
        )
    ]
    y0 = 0.88
    for p in combined_paragraphs:
        wrapped = textwrap.fill(p, width=96)
        combined_fig.text(0.07, y0, wrapped, va='top', ha='left', fontsize=11)
        y0 -= 0.07

    # Data Analysis section intro page (summary text recreated here)
    summary_text = (
        "Section 2 focuses on Apple's price behavior over the past year using the charts that follow. Key metrics "
        "such as the latest close, central tendency, moving-average levels, trend slope, and volatility-based "
        "projection range are plotted directly on the figures so each statistic is tied to its visual context. "
        "Event drivers across the period include dividends and earnings updates, product-cycle news, macro "
        "developments like rates and inflation, and industry supply or competitive shifts. Limitations: the analysis "
        "uses historical realized volatility and a linear trend baseline, and it does not capture intraday or "
        "options-implied signals or future structural breaks."
    )
    analysis_fig = plt.figure(figsize=(8.27, 11.69))
    analysis_fig.suptitle('Section 2 — Data Analysis', fontsize=18, y=0.95)
    plt.axis('off')
    wrapped_summary = textwrap.fill(summary_text, width=96)
    plt.text(0.07, 0.85, wrapped_summary, va='top', ha='left', fontsize=11)

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

        # Section 2: Data Analysis (intro)
        pdf.savefig(analysis_fig)
        plt.close(analysis_fig)
        page_no += 1

        # content pages (charts)
        gen_date = datetime.utcnow().date().isoformat()
        for fig, short_title, caption, description in pages:
            # add consistent minimal header/footer
            fig.text(header_x, 0.98, 'Apple Inc. (AAPL)', ha='left', va='top', fontsize=8)
            fig.text(0.5, caption_y, caption, ha='center', va='top', fontsize=9, fontweight='bold')
            wrapped_desc = textwrap.fill(description, width=description_wrap_width)
            fig.text(0.5, description_y_position, wrapped_desc, ha='center', va='top', fontsize=10, color='black', linespacing=description_line_spacing)
            fig.text(0.98, 0.02, f'Page {page_no}', ha='right', va='bottom', fontsize=8)
            fig.text(footer_x, 0.02, f'Generated: {gen_date}', ha='left', va='bottom', fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)
            page_no += 1

    print(f'Report saved to {out_pdf}')

    print(f'Report saved to {out_pdf}')


if __name__ == '__main__':
    df = parse_raw('yahoo_history_raw.txt')
    metrics = compute_metrics(df)
    make_report(df, metrics, out_pdf='AAPL_report.pdf')

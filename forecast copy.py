"""
Ticker-based forecasting wrapper for Kronos.

Usage:
    PYTHONPATH=. python forecast.py AAPL
    PYTHONPATH=. python forecast.py CBA.AX
    PYTHONPATH=. python forecast.py MSFT --horizon 60

ASX tickers need the .AX suffix (e.g. CBA.AX, BHP.AX, WBC.AX).
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from model import Kronos, KronosTokenizer, KronosPredictor

warnings.filterwarnings("ignore")

# ---------- config ----------
LOOKBACK = 500          # trading days of history to feed the model (max 512)
DEFAULT_HORIZON = 30    # trading days to forecast (~6 weeks)
SAMPLE_COUNT = 5        # number of forecast paths — more = better uncertainty estimate, slower
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_ID = "NeoQuasar/Kronos-small"


def fetch_data(ticker: str, lookback: int) -> pd.DataFrame:
    """Pull daily OHLCV from yfinance and shape it for Kronos."""
    # Grab extra to be safe; we'll slice to `lookback` at the end.
    raw = yf.download(ticker, period="3y", interval="1d", progress=False, auto_adjust=False)
    if raw.empty:
        raise SystemExit(f"No data returned for ticker '{ticker}'. Check the symbol.")

    # yfinance sometimes returns MultiIndex columns — flatten.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = pd.DataFrame({
        "timestamps": raw.index,
        "open":   raw["Open"].values,
        "high":   raw["High"].values,
        "low":    raw["Low"].values,
        "close":  raw["Close"].values,
        "volume": raw["Volume"].values.astype(float),
        "amount": (raw["Close"].values * raw["Volume"].values),  # rough $ traded
    })
    df = df.dropna().reset_index(drop=True)
    if len(df) < lookback + 10:
        raise SystemExit(f"Not enough history for {ticker}: got {len(df)} rows, need {lookback}+.")
    return df.tail(lookback).reset_index(drop=True)


def make_future_timestamps(last_ts: pd.Timestamp, n: int) -> pd.Series:
    """Generate n future business-day timestamps after last_ts."""
    return pd.Series(pd.bdate_range(start=last_ts + pd.Timedelta(days=1), periods=n))


def summarize(ticker: str, hist: pd.DataFrame, samples: list[pd.DataFrame], horizon: int) -> str:
    """Produce a plain-English summary from the forecast distribution."""
    last_close = float(hist["close"].iloc[-1])
    last_date = hist["timestamps"].iloc[-1].strftime("%Y-%m-%d")

    # Stack all sample closing-price paths: shape (sample_count, horizon)
    closes = np.stack([s["close"].values for s in samples])
    final_prices = closes[:, -1]

    mean_final = float(final_prices.mean())
    median_final = float(np.median(final_prices))
    p10 = float(np.percentile(final_prices, 10))
    p90 = float(np.percentile(final_prices, 90))

    expected_return = (mean_final / last_close - 1) * 100
    prob_up = float((final_prices > last_close).mean()) * 100

    # Per-day volatility of the mean forecast path
    mean_path = closes.mean(axis=0)
    daily_returns = np.diff(np.log(mean_path))
    forecast_vol_annual = float(np.std(daily_returns) * np.sqrt(252) * 100)

    # Historical realised vol for comparison
    hist_returns = np.diff(np.log(hist["close"].values[-60:]))
    hist_vol_annual = float(np.std(hist_returns) * np.sqrt(252) * 100)

    # Direction label
    if abs(expected_return) < 1.5:
        direction = "broadly flat"
    elif expected_return > 0:
        direction = "modestly higher" if expected_return < 5 else "meaningfully higher"
    else:
        direction = "modestly lower" if expected_return > -5 else "meaningfully lower"

    # Confidence label
    spread_pct = (p90 - p10) / last_close * 100
    if spread_pct < 8:
        confidence = "with a relatively tight distribution"
    elif spread_pct < 20:
        confidence = "with a moderately wide range of outcomes"
    else:
        confidence = "with a very wide range of outcomes — treat with caution"

    lines = [
        f"=== Forecast summary: {ticker} ===",
        f"As of close on {last_date}, {ticker} traded at ${last_close:.2f}.",
        f"",
        f"Over the next {horizon} trading days (~{horizon * 7 // 5} calendar days),",
        f"Kronos' {SAMPLE_COUNT}-path forecast points {direction} {confidence}.",
        f"",
        f"Central forecast (mean of paths):  ${mean_final:.2f}  ({expected_return:+.1f}%)",
        f"Median forecast:                   ${median_final:.2f}",
        f"10th–90th percentile range:        ${p10:.2f} – ${p90:.2f}",
        f"Share of paths ending above today: {prob_up:.0f}%",
        f"",
        f"Forecast annualised volatility:    {forecast_vol_annual:.1f}%",
        f"Recent realised volatility (60d):  {hist_vol_annual:.1f}%",
        f"",
        "Important context:",
        "  • Kronos forecasts candle patterns; it has no knowledge of earnings,",
        "    news, macro events, or anything fundamental about the company.",
        "  • A single forecast is not a trade recommendation. The wider the",
        "    percentile range, the less the model is committing to a direction.",
        "  • Past performance (and forecast performance) does not predict",
        "    future results. This is for personal research only.",
    ]
    return "\n".join(lines)


def plot_forecast(ticker: str, hist: pd.DataFrame, samples: list[pd.DataFrame],
                  future_ts: pd.Series, out_path: str) -> None:
    """Plot recent history plus forecast fan chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Show last ~120 days of history for context
    recent = hist.tail(120)
    ax.plot(recent["timestamps"], recent["close"], color="black", linewidth=1.5, label="Historical close")

    # Plot each sample path lightly
    closes = np.stack([s["close"].values for s in samples])
    for path in closes:
        ax.plot(future_ts, path, color="steelblue", alpha=0.25, linewidth=1)

    # Mean forecast bold
    mean_path = closes.mean(axis=0)
    ax.plot(future_ts, mean_path, color="steelblue", linewidth=2, label="Mean forecast")

    # 10–90 percentile band
    p10 = np.percentile(closes, 10, axis=0)
    p90 = np.percentile(closes, 90, axis=0)
    ax.fill_between(future_ts, p10, p90, color="steelblue", alpha=0.15, label="10–90% range")

    # Vertical line at the forecast boundary
    ax.axvline(hist["timestamps"].iloc[-1], color="grey", linestyle="--", alpha=0.5)

    ax.set_title(f"{ticker} — Kronos forecast ({len(future_ts)} trading days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close price")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Chart saved to {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Forecast a stock with Kronos.")
    parser.add_argument("ticker", help="Ticker symbol (e.g. AAPL, MSFT, CBA.AX, BHP.AX)")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                        help=f"Trading days to forecast (default {DEFAULT_HORIZON})")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    horizon = args.horizon

    print(f"\nFetching data for {ticker}...")
    hist = fetch_data(ticker, LOOKBACK)
    print(f"  Got {len(hist)} days, from {hist['timestamps'].iloc[0].date()} "
          f"to {hist['timestamps'].iloc[-1].date()}.")

    print(f"\nLoading Kronos model...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID)
    model = Kronos.from_pretrained(MODEL_ID)
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    x_df = hist[["open", "high", "low", "close", "volume", "amount"]]
    x_ts = hist["timestamps"]
    y_ts = make_future_timestamps(hist["timestamps"].iloc[-1], horizon)

    print(f"Running {SAMPLE_COUNT} forecast paths ({horizon} days each)...")
    samples = []
    for i in range(SAMPLE_COUNT):
        pred = predictor.predict(
            df=x_df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=horizon,
            T=1.0,
            top_p=0.9,
            sample_count=1,
        )
        samples.append(pred)
        print(f"  path {i+1}/{SAMPLE_COUNT} done")

    # Summary
    print()
    summary = summarize(ticker, hist, samples, horizon)
    print(summary)

    # Save summary to file
    summary_path = f"forecast_{ticker.replace('.', '_')}.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\nSummary saved to {summary_path}")

    # Plot
    chart_path = f"forecast_{ticker.replace('.', '_')}.png"
    plot_forecast(ticker, hist, samples, y_ts, chart_path)


if __name__ == "__main__":
    main()

"""
Backtest Kronos forecasts against actual historical outcomes.

For each ticker, this walks back through history, makes a forecast from a point
in the past, and compares it to what actually happened. Aggregates metrics
across many tickers and many forecast origins.

Usage:
    PYTHONPATH=. python backtest.py
    PYTHONPATH=. python backtest.py --horizon 20 --origins 6

Edit the TICKERS list below to choose what to test.
"""

import argparse
import warnings
import json
import numpy as np
import pandas as pd
import yfinance as yf

from model import Kronos, KronosTokenizer, KronosPredictor

warnings.filterwarnings("ignore")

# ---------- config ----------
TICKERS = [
    # US
    "AAPL", "MSFT", "NVDA", "JPM", "XOM",
    # ASX (note .AX suffix)
    "CBA.AX", "BHP.AX", "CSL.AX", "MQG.AX", "WES.AX",
]
LOOKBACK = 400          # days of history fed to model per forecast
HORIZON = 20            # trading days forecast (~1 month)
ORIGINS = 6             # how many past forecast points per ticker
ORIGIN_GAP = 20         # trading days between forecast origins
SAMPLE_COUNT = 5        # forecast paths per origin
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_ID = "NeoQuasar/Kronos-small"


def fetch(ticker):
    raw = yf.download(ticker, period="5y", interval="1d", progress=False, auto_adjust=False)
    if raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = pd.DataFrame({
        "timestamps": raw.index,
        "open": raw["Open"].values, "high": raw["High"].values,
        "low": raw["Low"].values, "close": raw["Close"].values,
        "volume": raw["Volume"].values.astype(float),
        "amount": (raw["Close"].values * raw["Volume"].values),
    }).dropna().reset_index(drop=True)
    return df


def backtest_ticker(predictor, ticker, df):
    """Run several historical forecasts for one ticker, return per-origin records."""
    records = []
    n = len(df)
    # Need: LOOKBACK history + HORIZON actual future, for each origin
    needed = LOOKBACK + HORIZON
    if n < needed + ORIGINS * ORIGIN_GAP:
        print(f"  {ticker}: not enough history ({n} rows), skipping.")
        return records

    # Forecast origins, working backwards from the most recent point that still
    # has HORIZON days of *actual* data after it.
    latest_origin = n - HORIZON
    origins = [latest_origin - i * ORIGIN_GAP for i in range(ORIGINS)]
    origins = [o for o in origins if o - LOOKBACK >= 0]

    for o in origins:
        hist = df.iloc[o - LOOKBACK:o].reset_index(drop=True)
        actual = df.iloc[o:o + HORIZON].reset_index(drop=True)

        x_df = hist[["open", "high", "low", "close", "volume", "amount"]]
        x_ts = hist["timestamps"]
        y_ts = actual["timestamps"]

        closes = []
        for _ in range(SAMPLE_COUNT):
            pred = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                                     pred_len=HORIZON, T=1.0, top_p=0.9, sample_count=1)
            closes.append(pred["close"].values)
        closes = np.stack(closes)               # (samples, horizon)
        mean_path = closes.mean(axis=0)

        last_close = float(hist["close"].iloc[-1])
        actual_final = float(actual["close"].iloc[-1])
        pred_final = float(mean_path[-1])

        pred_dir = np.sign(pred_final - last_close)
        actual_dir = np.sign(actual_final - last_close)
        direction_correct = bool(pred_dir == actual_dir)

        # % error on final price
        pct_error = abs(pred_final - actual_final) / actual_final * 100

        # Calibration: did actual final land inside the 10-90% band?
        p10 = np.percentile(closes[:, -1], 10)
        p90 = np.percentile(closes[:, -1], 90)
        in_band = bool(p10 <= actual_final <= p90)

        records.append({
            "ticker": ticker,
            "origin_date": str(hist["timestamps"].iloc[-1].date()),
            "last_close": round(last_close, 2),
            "pred_final": round(pred_final, 2),
            "actual_final": round(actual_final, 2),
            "pred_return_pct": round((pred_final / last_close - 1) * 100, 2),
            "actual_return_pct": round((actual_final / last_close - 1) * 100, 2),
            "direction_correct": direction_correct,
            "pct_error": round(pct_error, 2),
            "in_80pct_band": in_band,
        })
        print(f"  {ticker} @ {records[-1]['origin_date']}: "
              f"pred {records[-1]['pred_return_pct']:+.1f}% vs actual "
              f"{records[-1]['actual_return_pct']:+.1f}%  "
              f"{'✓' if direction_correct else '✗'} dir, "
              f"{'in' if in_band else 'out'} band")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--origins", type=int, default=ORIGINS)
    args = ap.parse_args()
    globals()["HORIZON"] = args.horizon
    globals()["ORIGINS"] = args.origins

    print("Loading Kronos...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID)
    model = Kronos.from_pretrained(MODEL_ID)
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    all_records = []
    for ticker in TICKERS:
        print(f"\n=== {ticker} ===")
        df = fetch(ticker)
        if df is None:
            print(f"  no data, skipping.")
            continue
        all_records.extend(backtest_ticker(predictor, ticker, df))

    if not all_records:
        print("\nNo results produced.")
        return

    res = pd.DataFrame(all_records)
    res.to_csv("backtest_results.csv", index=False)

    # ---- aggregate metrics ----
    n = len(res)
    dir_acc = res["direction_correct"].mean() * 100
    mean_err = res["pct_error"].mean()
    median_err = res["pct_error"].median()
    band_cov = res["in_80pct_band"].mean() * 100  # ideal ~80%

    # Per-market breakdown
    res["market"] = res["ticker"].apply(lambda t: "ASX" if t.endswith(".AX") else "US")
    by_market = res.groupby("market").agg(
        n=("ticker", "size"),
        dir_acc=("direction_correct", lambda x: round(x.mean() * 100, 1)),
        mean_err=("pct_error", lambda x: round(x.mean(), 2)),
    )

    summary = {
        "total_forecasts": n,
        "horizon_days": args.horizon,
        "sample_paths_each": SAMPLE_COUNT,
        "directional_accuracy_pct": round(dir_acc, 1),
        "mean_abs_pct_error": round(mean_err, 2),
        "median_abs_pct_error": round(median_err, 2),
        "band_coverage_pct": round(band_cov, 1),
        "band_coverage_ideal": 80.0,
    }

    print("\n" + "=" * 50)
    print("AGGREGATE RESULTS")
    print("=" * 50)
    print(json.dumps(summary, indent=2))
    print("\nBy market:")
    print(by_market.to_string())
    print("\nInterpretation:")
    print(f"  • Directional accuracy of {dir_acc:.0f}% "
          f"({'better than' if dir_acc > 50 else 'no better than'} a coin flip).")
    print(f"  • Typical price miss at {args.horizon}d: ~{median_err:.1f}%.")
    print(f"  • {band_cov:.0f}% of actuals fell in the model's 80% band "
          f"({'well calibrated' if 70 <= band_cov <= 90 else 'mis-calibrated'}).")

    with open("backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved: backtest_results.csv, backtest_summary.json")


if __name__ == "__main__":
    main()

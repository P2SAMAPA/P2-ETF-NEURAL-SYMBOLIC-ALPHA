"""
Trainer for Neural Symbolic Alpha: per ETF, test three windows, pick the one maximizing correlation
between predicted alpha and actual next‑day return, then save best expression.
Ranking is biased toward higher‑return ETFs by combining correlation and average return.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import config
import data_manager
from symbolic_alpha import SymbolicAlphaMiner
import push_results

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty:
            continue

        universe_results = {}

        for ticker in tickers:
            if ticker not in returns.columns:
                continue
            series = returns[ticker].dropna()
            if len(series) < max(config.WINDOWS) + 100:
                print(f"  {ticker}: insufficient data")
                continue

            # Optional: skip negative average return ETFs
            # avg_ret = series.mean() * 252
            # if avg_ret <= 0:
            #     print(f"  {ticker}: negative average return, skipping")
            #     continue

            best_corr = -np.inf
            best_window = None
            best_miner = None
            best_expression = None
            best_complexity = None
            best_mse = None

            for win in config.WINDOWS:
                print(f"    Testing window {win} for {ticker}")
                miner = SymbolicAlphaMiner(
                    window=win,
                    niterations=config.NITERATIONS,
                    populations=config.POPULATIONS,
                    parsimony=config.PARSIMONY,
                    max_complexity=config.MAX_COMPLEXITY,
                    binary_operators=["+", "-", "*", "/"],
                    unary_operators=["square", "sqrt", "log1p", "tanh", "sin", "cos"],
                    feature_names=[f"lag_{l}" for l in config.FEATURE_LAGS]
                )
                success = miner.fit(series, config.FEATURE_LAGS)
                if not success or miner.model is None:
                    continue

                # Validation on last 50 days of the training window
                X_val = []
                y_val = []
                values = series.values
                lags = config.FEATURE_LAGS
                for i in range(max(lags), len(values) - 1):
                    if i >= len(values) - 50:
                        row = [values[i - lag] for lag in lags]
                        X_val.append(row)
                        y_val.append(values[i + 1])
                if len(X_val) < 10:
                    continue

                try:
                    y_pred = miner.model.predict(X_val)
                except Exception:
                    continue

                if np.std(y_pred) < 1e-8:
                    continue
                corr = np.corrcoef(y_pred, y_val)[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    best_window = win
                    best_miner = miner
                    best_expression = miner.best_expression_
                    best_complexity = miner.complexity_
                    best_mse = miner.mse_

            if best_miner is None:
                print(f"  {ticker}: no valid model")
                continue

            # Compute average return over the same validation period
            values = series.values
            lags = config.FEATURE_LAGS
            y_val = []
            for i in range(max(lags), len(values) - 1):
                if i >= len(values) - 50:
                    y_val.append(values[i + 1])
            avg_return = np.mean(y_val) if y_val else 0.0

            print(f"  {ticker}: best window {best_window} days, correlation {best_corr:.3f}, avg_return {avg_return:.6f}")
            universe_results[ticker] = {
                "expression": str(best_expression),
                "complexity": int(best_complexity),
                "mse": float(best_mse),
                "validation_correlation": float(best_corr),
                "avg_return": float(avg_return),
                "selected_window": best_window
            }

        # Rank ETFs by combined score = correlation * avg_return
        # (Higher returns and higher predictability get higher rank)
        for ticker, data in universe_results.items():
            data["combined_score"] = data["validation_correlation"] * data["avg_return"]

        sorted_etfs = sorted(universe_results.items(), key=lambda x: x[1]["combined_score"], reverse=True)
        top_etfs = [{"ticker": t, "combined_score": v["combined_score"]} for t, v in sorted_etfs[:config.TOP_N]]
        all_results[universe_name] = {
            "top_expressions": top_etfs,
            "all_tickers": universe_results
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/symbolic_alpha_{config.TODAY}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": config.TODAY, "universes": all_results}, f, indent=2)

    push_results.push_daily_result(local_path)
    print("\n=== Neural Symbolic Alpha complete (biased toward higher-return ETFs) ===")

if __name__ == "__main__":
    main()

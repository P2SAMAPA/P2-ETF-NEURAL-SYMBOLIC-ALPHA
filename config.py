"""
Configuration for P2-ETF-NEURAL-SYMBOLIC-ALPHA engine.
"""

import os
from datetime import datetime

# --- Hugging Face ---
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
DATA_FILE = "master_data.parquet"
OUTPUT_REPO = "P2SAMAPA/p2-etf-neural-symbolic-alpha-results"

# --- Universe definitions ---
FI_COMMODITIES = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
    "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
]
COMBINED = list(set(FI_COMMODITIES + EQUITY_SECTORS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES,
    "EQUITY_SECTORS": EQUITY_SECTORS,
    "COMBINED": COMBINED
}

# --- Macro features (optional, but can be used) ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- Symbolic regression parameters ---
WINDOWS = [60, 120, 252]                     # rolling training windows (days)
LOOKAHEAD = 1                                # predict next day's return
NITERATIONS = 500                            # PySR iterations
POPULATIONS = 30                             # number of populations
PARSIMONY = 1.0                              # complexity penalty
MAX_COMPLEXITY = 20                          # maximum expression complexity
OPERATORS = ["+", "-", "*", "/", "square", "sqrt", "log1p", "tanh", "sin", "cos"]
FEATURE_LAGS = [1, 2, 3, 5, 10, 20]          # lags of returns used as features (days)

TOP_N = 3                                    # top expressions to save per universe

# --- Output ---
TODAY = datetime.now().strftime("%Y-%m-%d")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

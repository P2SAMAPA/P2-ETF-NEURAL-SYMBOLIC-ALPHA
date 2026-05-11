# P2-ETF-NEURAL-SYMBOLIC-ALPHA

**Neural symbolic regression** using PySR. Discovers interpretable closed‑form alpha factors for each ETF, ranking them by predictive power.

## Features

- **PySR backend** – GPU‑accelerated, regularised evolutionary search.
- Tests three rolling windows (60, 120, 252 days), picks the window that maximises correlation with next‑day return.
- Builds features from multiple lags of returns (configurable).
- Outputs per‑ETF mathematical expression, complexity, MSE, and validation correlation.
- Dashboard shows top formulas per universe and full table.

## Data

Uses `P2SAMAPA/fi-etf-macro-signal-master-data`.  
Results stored in `P2SAMAPA/p2-etf-neural-symbolic-alpha-results`.

## Installation

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-NEURAL-SYMBOLIC-ALPHA.git
cd P2-ETF-NEURAL-SYMBOLIC-ALPHA
pip install -r requirements.txt
# Julia will be installed automatically by PySR on first run

GitHub Actions
Daily run at 23:00 UTC weekdays. Set HF_TOKEN secret.
Note: The first run will take longer because it installs Julia and PySR packages; subsequent runs will use caching.

Configuration
Edit config.py to change lags, windows, number of iterations, operators, etc.

References
Cranmer et al. (2020) – PySR: High‑performance symbolic regression.

Udrescu & Tegmark (2020) – AI Feynman.

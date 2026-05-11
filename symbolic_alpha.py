"""
Neural Symbolic Regression using PySR.
Discovers closed‑form alpha factors per ETF.
"""

import numpy as np
import pandas as pd
from pysr import PySRRegressor
import warnings

class SymbolicAlphaMiner:
    def __init__(self, window=252, niterations=500, populations=30, parsimony=1.0,
                 max_complexity=20, operators=None, feature_names=None):
        self.window = window
        self.niterations = niterations
        self.populations = populations
        self.parsimony = parsimony
        self.max_complexity = max_complexity
        self.operators = operators or ["+", "-", "*", "/", "square", "sqrt", "log1p", "tanh", "sin", "cos"]
        self.feature_names = feature_names
        self.model = None
        self.best_expression_ = None
        self.complexity_ = None
        self.mse_ = None
        # Store lag order used during fit, for predict
        self._feature_lags = None

    def _prepare_features(self, returns_series, feature_lags):
        """
        Build feature matrix X (each row corresponds to a date)
        and target y (next day return).
        """
        X = []
        y = []
        values = returns_series.values
        for i in range(max(feature_lags), len(values) - 1):
            row = []
            for lag in feature_lags:
                row.append(values[i - lag])
            X.append(row)
            y.append(values[i + 1])   # tomorrow's return
        X = np.array(X)
        y = np.array(y)
        return X, y

    def fit(self, returns_series, feature_lags):
        """
        Run symbolic regression on the last `window` days of returns.
        """
        # Use only the most recent `window` days
        series = returns_series.iloc[-self.window:]
        X, y = self._prepare_features(series, feature_lags)
        if len(X) < 50:
            return False

        # Store feature lags and names for later prediction
        self._feature_lags = feature_lags
        if self.feature_names is None:
            self.feature_names = [f"lag_{lag}" for lag in feature_lags]
        # For predict, we need the same names
        self._feature_names_internal = self.feature_names

        # PySR model with CORRECT loss parameter (lowercase "mse")
        model = PySRRegressor(
            niterations=self.niterations,
            populations=self.populations,
            binary_operators=self.operators,       # only binary operators here
            unary_operators=["square", "sqrt", "log1p", "tanh", "sin", "cos"],
            parsimony=self.parsimony,
            maxsize=self.max_complexity,
            elementwise_loss="mse",                # ✅ FIXED: was loss="MSE"
            model_selection="best",
            progress=False,
            verbosity=0,
            output_directory="pysr_cache"
        )
        try:
            model.fit(X, y, variable_names=self.feature_names)
            # Get best equation
            best = model.get_best()
            if best is None:
                return False
            self.best_expression_ = model.sympy()
            self.complexity_ = best["complexity"]
            self.mse_ = best["loss"]
            self.model = model
            return True
        except Exception as e:
            warnings.warn(f"PySR failed: {e}")
            return False

    def predict(self, last_returns):
        """
        Apply the discovered formula to the most recent feature vector
        (last `feature_lags` returns) to generate the alpha factor.
        `last_returns` should be a list or array of length equal to the number of lags.
        """
        if self.model is None or self._feature_lags is None:
            return 0.0

        # Ensure input is a 2D array (one row)
        last_returns = np.asarray(last_returns).reshape(1, -1)
        try:
            return float(self.model.predict(last_returns)[0])
        except Exception:
            return 0.0

    def get_expression_string(self):
        if self.best_expression_ is None:
            return ""
        return str(self.best_expression_)

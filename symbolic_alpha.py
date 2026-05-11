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
                 max_complexity=20, binary_operators=None, unary_operators=None, feature_names=None):
        self.window = window
        self.niterations = niterations
        self.populations = populations
        self.parsimony = parsimony
        self.max_complexity = max_complexity
        # Default binary operators (only proper binary operators)
        self.binary_operators = binary_operators or ["+", "-", "*", "/"]
        # Default unary operators
        self.unary_operators = unary_operators or ["square", "sqrt", "log1p", "tanh", "sin", "cos"]
        self.feature_names = feature_names
        self.model = None
        self.best_expression_ = None
        self.complexity_ = None
        self.mse_ = None
        self._feature_lags = None

    def _prepare_features(self, returns_series, feature_lags):
        X = []
        y = []
        values = returns_series.values
        for i in range(max(feature_lags), len(values) - 1):
            row = [values[i - lag] for lag in feature_lags]
            X.append(row)
            y.append(values[i + 1])
        return np.array(X), np.array(y)

    def fit(self, returns_series, feature_lags):
        series = returns_series.iloc[-self.window:]
        X, y = self._prepare_features(series, feature_lags)
        if len(X) < 50:
            return False

        self._feature_lags = feature_lags
        if self.feature_names is None:
            self.feature_names = [f"lag_{lag}" for lag in feature_lags]

        # PySR model – separate binary and unary operators
        model = PySRRegressor(
            niterations=self.niterations,
            populations=self.populations,
            binary_operators=self.binary_operators,   # only + - * /
            unary_operators=self.unary_operators,     # square, sqrt, log1p, tanh, sin, cos
            parsimony=self.parsimony,
            maxsize=self.max_complexity,
            model_selection="best",
            progress=False,
            verbosity=0,
            output_directory="pysr_cache"
        )
        try:
            model.fit(X, y, variable_names=self.feature_names)
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
        if self.model is None or self._feature_lags is None:
            return 0.0
        last_returns = np.asarray(last_returns).reshape(1, -1)
        try:
            return float(self.model.predict(last_returns)[0])
        except Exception:
            return 0.0

    def get_expression_string(self):
        if self.best_expression_ is None:
            return ""
        return str(self.best_expression_)

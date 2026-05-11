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

    def _prepare_features(self, returns_series, feature_lags):
        """
        Build feature matrix X (each row corresponds to a date)
        and target y (next day return).
        """
        X = []
        y = []
        dates = returns_series.index
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

        # Create feature names
        if self.feature_names is None:
            self.feature_names = [f"lag_{lag}" for lag in feature_lags]

        # PySR model
        model = PySRRegressor(
            niterations=self.niterations,
            populations=self.populations,
            binary_operators=self.operators,
            unary_operators=["square", "sqrt", "log1p", "tanh", "sin", "cos"],
            parsimony=self.parsimony,
            maxsize=self.max_complexity,
            loss="MSE",
            model_selection="best",
            progress=False,
            verbosity=0,
            output_directory="pysr_cache"
        )
        try:
            model.fit(X, y, variable_names=self.feature_names)
            # Get best equation
            self.best_expression_ = model.sympy()
            self.complexity_ = model.get_best()["complexity"]
            self.mse_ = model.get_best()["loss"]
            self.model = model
            return True
        except Exception as e:
            warnings.warn(f"PySR failed: {e}")
            return False

    def predict(self, last_returns):
        """
        Apply the discovered formula to the most recent feature vector
        (last `feature_lags` returns) to generate the alpha factor.
        """
        if self.model is None:
            return 0.0
        # Build the feature vector: lags in the same order as during training
        # We need to know the lag order; store during fit
        # For simplicity, we use the model.predict directly if we have the same X structure.
        # In PySR, we can pass a single row as a DataFrame with column names.
        if not hasattr(self, 'feature_names_'):
            return 0.0
        # Assuming last_returns is a dict or array of length len(feature_names_)
        return float(self.model.predict([last_returns])[0])

    def get_expression_string(self):
        if self.best_expression_ is None:
            return ""
        return str(self.best_expression_)

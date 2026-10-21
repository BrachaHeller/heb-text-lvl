"""Stateful feature selection, log1p transformation, MinMaxScaler, and directional inversion."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from hebrew_readability.config import PipelineConfig


class HebrewReadabilityPreprocessor(BaseEstimator, TransformerMixin):
    """Fits log-skew, MinMax [0, 1], and directional inversions (1.0 - x) on Train only."""

    def __init__(
        self,
        selected_base_features: Optional[List[str]] = None,
        inverse_base_features: Optional[List[str]] = None,
        skew_threshold: float = 1.0,
    ):
        default_cfg = PipelineConfig()
        self.selected_base_features = (
            selected_base_features or default_cfg.selected_base_features
        )
        self.inverse_base_features = inverse_base_features or default_cfg.inverse_base_features
        self.skew_threshold = skew_threshold

        self.skewed_cols_: List[str] = []
        self.scaler_: Optional[MinMaxScaler] = None
        self.feature_names_out_: List[str] = []
        self.log_map_: Dict[str, str] = {}
        self.inv_map_: Dict[str, str] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "HebrewReadabilityPreprocessor":
        df = X[self.selected_base_features].copy()

        self.skewed_cols_ = [
            col for col in df.columns if abs(df[col].skew()) > self.skew_threshold
        ]
        self.log_map_ = {
            col: f"{col}_log" if col in self.skewed_cols_ else col for col in df.columns
        }

        for col in self.skewed_cols_:
            df[self.log_map_[col]] = np.log1p(np.maximum(0.0, df[col]))
            df = df.drop(columns=[col])

        ordered_log_cols = [self.log_map_[col] for col in self.selected_base_features]
        self.scaler_ = MinMaxScaler(clip=True)
        self.scaler_.fit(df[ordered_log_cols])

        self.inv_map_ = {}
        for base_col in self.selected_base_features:
            logged_col = self.log_map_[base_col]
            if base_col in self.inverse_base_features:
                self.inv_map_[logged_col] = f"{logged_col}_inv"
            else:
                self.inv_map_[logged_col] = logged_col

        self.feature_names_out_ = [
            self.inv_map_[self.log_map_[col]] for col in self.selected_base_features
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_ is None:
            raise RuntimeError("HebrewReadabilityPreprocessor must be fitted before transform().")

        df = X[self.selected_base_features].copy()

        for col in self.skewed_cols_:
            df[self.log_map_[col]] = np.log1p(np.maximum(0.0, df[col]))
            df = df.drop(columns=[col])

        ordered_log_cols = [self.log_map_[col] for col in self.selected_base_features]
        df[ordered_log_cols] = self.scaler_.transform(df[ordered_log_cols])

        for base_col in self.selected_base_features:
            logged_col = self.log_map_[base_col]
            final_col = self.inv_map_[logged_col]
            if base_col in self.inverse_base_features:
                df[final_col] = 1.0 - df[logged_col]
                df = df.drop(columns=[logged_col])

        return df[self.feature_names_out_]
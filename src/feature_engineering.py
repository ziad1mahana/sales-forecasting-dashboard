"""
feature_engineering.py
======================
Time series feature engineering pipeline designed to prevent data leakage.
Constructs lag features, rolling window aggregates, cyclical calendar representations,
and momentum indicators strictly using historical information.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


class TimeSeriesFeatureEngineer:
    """
    Transforms raw continuous time series data into ML-ready feature matrices
    with guaranteed zero lookahead bias.
    
    IMPORTANT: Current-day operational targets (such as daily Quantity, Profit, Orders)
    are strictly excluded or lagged to ensure no future information is accessible
    during training or inference.
    """

    def __init__(
        self,
        target_col: str = "Sales",
        date_col: str = "Date",
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
    ):
        self.target_col = target_col
        self.date_col = date_col
        self.lags = lags or [1, 2, 3, 7, 14, 21, 28, 30]
        self.rolling_windows = rolling_windows or [7, 14, 30]
        self.feature_names: List[str] = []

    def build_features(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Creates temporal, lag, rolling, cyclical, and momentum features.
        All rolling and lag statistics are derived strictly from lagged past values.
        Any ancillary columns present in df_in (e.g. Quantity, Profit) are either
        dropped or converted into lagged features to strictly prevent lookahead bias.
        """
        # Isolate Date and Target columns first to prevent leaking contemporaneous aggregates
        df = pd.DataFrame({
            self.date_col: pd.to_datetime(df_in[self.date_col]),
            self.target_col: pd.to_numeric(df_in[self.target_col], errors="coerce").fillna(0.0),
        })

        df = df.sort_values(self.date_col).reset_index(drop=True)

        # 1. Calendar Features (Derived strictly from the date index)
        df["day_of_week"] = df[self.date_col].dt.dayofweek
        df["day_of_month"] = df[self.date_col].dt.day
        df["month"] = df[self.date_col].dt.month
        df["quarter"] = df[self.date_col].dt.quarter
        df["year"] = df[self.date_col].dt.year
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["day_of_year"] = df[self.date_col].dt.dayofyear
        df["week_of_year"] = df[self.date_col].dt.isocalendar().week.astype(int)

        # 2. Cyclical Encodings (Trigonometric representations)
        # Guarantees smooth continuity across periodic time boundaries
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
        df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["sin_day"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
        df["cos_day"] = np.cos(2 * np.pi * df["day_of_month"] / 31)

        # 3. Pure Lag Features (Past target values)
        for lag in self.lags:
            df[f"lag_{lag}"] = df[self.target_col].shift(lag)

        # 4. Rolling Window Statistics
        # CRITICAL LEAKAGE PREVENTION:
        # Applied to shift(1) so that day t's target value is NEVER included
        # in the rolling window statistics computed to predict day t.
        shifted_target = df[self.target_col].shift(1)
        for window in self.rolling_windows:
            df[f"rolling_mean_{window}"] = shifted_target.rolling(window=window).mean()
            df[f"rolling_std_{window}"] = shifted_target.rolling(window=window).std()
            df[f"rolling_min_{window}"] = shifted_target.rolling(window=window).min()
            df[f"rolling_max_{window}"] = shifted_target.rolling(window=window).max()

        # 5. Momentum & Interaction Features
        if 7 in self.rolling_windows and 30 in self.rolling_windows:
            df["momentum_7_30"] = df["rolling_mean_7"] / (df["rolling_mean_30"] + 1e-5)
        if 1 in self.lags and 7 in self.lags:
            df["sales_diff_1_7"] = df["lag_1"] - df["lag_7"]

        # Drop rows where initial rolling or lag windows are NaN
        df_clean = df.dropna().reset_index(drop=True)

        # Save feature list excluding Date and Target
        excluded_cols = [self.date_col, self.target_col]
        self.feature_names = [c for c in df_clean.columns if c not in excluded_cols]

        return df_clean

    def chronological_train_test_split(
        self,
        df_feat: pd.DataFrame,
        test_size: int = 90,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Splits the feature dataset chronologically without shuffling.

        Methodology Rationale:
        Random shuffling destroys temporal order and allows future information
        to inform past predictions. Chronological splitting simulates genuine deployment
        conditions where the model forecasts unseen future dates.
        """
        if test_size >= len(df_feat):
            raise ValueError(f"test_size ({test_size}) cannot exceed dataset length ({len(df_feat)})")

        train_df = df_feat.iloc[:-test_size].copy()
        test_df = df_feat.iloc[-test_size:].copy()

        X_train = train_df[self.feature_names]
        y_train = train_df[self.target_col]
        dates_train = train_df[self.date_col]

        X_test = test_df[self.feature_names]
        y_test = test_df[self.target_col]
        dates_test = test_df[self.date_col]

        return X_train, X_test, y_train, y_test, dates_train, dates_test


if __name__ == "__main__":
    from data_processor import SalesDataProcessor

    processor = SalesDataProcessor()
    daily_df = processor.get_daily_series()
    engineer = TimeSeriesFeatureEngineer()
    feat_df = engineer.build_features(daily_df)

    print(f"Engineered features dataframe shape: {feat_df.shape}")
    print(f"Number of generated features: {len(engineer.feature_names)}")
    print(f"Features list:\n{engineer.feature_names}")

    X_train, X_test, y_train, y_test, d_train, d_test = engineer.chronological_train_test_split(
        feat_df, test_size=90
    )
    print(f"\nChronological Split Verification:")
    print(f"Train period: {d_train.min().date()} to {d_train.max().date()} ({len(X_train)} samples)")
    print(f"Test period:  {d_test.min().date()} to {d_test.max().date()} ({len(X_test)} samples)")

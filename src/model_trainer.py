"""
model_trainer.py
================
Model training, evaluation, comparison, recursive out-of-sample forecasting,
and model persistence module for Sales Forecasting & Business Analytics.
"""

from typing import Dict, Any, Tuple, List
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from data_processor import SalesDataProcessor
from feature_engineering import TimeSeriesFeatureEngineer


class SalesForecastingTrainer:
    """
    Orchestrates the training, comparison, recursive multi-step forecasting,
    and serialization of sales forecasting models.
    """

    def __init__(
        self,
        data_path: str = "data/sales.csv",
        test_size: int = 90,
        random_state: int = 42,
    ):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state

        self.processor = SalesDataProcessor(data_path)
        self.engineer = TimeSeriesFeatureEngineer()
        
        self.daily_df: pd.DataFrame = pd.DataFrame()
        self.feat_df: pd.DataFrame = pd.DataFrame()
        self.X_train: pd.DataFrame = pd.DataFrame()
        self.X_test: pd.DataFrame = pd.DataFrame()
        self.y_train: pd.Series = pd.Series(dtype=float)
        self.y_test: pd.Series = pd.Series(dtype=float)
        self.dates_train: pd.Series = pd.Series(dtype="datetime64[ns]")
        self.dates_test: pd.Series = pd.Series(dtype="datetime64[ns]")

        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.best_model_name: str = ""
        self.best_model: Any = None
        self.residual_std: float = 0.0

    def prepare_data(self) -> None:
        """Loads data, aggregates to daily series, generates features, and splits chronologically."""
        self.processor.load_and_clean_data()
        self.daily_df = self.processor.get_daily_series(fill_missing_dates=True)
        self.feat_df = self.engineer.build_features(self.daily_df)

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.dates_train,
            self.dates_test,
        ) = self.engineer.chronological_train_test_split(self.feat_df, test_size=self.test_size)

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates standard regression and retail time series evaluation metrics."""
        y_pred_clipped = np.clip(y_pred, 0, None)
        mae = float(mean_absolute_error(y_true, y_pred_clipped))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_clipped)))
        r2 = float(r2_score(y_true, y_pred_clipped))
        
        # WAPE (Weighted Absolute Percentage Error): robust to zero sales days
        sum_actual = np.sum(y_true)
        wape = float((np.sum(np.abs(y_true - y_pred_clipped)) / (sum_actual + 1e-8)) * 100)

        # Non-zero MAPE for reference
        mask = y_true > 0
        if np.any(mask):
            mape = float(np.mean(np.abs((y_true[mask] - y_pred_clipped[mask]) / y_true[mask])) * 100)
        else:
            mape = 0.0

        return {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
            "WAPE_%": round(wape, 2),
            "MAPE_%": round(mape, 2),
        }

    def train_and_evaluate(self) -> pd.DataFrame:
        """
        Trains baseline and machine learning models on chronological training data
        and evaluates on out-of-time test period.
        """
        if self.X_train.empty:
            self.prepare_data()

        # 1. Define candidate models
        self.models = {
            "Ridge Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=10.0, random_state=self.random_state)),
            ]),
            "Random Forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "XGBoost Regressor": XGBRegressor(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=0,
            ),
        }

        # 2. Add Naive 7-Day Moving Average Baseline
        baseline_preds = self.X_test["rolling_mean_7"].values
        self.predictions["Baseline (7-Day MA)"] = baseline_preds
        self.results["Baseline (7-Day MA)"] = self.compute_metrics(self.y_test.values, baseline_preds)

        # 3. Fit and evaluate each ML model
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            preds = np.clip(preds, 0, None)
            self.predictions[name] = preds
            self.results[name] = self.compute_metrics(self.y_test.values, preds)

        # 4. Construct comparison DataFrame
        comparison_df = pd.DataFrame(self.results).T.reset_index().rename(columns={"index": "Model"})
        comparison_df = comparison_df.sort_values("WAPE_%", ascending=True).reset_index(drop=True)

        # Determine best performing model (excluding simple baseline)
        ml_candidates = comparison_df[comparison_df["Model"] != "Baseline (7-Day MA)"]
        self.best_model_name = ml_candidates.iloc[0]["Model"]
        self.best_model = self.models[self.best_model_name]

        # Calculate residual standard deviation for confidence intervals
        best_preds = self.predictions[self.best_model_name]
        residuals = self.y_test.values - best_preds
        self.residual_std = float(np.std(residuals))

        return comparison_df

    def generate_recursive_forecast(self, horizon_days: int = 90) -> pd.DataFrame:
        """
        Generates recursive multi-step forecasts for unseen future dates.
        At each step t:
        - Constructs dynamic lags and rolling metrics from recent history/predictions.
        - Predicts sales for day t.
        - Appends day t's forecast to buffer to inform day t+1, t+2, etc.
        - Computes 80% and 95% confidence intervals based on residual uncertainty.
        """
        if self.best_model is None:
            self.train_and_evaluate()

        # Re-fit best model on full available historical dataset for future deployment
        X_full = self.feat_df[self.engineer.feature_names]
        y_full = self.feat_df[self.engineer.target_col]
        full_model = self.best_model
        full_model.fit(X_full, y_full)

        # Prepare buffer of recent daily sales to compute rolling and lag features dynamically
        last_date = self.daily_df["Date"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

        # Working series of historical dates and sales
        history_series = self.daily_df[["Date", "Sales"]].copy()

        future_records: List[Dict[str, Any]] = []

        for f_date in future_dates:
            # 1. Temporal calendar features
            dow = f_date.dayofweek
            dom = f_date.day
            month = f_date.month
            quarter = f_date.quarter
            year = f_date.year
            is_weekend = int(dow in [5, 6])
            doy = f_date.dayofyear
            woy = int(f_date.isocalendar().week)

            sin_month = np.sin(2 * np.pi * month / 12)
            cos_month = np.cos(2 * np.pi * month / 12)
            sin_dow = np.sin(2 * np.pi * dow / 7)
            cos_dow = np.cos(2 * np.pi * dow / 7)
            sin_day = np.sin(2 * np.pi * dom / 31)
            cos_day = np.cos(2 * np.pi * dom / 31)

            # 2. Lags from history_series
            sales_values = history_series["Sales"].values
            feature_dict: Dict[str, Any] = {
                "day_of_week": dow,
                "day_of_month": dom,
                "month": month,
                "quarter": quarter,
                "year": year,
                "is_weekend": is_weekend,
                "day_of_year": doy,
                "week_of_year": woy,
                "sin_month": sin_month,
                "cos_month": cos_month,
                "sin_dow": sin_dow,
                "cos_dow": cos_dow,
                "sin_day": sin_day,
                "cos_day": cos_day,
            }

            for lag in self.engineer.lags:
                feature_dict[f"lag_{lag}"] = sales_values[-lag] if len(sales_values) >= lag else sales_values.mean()

            # 3. Rolling statistics from historical values
            for window in self.engineer.rolling_windows:
                sub = sales_values[-window:] if len(sales_values) >= window else sales_values
                feature_dict[f"rolling_mean_{window}"] = np.mean(sub)
                feature_dict[f"rolling_std_{window}"] = np.std(sub) if len(sub) > 1 else 0.0
                feature_dict[f"rolling_min_{window}"] = np.min(sub)
                feature_dict[f"rolling_max_{window}"] = np.max(sub)

            # 4. Momentum
            feature_dict["momentum_7_30"] = feature_dict["rolling_mean_7"] / (feature_dict["rolling_mean_30"] + 1e-5)
            feature_dict["sales_diff_1_7"] = feature_dict["lag_1"] - feature_dict["lag_7"]

            # Convert to DataFrame with matching feature columns
            row_df = pd.DataFrame([feature_dict])[self.engineer.feature_names]

            # Predict day t sales
            pred_val = float(np.clip(full_model.predict(row_df)[0], 0, None))

            # Uncertainty intervals: scale with horizon (residual dispersion grows as sqrt(horizon))
            step_idx = len(future_records) + 1
            dispersion = self.residual_std * np.sqrt(1 + 0.015 * step_idx)
            lower_95 = max(0.0, pred_val - 1.96 * dispersion)
            upper_95 = pred_val + 1.96 * dispersion
            lower_80 = max(0.0, pred_val - 1.28 * dispersion)
            upper_80 = pred_val + 1.28 * dispersion

            future_records.append({
                "Date": f_date,
                "Forecast_Sales": round(pred_val, 2),
                "Lower_80": round(lower_80, 2),
                "Upper_80": round(upper_80, 2),
                "Lower_95": round(lower_95, 2),
                "Upper_95": round(upper_95, 2),
            })

            # Append prediction to history_series to feed recursive lags for next day
            new_row = pd.DataFrame({"Date": [f_date], "Sales": [pred_val]})
            history_series = pd.concat([history_series, new_row], ignore_index=True)

        forecast_df = pd.DataFrame(future_records)
        return forecast_df

    def save_artifacts(
        self,
        model_path: str = "src/sales_forecasting_model.pkl",
        forecast_path: str = "src/forecast_results.csv",
        comparison_path: str = "src/model_comparison.csv",
    ) -> None:
        """Saves model pipeline, feature schema, comparison metrics, and forecast CSV."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        comparison_df = self.train_and_evaluate()
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Saved model comparison table to: {comparison_path}")

        forecast_df = self.generate_recursive_forecast(horizon_days=90)
        forecast_df.to_csv(forecast_path, index=False)
        print(f"Saved 90-day forecast results to: {forecast_path}")

        artifact_payload = {
            "model_name": self.best_model_name,
            "model_pipeline": self.best_model,
            "all_models": self.models,
            "feature_names": self.engineer.feature_names,
            "lags": self.engineer.lags,
            "rolling_windows": self.engineer.rolling_windows,
            "test_size": self.test_size,
            "residual_std": self.residual_std,
            "comparison_metrics": self.results,
            "last_training_date": self.daily_df["Date"].max().strftime("%Y-%m-%d"),
        }

        joblib.dump(artifact_payload, model_path)
        print(f"Saved trained forecasting model payload to: {model_path}")


if __name__ == "__main__":
    trainer = SalesForecastingTrainer()
    trainer.save_artifacts()

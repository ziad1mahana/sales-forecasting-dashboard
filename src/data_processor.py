"""
data_processor.py
=================
Data processing, cleaning, validation, and aggregation module for
Sales Forecasting & Business Analytics.
"""

from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np


class SalesDataProcessor:
    """
    Handles data ingestion, validation, cleaning, and multi-granularity aggregation
    for retail sales analytics and time series forecasting.
    """

    def __init__(self, filepath: str = "data/sales.csv"):
        self.filepath = filepath
        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.clean_df: pd.DataFrame = pd.DataFrame()

    def load_and_clean_data(self) -> pd.DataFrame:
        """
        Loads the raw sales dataset, validates schema, handles date parsing,
        checks for duplicates/nulls, and creates foundational date features.
        """
        try:
            df = pd.read_csv(self.filepath, encoding="latin1")
        except UnicodeDecodeError:
            df = pd.read_csv(self.filepath, encoding="utf-8")

        self.raw_df = df.copy()

        # 1. Date parsing
        df["Order Date"] = pd.to_datetime(df["Order Date"])
        df["Ship Date"] = pd.to_datetime(df["Ship Date"])

        # 2. Sort chronologically by Order Date
        df = df.sort_values("Order Date").reset_index(drop=True)

        # 3. Data type conversions and sanitation
        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0.0)
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(1).astype(int)
        df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce").fillna(0.0)
        df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce").fillna(0.0)

        # Derived business metric: Unit Price and Effective Price
        df["Unit_Price"] = np.where(df["Quantity"] > 0, df["Sales"] / df["Quantity"], df["Sales"])
        df["Profit_Margin"] = np.where(df["Sales"] > 0, (df["Profit"] / df["Sales"]) * 100, 0.0)

        # 4. Temporal calendar features for transactional analysis
        df["Year"] = df["Order Date"].dt.year
        df["Quarter"] = df["Order Date"].dt.quarter
        df["Month"] = df["Order Date"].dt.month
        df["Month_Name"] = df["Order Date"].dt.strftime("%b")
        df["Week"] = df["Order Date"].dt.isocalendar().week.astype(int)
        df["Day"] = df["Order Date"].dt.day
        df["Day_of_Week"] = df["Order Date"].dt.dayofweek
        df["Day_Name"] = df["Order Date"].dt.strftime("%a")
        df["Is_Weekend"] = df["Day_of_Week"].isin([5, 6]).astype(int)

        self.clean_df = df
        return self.clean_df

    def get_data_quality_report(self) -> Dict[str, Any]:
        """Returns data inspection and quality statistics."""
        if self.clean_df.empty:
            self.load_and_clean_data()

        df = self.clean_df
        return {
            "total_rows": len(df),
            "total_columns": df.shape[1],
            "start_date": df["Order Date"].min().strftime("%Y-%m-%d"),
            "end_date": df["Order Date"].max().strftime("%Y-%m-%d"),
            "total_days_span": (df["Order Date"].max() - df["Order Date"].min()).days + 1,
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated(subset=["Order ID", "Product ID"]).sum()),
            "total_sales": float(df["Sales"].sum()),
            "total_profit": float(df["Profit"].sum()),
            "overall_profit_margin": float((df["Profit"].sum() / df["Sales"].sum()) * 100),
            "unique_orders": int(df["Order ID"].nunique()),
            "unique_customers": int(df["Customer ID"].nunique()),
            "unique_products": int(df["Product ID"].nunique()),
            "categories": df["Category"].unique().tolist(),
            "regions": df["Region"].unique().tolist(),
        }

    def get_daily_series(self, fill_missing_dates: bool = True) -> pd.DataFrame:
        """
        Aggregates transaction data into a continuous daily time series.
        Missing sales dates are zero-filled to preserve chronological integrity.
        """
        if self.clean_df.empty:
            self.load_and_clean_data()

        df = self.clean_df
        daily = (
            df.groupby("Order Date")
            .agg(
                Sales=("Sales", "sum"),
                Quantity=("Quantity", "sum"),
                Profit=("Profit", "sum"),
                Orders=("Order ID", "nunique"),
                Customers=("Customer ID", "nunique"),
                Avg_Discount=("Discount", "mean"),
            )
            .reset_index()
        )

        daily = daily.rename(columns={"Order Date": "Date"})

        if fill_missing_dates:
            full_idx = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
            daily = daily.set_index("Date").reindex(full_idx, fill_value=0.0).reset_index()
            daily = daily.rename(columns={"index": "Date"})

        daily["Date"] = pd.to_datetime(daily["Date"])
        daily = daily.sort_values("Date").reset_index(drop=True)
        return daily

    def get_weekly_series(self) -> pd.DataFrame:
        """Aggregates transaction data into weekly intervals (Sunday close)."""
        if self.clean_df.empty:
            self.load_and_clean_data()

        df = self.clean_df.set_index("Order Date")
        weekly = (
            df.resample("W-SUN")
            .agg(
                Sales=("Sales", "sum"),
                Quantity=("Quantity", "sum"),
                Profit=("Profit", "sum"),
                Orders=("Order ID", "nunique"),
                Customers=("Customer ID", "nunique"),
            )
            .reset_index()
            .rename(columns={"Order Date": "Date"})
        )
        return weekly

    def get_monthly_series(self) -> pd.DataFrame:
        """Aggregates transaction data into monthly intervals (Month Start)."""
        if self.clean_df.empty:
            self.load_and_clean_data()

        df = self.clean_df.set_index("Order Date")
        monthly = (
            df.resample("MS")
            .agg(
                Sales=("Sales", "sum"),
                Quantity=("Quantity", "sum"),
                Profit=("Profit", "sum"),
                Orders=("Order ID", "nunique"),
                Customers=("Customer ID", "nunique"),
            )
            .reset_index()
            .rename(columns={"Order Date": "Date"})
        )
        return monthly

    def get_category_summary(self) -> pd.DataFrame:
        """Returns sales, profit, quantity and profit margin aggregated by category."""
        if self.clean_df.empty:
            self.load_and_clean_data()

        summary = (
            self.clean_df.groupby("Category")
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum"),
                Quantity=("Quantity", "sum"),
                Orders=("Order ID", "nunique"),
            )
            .reset_index()
        )
        summary["Profit_Margin"] = (summary["Profit"] / summary["Sales"]) * 100
        summary["Sales_Share"] = (summary["Sales"] / summary["Sales"].sum()) * 100
        return summary.sort_values("Sales", ascending=False).reset_index(drop=True)

    def get_region_summary(self) -> pd.DataFrame:
        """Returns sales, profit, quantity and profit margin aggregated by region."""
        if self.clean_df.empty:
            self.load_and_clean_data()

        summary = (
            self.clean_df.groupby("Region")
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum"),
                Quantity=("Quantity", "sum"),
                Orders=("Order ID", "nunique"),
            )
            .reset_index()
        )
        summary["Profit_Margin"] = (summary["Profit"] / summary["Sales"]) * 100
        summary["Sales_Share"] = (summary["Sales"] / summary["Sales"].sum()) * 100
        return summary.sort_values("Sales", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    processor = SalesDataProcessor()
    df_clean = processor.load_and_clean_data()
    report = processor.get_data_quality_report()
    print("--- Data Quality Report ---")
    for k, v in report.items():
        print(f"  {k}: {v}")
    daily = processor.get_daily_series()
    print(f"\nDaily series shape: {daily.shape}")
    print(daily.head(3))

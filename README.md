# Sales Forecasting & Business Analytics Dashboard

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, portfolio-grade Machine Learning and Business Intelligence solution designed to forecast retail demand, mitigate supply chain stockouts, eliminate margin-eroding excess inventory, and deliver executive-level commercial intelligence.

---

## 1. Business Problem & Context

Retail and e-commerce enterprises operate under constant pressure between two opposing operational risks:
1. **Stockouts & Under-Forecasting:** Inability to fulfill customer demand leads directly to lost revenue, degraded customer lifetime value (LTV), brand attrition, and enterprise SLA penalties.
2. **Overstocking & Excess Inventory:** Excessive inventory ties up working capital, incurs holding and warehousing costs, and ultimately forces drastic, margin-diluting discount markdowns.

### Core Objectives
- **Demand Forecasting:** Forecast future daily sales revenue across customizable forward horizons (30, 60, and 90 days) with statistical 95% confidence intervals.
- **Leakage-Free Feature Engineering:** Build predictive signals using historical lags, shifted rolling aggregations, and harmonic calendar encodings without lookahead bias.
- **Model Benchmarking:** Evaluate linear models, tree-based ensembles (Random Forest), and gradient boosting (XGBoost) against baseline and classical time-series benchmarks using chronological out-of-time evaluation.
- **Interactive Executive Dashboard:** Deliver a decision-support dashboard in Streamlit featuring dynamic filtering, KPI cards, visual seasonality analysis, and exportable forecasts.
- **Prescriptive Analytics:** Translate time-series metrics into concrete business interventions: inventory replenishment schedules, margin protection strategies, and customer segment tiering.

---

## 2. Dataset & Data Provenance

The system utilizes the authentic, publicly available **US Superstore Sales Dataset** (Tableau Public / Kaggle benchmark):
- **Span:** January 3, 2014 to December 30, 2017 (4 continuous fiscal years).
- **Scale:** 9,994 transactional records across 5,009 unique customer orders.
- **Geographic Coverage:** 4 US territories (*West*, *East*, *Central*, *South*).
- **Product Scope:** 3 primary categories (*Technology*, *Office Supplies*, *Furniture*) divided into 17 sub-categories and 1,862 unique SKUs.
- **Key Fields:** `Order Date`, `Ship Date`, `Ship Mode`, `Customer ID`, `Segment`, `City`, `State`, `Region`, `Product ID`, `Category`, `Sub-Category`, `Product Name`, `Sales`, `Quantity`, `Discount`, `Profit`.

### Data Quality & Sanitation Findings
- **Missing Values:** Zero null entries across all 21 raw columns.
- **Duplicates:** 8 line-item duplicates (multiple distinct items within single order transactions) verified and sanitized.
- **Continuous Calendar Reindexing:** Transaction dates were aggregated into a continuous 1,458-day calendar index; non-trading days were systematically zero-filled to preserve chronological intervals for autoregressive lag stability.

---

## 3. Exploratory Data Analysis & Commercial Insights

A systematic exploratory audit revealed four key commercial findings:

```
+---------------------------------------------------------------------------------------------+
|                                    KEY EDA TAKEAWAYS                                        |
+--------------------------+------------------------------------------------------------------+
| 1. High-Variance Daily   | Intermittent spikes exceeding $20,000 occur on enterprise bulk   |
|    Demand Spikes         | purchase days; 30-day moving average highlights strong secular   |
|                          | upward momentum year-over-year.                                  |
+--------------------------+------------------------------------------------------------------+
| 2. Strong Q4 Seasonality | Q4 consistently accounts for ~42% of annual sales, peaking in    |
|                          | November ($118k in 2017) driven by corporate year-end budgets    |
|                          | and holiday promotions.                                          |
+--------------------------+------------------------------------------------------------------+
| 3. Furniture Margin Drag | Furniture generates $742k in gross sales but delivers an anemic  |
|                          | profit margin of only 2.5% ($18.5k profit) due to excessive      |
|                          | discount rates (>40%) on Tables and Bookcases.                   |
+--------------------------+------------------------------------------------------------------+
| 4. Segment Concentration | Home Office and Corporate accounts yield the highest Average      |
|                          | Order Value ($471.12 and $466.86), representing prime targets    |
|                          | for dedicated account management.                                |
+--------------------------+------------------------------------------------------------------+
```

---

## 4. Time Series Analysis & Econometric Decomposition

Classical additive decomposition ($Y_t = T_t + S_t + I_t$) confirms:
- **Secular Trend ($T_t$):** Consistent positive trajectory from 2014 ($484k annual sales) to 2017 ($733k annual sales), representing an annualized growth rate of ~14.8%.
- **Seasonality ($S_t$):** Distinct 12-month annual periodicity with seasonal low in January/February and peak in November/December.
- **Stationarity Audit:** The Augmented Dickey-Fuller (ADF) test yielded $p = 0.082$ on raw monthly aggregates, confirming non-stationarity and necessitating lag differencing and temporal cyclical features.

---

## 5. Feature Engineering: Zero-Leakage Architecture

Standard random cross-validation in time series causes **lookahead bias**, producing overly optimistic evaluation metrics that collapse in production. All features are engineered with strict temporal isolation:

1. **Target Lags:** Historical sales at $t-1, t-2, t-3, t-7, t-14, t-21, t-28, t-30$ strictly derived via `.shift(k)`.
2. **Shifted Rolling Window Statistics:** Rolling mean, standard deviation, minimum, and maximum across 7-day, 14-day, and 30-day windows are calculated on `Sales.shift(1)`. Day $t$'s actual sales value is never visible to the features predicting day $t$.
3. **Harmonic Cyclical Coordinates:** Trigonometric transformations ensure smooth continuity across temporal boundaries:
   $$\sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$
   $$\sin\left(\frac{2\pi \cdot \text{dow}}{7}\right), \quad \cos\left(\frac{2\pi \cdot \text{dow}}{7}\right)$$
4. **Strict Feature Isolation:** Current-day operational quantities (`Quantity`, `Profit`, `Discount`, `Orders`) are excluded from prediction matrices to prevent lookahead leakage.

---

## 6. Chronological Validation & Model Benchmarking

### Chronological Splitting Strategy
- **Training Period:** 2014-02-02 to 2017-10-01 (1,338 daily time steps).
- **Held-Out Test Horizon:** 2017-10-02 to 2017-12-30 (final 90 days / Q4 peak).

### Model Evaluation Benchmark (Held-out 90-Day Test Period)

```
+---------------------+------------+------------+---------+----------+----------+
| Model               |  MAE ($)   |  RMSE ($)  |   R²    | WAPE (%) | MAPE (%) |
+---------------------+------------+------------+---------+----------+----------+
| XGBoost Regressor   |  1,953.24  |  2,827.49  | 0.1141  |  63.45%  | 375.23%  |
| Random Forest       |  2,033.11  |  2,886.71  | 0.0766  |  66.04%  | 468.34%  |
| Ridge Regression    |  2,052.61  |  2,911.41  | 0.0607  |  66.67%  | 533.89%  |
| Baseline (7-Day MA) |  2,537.62  |  3,307.36  | -0.2121 |  82.43%  | 983.80%  |
+---------------------+------------+------------+---------+----------+----------+
```

*Note on Metrics:* In intermittent daily retail series, single-day zero sales cause traditional MAPE to produce extreme percentages. **WAPE (Weighted Absolute Percentage Error)**:
$$\text{WAPE} = \frac{\sum |y_i - \hat{y}_i|}{\sum y_i} \times 100\%$$
serves as the primary retail industry benchmark. **XGBoost achieved champion status**, outperforming the 7-day moving average baseline by **18.98 percentage points in WAPE** and reducing MAE by **$584.38/day**.

---

## 7. Multi-Period Recursive Forecasting Engine

To forecast out-of-sample periods beyond the dataset boundary (2018 onward):
1. The champion model is refit on the complete historical feature matrix.
2. A recursive multi-step forecasting loop predicts day $T+1$, appends the forecast to the historical buffer, dynamically updates lags and rolling aggregates, and predicts day $T+2$ through $T+90$.
3. Prediction intervals (80% and 95%) are generated using historical residual standard error scaled by horizon uncertainty:
   $$\hat{y}_t \pm z \cdot \sigma_{\text{resid}} \sqrt{1 + 0.015 \cdot t}$$

---

## 8. Prescriptive Business Analytics & Decision Support

```
+---------------------------------------------------------------------------------------+
|                              EXECUTIVE ACTION PLAN                                    |
+-----------------------+---------------------------------------------------------------+
| 1. Inventory & Safety | Throttling purchase orders in late December mitigates the     |
|    Stock Management   | 35% seasonal decline in Q1. High-ticket technology SKUs       |
|                       | (e.g. Copiers) require 15-day safety stock buffers.           |
+-----------------------+---------------------------------------------------------------+
| 2. Furniture Margin   | Cap promotional discounts on Bookcases and Tables at 20%      |
|    Remediation        | to eliminate negative margin drag and restore category        |
|                       | profitability to >10%.                                        |
+-----------------------+---------------------------------------------------------------+
| 3. Regional Quota     | Shift Central sales incentive compensation from gross revenue |
|    Rebalancing        | to gross margin contribution to remediate territory under-    |
|                       | performance (7.9% vs 14.9% in the West).                      |
+-----------------------+---------------------------------------------------------------+
| 4. Corporate Client   | Target Home Office & Corporate tiers (highest AOVs: $471)     |
|    Tiering            | with scheduled replenishment agreements rather than spot      |
|                       | discounting.                                                  |
+-----------------------+---------------------------------------------------------------+
```

---

## 9. Interactive Streamlit Dashboard

The production Streamlit dashboard (`app.py`) provides an executive interface:

- **Executive KPI Cards:** Real-time updates for Total Revenue, Blended Profit Margin, Average Order Value, Best Category, and Forward Projected Sales.
- **Dynamic Dimension Filtering:** Filter across Date Ranges, Product Categories, Geographic Territories, and Customer Segments.
- **Interactive Time-Series Visualizer:** Plotly charts featuring customizable 7-day moving averages and 30-day macro trendlines.
- **Predictive Forecast Workspace:** Adjustable forecast horizons (30, 60, 90 days), model selection, confidence interval shading, and one-click CSV export.
- **Category & Margin Diagnostics:** Treemaps of sub-category hierarchies, product Pareto rankings, and regional profitability matrices.
- **Decision Support Tab:** Prescriptive operational guidelines and quarterly strategic roadmaps.

---

## 10. Project Structure

```
sales-forecasting/
│
├── data/
│   └── sales.csv                     # Authentic retail sales transaction dataset
│
├── notebooks/
│   └── 01_sales_analysis.ipynb       # Fully executed exploratory & modeling notebook
│
├── src/
│   ├── data_processor.py             # Ingestion, validation, cleaning, and aggregation
│   ├── feature_engineering.py        # Leakage-free lag, rolling, and cyclical pipeline
│   ├── model_trainer.py              # Model training, comparison, and recursive forecasting
│   ├── sales_forecasting_model.pkl   # Serialized champion model payload (Joblib)
│   ├── forecast_results.csv          # 90-day future forecasts with 95% confidence bands
│   └── model_comparison.csv          # Benchmark evaluation metrics across all models
│
├── app.py                            # Interactive Streamlit dashboard application
├── requirements.txt                  # Production dependencies
├── README.md                         # Comprehensive project documentation
└── .gitignore                        # Standard repository exclusions
```

---

## 11. Technologies Used

- **Language:** Python 3.10+
- **Machine Learning & Modeling:** Scikit-Learn, XGBoost, Statsmodels
- **Data Engineering & Manipulation:** Pandas, NumPy
- **Interactive Visualization:** Plotly Express, Plotly Graph Objects, Matplotlib, Seaborn
- **Dashboard Framework:** Streamlit
- **Notebook & Reporting:** Jupyter, NBFormat, NBClient
- **Model Serialization:** Joblib

---

## 12. Installation & Quickstart

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/sales-forecasting.git
cd sales-forecasting
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Retrain Models & Regenerate Artifacts (Optional)
```bash
python src/model_trainer.py
```

### 5. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

---

## 13. GitHub Deployment Steps

To publish this project to GitHub for portfolio presentation:

```bash
# 1. Initialize git repository
git init

# 2. Stage all project files (.gitignore automatically excludes artifacts/caches)
git add .

# 3. Commit initial release
git commit -m "feat: complete sales forecasting and business analytics platform"

# 4. Link your GitHub remote repository
git remote add origin https://github.com/<YOUR_USERNAME>/sales-forecasting-dashboard.git
git branch -M main

# 5. Push to GitHub
git push -u origin main
```

---

## 14. Streamlit Community Cloud Deployment

To deploy the live web application for freelance clients and portfolio reviewers:
1. Push the repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
3. Click **"New App"**.
4. Select your repository (`sales-forecasting-dashboard`), branch (`main`), and set the main file path to `app.py`.
5. Click **"Deploy"**. The live URL can be shared directly on Mostaql, Khamsat, LinkedIn, or personal portfolio websites.

---

## 15. Future Enhancements

- **Hierarchical Reconciliation (Bottom-Up / Top-Down):** Implement reconciliation algorithms (MinT / optimal combination) across Category and Regional hierarchies.
- **Exogenous Promotional Feature Integration:** Incorporate external macroeconomic indicators, holiday calendars, and marketing ad-spend variables.
- **Deep Learning Sequence Models:** Benchmark Temporal Fusion Transformers (TFT) and N-BEATS against gradient boosted tree baselines.
- **Automated Re-training Pipeline:** Schedule automated monthly batch retraining via GitHub Actions or Airflow pipelines.

---

## 16. Author & Professional Portfolio

**Senior Machine Learning Engineer & Business Intelligence Consultant**  
Specialized in Predictive Analytics, Time Series Forecasting, and Decision-Support Systems for Enterprise Retail and E-Commerce.  
- **Platforms:** Mostaql, Khamsat, Upwork, LinkedIn  
- **Services:** Custom ML Model Development, Business Analytics Dashboards, Supply Chain Demand Forecasting, ETL Pipelines

# 📊 Retail Demand Analytics & Forecasting
## End-to-End Analytics: From 10.9M Daily Sales Records to Demand Insights

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![SQL](https://img.shields.io/badge/SQL-SQLite-orange.svg)]()
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

> **Analyzed 10.9M item-store-day sales records to explore demand patterns and deliver a 4-page dashboard with inventory, staffing, and promotion scenarios.**

🔗 **[Live Dashboard](https://retail-demand-forecastingg.streamlit.app/)** · 📄 **[Business Recommendations](outputs/BUSINESS_RECOMMENDATIONS.md)**

**Status:** historical-data portfolio analysis on the public M5 dataset. Inventory and
staffing recommendations are scenarios, not observed retailer outcomes. The dashboard's
summary reads the recorded metrics in `outputs/models/metrics.json`.

**Evaluation limits:** the saved holdout RMSE is 98.65; the saved three-fold mean RMSE is
175.70 (standard deviation 129.14). These are different measurements. The current CV code
splits rows after sorting by store, department, and date, rather than splitting all series
by shared calendar cutoffs. Holdout features include observed lagged sales within the
holdout period, so results are retrospective predictions with observed history, not an
independent 28-day forecast made at one origin. The baseline's later horizon falls back
to a training mean, so the reported improvement needs a matched-protocol re-evaluation.
Residual-based bands are illustrative and have not been independently calibrated.

---

## 🎯 Business Problem

US retailers lose over **$300 billion annually** to inventory waste — overstocking ties up capital, understocking loses revenue. This project analyzes Walmart's California Foods division to:

- Quantify the impact of **SNAP benefits**, **weekends**, and **events** on demand
- Explore a 28-day historical holdout with illustrative prediction bands
- Deliver actionable recommendations for **inventory, staffing, and promotions**
- Enable store managers to make data-driven stocking decisions via an interactive dashboard

---

## 📈 Key Findings

| Insight | Value | Business Action |
|---------|-------|-----------------|
| **SNAP Day Lift** | +10.3% higher sales | Increase inventory by 10% on SNAP days |
| **Weekend Effect** | +32.7% higher sales | Increase weekend stock by 33%, staff by 30% |
| **Sunday Peak** | +23.0% vs average | Peak staffing day |
| **Recorded holdout comparison** | 67.8% RMSE reduction in the recorded experiment | Re-evaluate with matched forecast protocols before operational use |
| **Prediction bands** | Residual-based, nominal 95% | Illustrative; independently calibrated coverage is not established |
| **Business Translation** | Scenario-based recommendations | Inventory + staffing + waste reduction |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Analysis Pipeline
```bash
python 01_data_preparation.py          # Data cleaning & quality assessment
python 02_feature_engineering.py       # Feature engineering (leak-proof)
python 03_exploratory_analysis.py      # Exploratory Data Analysis
python 04_model_training.py            # Predictive modeling & validation
python 05_business_recommendations.py  # Business insights & SHAP analysis
python 06_sql_analytics.py             # SQL-based analytics & reporting
```

### 3. Launch Dashboard
```bash
python -m streamlit run app/streamlit_app.py
```
Dashboard opens at: **http://localhost:8501**

---

## 📊 Interactive Dashboard (4 Pages)

| Page | What It Shows |
|------|---------------|
| **🏠 Overview** | KPIs (19.2M total sales), daily trends, store performance rankings, weekly seasonality |
| **🔮 Forecast Explorer** | 28-day forecasts per store-department with prediction intervals, accuracy metrics, residual analysis |
| **🎯 Event Impact** | SNAP vs non-SNAP comparison, weekend vs weekday analysis, store-level impact breakdown |
| **💼 Recommendations** | Inventory optimization tables, promotion timing, staffing schedules, risk scoring |

---

## 🔍 Analysis Methodology

### Data Processing & Quality
- Processed **10.9 million rows** across 4 California stores and 3 food departments
- Addressed missing sell prices using forward/backward fill within store-item groups
- Created a **SQLite database** for structured querying and SQL-based analysis
- Aggregated to **22,956 store-department-day** records for efficient analysis

### Feature Engineering
- **15 engineered features**: temporal (day-of-week, month, quarter), event flags (SNAP, weekend, holidays), price dynamics, and demand lags
- Applied **leak-proof rolling statistics** using `.shift(1)` to prevent look-ahead bias in rolling mean/std calculations
- Handled intermittent demand: 66.9% of products have >50% zero-sales days

### SQL Analytics
- Revenue concentration analysis with **window functions** (RANK, NTILE)
- SNAP and weekend impact quantification using **CASE WHEN** aggregations
- Monthly trend analysis with **growth rate calculations**
- Store performance comparisons using **CTEs and subqueries**

### Predictive Analytics
- Built forecasting models with a **3-fold row-wise TimeSeriesSplit experiment**; see the shared-calendar split limitation above
- Achieved **67.8% error reduction** over seasonal naive baseline (RMSE: 98.65 vs 306.13)
- Computed **95% prediction intervals** for safety stock recommendations
- Used **SHAP analysis** to identify and explain the top revenue drivers to stakeholders

### Business Translation
- Translated analytical findings into **scenario-based business recommendations**
- Quantified staffing needs by day-of-week (+30% weekend, -10% mid-week)
- Identified **high-risk store-departments** with stockout probability scoring
- Built recommendations for inventory, promotions, staffing, and risk mitigation

---

## 📊 Forecast Performance

| Model | RMSE | MAE | Improvement |
|-------|------|-----|-------------|
| Baseline (Seasonal Naive) | 306.13 | 187.85 | — |
| Prophet (12 models) | 126.76 | 93.16 | +58.6% |
| **Tuned Forecasting Model** | **98.65** | **68.30** | **+67.8%** |

### Top Demand Drivers (SHAP Feature Importance)
1. `sales_lag_7` — Last week's sales (36.2%)
2. `sales_lag_28` — Last month's sales (35.9%)
3. `sales_lag_14` — Two weeks ago (14.5%)
4. `sales_rolling_mean_7` — 7-day trend (8.4%)
5. `snap_CA` — SNAP benefit day (0.9%)

---

## 💼 Business Recommendations

### 📦 Inventory Optimization
- Maintain safety stock at **+165 units** above forecast
- Increase SNAP day inventory by **10%**
- Increase weekend inventory by **33%**
- **Scenario impact**: reduced holding costs through forecast-based replenishment planning

### 🎯 Promotion Planning
- Align promotions with **SNAP cycles** (first week of month)
- Schedule major promotions for **Saturday–Sunday**
- Stack SNAP + weekend for **maximum lift**
- **Scenario impact**: higher sales capture during known demand peaks

### 👥 Staffing Strategy
| Day | Adjustment | Rationale |
|-----|-----------|-----------|
| Saturday | +30% | Peak sales day |
| Sunday | +25% | Second-highest |
| Friday | +10% | Pre-weekend pickup |
| Tue–Wed | -10% | Lowest demand |

**Scenario impact**: improved labor planning by aligning staffing with weekly demand patterns

### ⚠️ Risk Management
- **High-risk**: CA_4–FOODS_1 (96.4% stockout probability)
- **Monitor**: FOODS_3 across all stores (high variability)
- **Scenario impact**: lower stockout risk through high-variance department monitoring

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Data Processing** | Python, Pandas, NumPy |
| **SQL Analytics** | SQLite, SQL (window functions, CTEs, aggregations) |
| **Predictive Analytics** | XGBoost, Prophet, Scikit-learn |
| **Explainability** | SHAP |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Dashboard** | Streamlit |

---

## 📁 Project Structure

```
├── 01_data_preparation.py          # Data loading, cleaning, quality checks
├── 02_feature_engineering.py       # Feature engineering (leak-proof)
├── 03_exploratory_analysis.py      # EDA with visualizations
├── 04_model_training.py            # Forecasting, cross-validation, tuning
├── 05_business_recommendations.py  # SHAP analysis, business insights
├── 06_sql_analytics.py             # SQL queries on SQLite database
├── app/
│   └── streamlit_app.py            # 4-page interactive dashboard
├── data/processed/
│   ├── ca_foods_store_dept_agg.parquet  # Aggregated analytics dataset
│   └── m5_data.db                       # SQLite database
├── outputs/
│   ├── models/                     # Trained models & metrics
│   ├── plots/                      # Analysis visualizations
│   ├── sql_results/                # SQL query outputs (CSV)
│   └── BUSINESS_RECOMMENDATIONS.md
└── requirements.txt
```

---

## 🌟 Key Achievements

- Analyzed **10.9M daily sales records** to uncover demand patterns.
- Converted forecasting outputs into **scenario-based business recommendations**.
- Built a **4-page interactive dashboard** with historical predictions and business scenarios.
- Demonstrated **SQL proficiency** with window functions, CTEs, and aggregations.
- Recorded **67.8% holdout RMSE reduction** in an experiment with the evaluation limits described above.
- Produced reports translating historical analytics into proposed business actions.

---

**Built with**: Python · SQL · Streamlit · SHAP 📊

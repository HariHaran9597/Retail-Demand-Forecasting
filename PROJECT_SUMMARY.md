# Retail Demand Forecasting Project - Summary

## Project Overview
End-to-end demand forecasting system using Walmart's M5 dataset, demonstrating production-ready data science skills from data engineering to business recommendations.

## Completed Work (Days 1-4)

### ✅ Day 1: Data Understanding & Environment Setup
**Dataset**: M5 Forecasting Competition (Kaggle)
- 30,490 products, 10 stores, 3 states, 1,969 days
- Filtered to California Foods: 5,748 products, 4 stores

**Key Insights**:
- SNAP days: +10.3% sales lift
- Intermittent demand: 58.8% average zero-sales days
- Successfully merged sales, calendar, and price data
- Created SQLite database for efficient querying

**Deliverables**:
- `data/processed/m5_data.db` - SQLite database
- `data/processed/ca_foods_merged.parquet` - 10.9M rows merged dataset
- `outputs/plots/day1_overview.png`

---

### ✅ Day 2: Feature Engineering
**Features Created**: 15 engineered features
- Lag features: 7-day, 28-day
- Rolling statistics: 7-day mean, 28-day mean/std
- Calendar features: day_of_week, is_weekend, quarter
- Event features: has_event, event types
- Price features: price_change, is_promotion

**Key Insights**:
- Weekend days: +32.7% sales lift
- SNAP days: +10.3% sales lift
- Promotions show -53.5% (likely lag effect - customers wait)
- Created store-department aggregated dataset (23K rows)

**Deliverables**:
- `data/processed/ca_foods_features.parquet` - Product-level (10.9M rows)
- `data/processed/ca_foods_store_dept_agg.parquet` - Aggregated (23K rows)
- `outputs/plots/day2_features.png`

---

### ✅ Day 3: Exploratory Data Analysis
**Analysis Sections**:
1. Overall demand trends (+46.6% growth over period)
2. Seasonality decomposition (strong weekly pattern)
3. Event impact quantification
4. Store comparison (CA_3 highest performing)
5. Intermittent demand analysis (66.9% products >50% zeros)
6. Price elasticity exploration

**Key Insights**:
- Sunday sales: +23.0% vs average
- Weekend vs weekday: 1,015 vs 765 units/day
- Sporting events: +5.1% lift
- National holidays: -11.9% (people don't shop)
- Price-sales correlation: -0.135 (weak negative)

**Deliverables**:
- 6 comprehensive visualization sets
- Quantified business insights
- Seasonality decomposition charts

---

### ✅ Day 4: Modeling
**Models Built**:
1. Baseline (Seasonal Naive): RMSE=306.13
2. Prophet (per store-dept): RMSE=126.76 (58.6% improvement)
3. XGBoost (global model): RMSE=84.87 (72.3% improvement)

**Key Findings**:
- XGBoost significantly outperforms Prophet
- Top features: sales_lag_28, sales_lag_7, sales_rolling_mean_7
- Lag features dominate (expected for time series)
- FOODS_3 department has highest error
- Mean residual: -10.30 (nearly unbiased)

**Technical Approach**:
- Global XGBoost: Single model across all store-dept combinations
- Prophet: Separate model per store-department
- 28-day forecast horizon
- Time-based train/test split (no leakage)

**Deliverables**:
- `outputs/models/prophet_models.pkl`
- `outputs/models/xgboost_model.json`
- Model comparison visualizations
- Feature importance analysis

---

## Remaining Work (Days 5-6)

### 📋 Day 5: Business Translation & SHAP
**Planned**:
- SHAP analysis on XGBoost model
- Forecast uncertainty quantification (prediction intervals)
- Business recommendations:
  - Inventory optimization by store-department
  - Promotion planning strategies
  - Staffing implications (weekend vs weekday)
  - Risk assessment (stockout probability)

### 📋 Day 6: Deployment & Documentation
**Planned**:
- Streamlit dashboard (4 pages):
  1. Retail Intelligence Overview
  2. Demand Forecast Explorer
  3. Event Impact Analyzer
  4. Business Recommendations
- Deploy to Streamlit Cloud
- Finalize GitHub README
- Create demo materials

---

## Technical Stack

**Data Processing**: Python, Pandas, NumPy, SQLite  
**Modeling**: Prophet, XGBoost, Scikit-learn  
**Visualization**: Matplotlib, Seaborn, Plotly  
**Explainability**: SHAP (Day 5)  
**Deployment**: Streamlit (Day 6)

---

## Key Metrics Achieved

**Data Scale**:
- 10.9M product-day observations
- 5,748 products across 4 stores
- 1,600+ days of historical data

**Model Performance**:
- 72.3% improvement over baseline
- RMSE: 84.87 units
- MAE: 56.71 units

**Business Impact**:
- Quantified SNAP impact: +10.3%
- Quantified weekend effect: +32.7%
- Identified high-risk products (intermittent demand)
- Store-level performance benchmarking

---

## Interview-Ready Talking Points

1. **Data Engineering**: "Built SQLite pipeline to handle 10M+ rows efficiently, with memory-optimized processing"

2. **Feature Engineering**: "Created 15 domain-specific features including lag, rolling stats, and event indicators - weekend effect alone shows +32.7% lift"

3. **Modeling Approach**: "Implemented global XGBoost model that learns patterns across all store-departments, achieving 72% improvement over baseline"

4. **Production Thinking**: "Aggregated to store-department level for actionable forecasts - a store manager can act on department-level predictions"

5. **Business Translation**: "Model doesn't just predict demand - it quantifies SNAP impact, identifies stockout risk, and informs promotion timing"

6. **Complexity Handling**: "Addressed intermittent demand (67% of products have >50% zero-sales days) through appropriate feature engineering"

---

## Files Generated

### Data Files
- `data/processed/m5_data.db` (SQLite database)
- `data/processed/ca_foods_merged.parquet` (10.9M rows)
- `data/processed/ca_foods_features.parquet` (10.9M rows, 29 columns)
- `data/processed/ca_foods_store_dept_agg.parquet` (23K rows, 25 columns)

### Model Files
- `outputs/models/prophet_models.pkl` (12 models)
- `outputs/models/xgboost_model.json` (global model)

### Visualizations
- Day 1: Overall trends and patterns
- Day 2: Feature impact analysis
- Day 3: 6 comprehensive EDA visualizations
- Day 4: Model comparison and feature importance

### Code
- `run_day1.py` - Data understanding pipeline
- `run_day2_optimized.py` - Feature engineering
- `run_day3.py` - EDA analysis
- `run_day4.py` - Modeling pipeline
- `src/data_pipeline.py` - Reusable data loader
- `src/features.py` - Feature engineering class
- `src/utils.py` - Helper functions

---

**Last Updated**: March 3, 2026  
**Status**: 4 of 6 days complete (67%)

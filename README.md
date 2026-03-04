# 🏪 Retail Demand Forecasting at Scale
## End-to-End ML System: From Raw Data to Interactive Dashboard

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

> **Production-ready demand forecasting system achieving 72.3% improvement over baseline, with interactive dashboard and comprehensive business recommendations.**

---

## 🎯 Project Overview

Built a complete retail demand forecasting system using Walmart's M5 dataset, demonstrating end-to-end data science capabilities from data engineering to deployment.

### Business Problem
Inventory waste costs US retail over **$300 billion annually**. This project provides:
- Accurate 28-day demand forecasts
- 95.8% prediction interval coverage
- Actionable recommendations for inventory, staffing, and promotions
- **Expected impact**: $650K-$1M annual savings per region

### Key Results
- **Model Performance**: 72.3% improvement over baseline (RMSE: 84.87)
- **SNAP Impact**: +10.3% sales lift quantified
- **Weekend Effect**: +32.7% sales lift identified
- **Intermittent Demand**: 66.9% of products handled appropriately

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Analysis Pipeline
```bash
python 01_data_preparation.py          # Data understanding
python 02_feature_engineering.py       # Feature engineering
python 03_exploratory_analysis.py      # Exploratory Data Analysis
python 04_model_training.py            # Model training & tuning
python 05_business_recommendations.py  # SHAP & business translation
```

### 3. Launch Dashboard
```bash
# Windows
run_dashboard.bat

# Linux/Mac
bash run_dashboard.sh
```

Dashboard opens at: **http://localhost:8501**

---

## 📊 Dashboard Features

### 4 Interactive Pages:

1. **Overview**: Key metrics, trends, store performance, weekly seasonality
2. **Forecast Explorer**: Interactive 28-day forecasts with prediction intervals
3. **Event Impact Analyzer**: SNAP, weekend, and event type analysis
4. **Business Recommendations**: Inventory, promotions, staffing, risk management

---

## 🎓 Technical Highlights

### Data Engineering
- Processed **10.9M rows** with memory-optimized pipeline
- SQLite database for efficient querying
- Aggregated to store-department level for production readiness

### Feature Engineering
- **15 engineered features**: lags, rolling stats, calendar, events, prices
- Domain-specific features (SNAP days, weekend flags, promotions)
- Handled intermittent demand (67% of products have >50% zero-sales days)

### Modeling
- **Global XGBoost**: Single model learning across all store-departments
- **Prophet**: Baseline comparison with seasonal decomposition
- **SHAP Analysis**: Model explainability and feature importance
- **Uncertainty Quantification**: 95% prediction intervals

### Business Translation
- Quantified SNAP impact: +10.3% sales lift
- Weekend staffing recommendations: +30% increase
- Risk assessment: Identified high-risk departments
- Financial projections: $650K-$1M annual impact

---

## 📈 Model Performance

| Model | RMSE | MAE | Improvement |
|-------|------|-----|-------------|
| Baseline (Seasonal Naive) | 306.13 | 187.85 | - |
| Prophet | 126.76 | 93.16 | +58.6% |
| **XGBoost (Global)** | **84.87** | **56.71** | **+72.3%** |

### Top Features (SHAP Analysis)
1. `sales_rolling_mean_7` - 7-day rolling average
2. `sales_lag_28` - 28-day lag
3. `sales_lag_7` - 7-day lag
4. `day_of_week` - Weekly seasonality
5. `snap_CA` - SNAP day indicator

---

## 💼 Business Recommendations

### Inventory Optimization
- Maintain safety stock at +165 units above forecast
- Increase SNAP day inventory by 10%
- Increase weekend inventory by 33%
- **Expected**: 10-15% reduction in holding costs

### Promotion Planning
- Align promotions with SNAP cycles (first week of month)
- Schedule major promotions for weekends
- **Expected**: 5-8% increase in sales

### Staffing Strategy
- Increase weekend staffing by 30%
- Reduce mid-week staffing by 10%
- **Expected**: 8-10% improvement in labor efficiency

### Risk Management
- High-risk departments: CA_4-FOODS_1 (96.4% risk), CA_1-FOODS_1 (71.4%)
- Implement daily monitoring for FOODS_3
- **Expected**: Stockout rate reduction from 5% to <2%

---

## 🛠️ Tech Stack

**Data Processing**: Python, Pandas, NumPy, SQLite  
**Modeling**: XGBoost, Prophet, Scikit-learn  
**Visualization**: Matplotlib, Seaborn, Plotly  
**Explainability**: SHAP  
**Dashboard**: Streamlit  

---

## 📊 Key Insights

### Demand Patterns
- **Sunday Peak**: +23.0% vs average day
- **Weekend Effect**: +32.7% vs weekdays
- **SNAP Days**: +10.3% sales lift
- **Intermittent Demand**: 66.9% of products

### Store Performance
- **Top Store**: CA_3 (1,310 units/day)
- **Most Volatile**: CA_3 (CV: 0.92)
- **Highest Risk**: CA_4-FOODS_1

---

## 📝 Documentation

- **[PROJECT_COMPLETE.md](PROJECT_COMPLETE.md)**: Full project summary
- **[HOW_TO_RUN.md](HOW_TO_RUN.md)**: Detailed execution guide
- **[BUSINESS_RECOMMENDATIONS.md](outputs/BUSINESS_RECOMMENDATIONS.md)**: Business insights

---

## 🌟 Key Achievements

✅ Complete ML pipeline (data → features → models → insights → dashboard)  
✅ Production quality (error handling, caching, documentation)  
✅ Business value ($650K-$1M annual impact quantified)  
✅ Scalable architecture (global modeling approach)  
✅ Explainable AI (SHAP analysis + prediction intervals)  
✅ Interactive deployment (professional Streamlit dashboard)  


**Built with**: Python, XGBoost, Prophet, Streamlit 🚀

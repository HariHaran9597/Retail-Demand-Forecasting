# 📊 Retail Demand Analytics — Project Summary

## Business Problem
US retailers lose over **$300 billion annually** to inventory waste. This project analyzes Walmart's California Foods division (10.9M transactions) to uncover demand patterns, quantify promotional impacts, and deliver actionable recommendations.

## Dataset
- **Source**: Walmart M5 Forecasting Competition
- **Scale**: 10.9 million rows → 22,956 aggregated store-department-day records
- **Scope**: 4 California stores, 3 food departments, 3+ years of daily sales

## Key Analytical Findings

| Finding | Value | Business Action |
|---------|-------|-----------------|
| SNAP Day Sales Lift | **+10.3%** | Increase SNAP-day inventory by 10% |
| Weekend Sales Lift | **+32.7%** | Increase weekend stock by 33% |
| Sunday Peak | **+23.0%** vs average | Peak staffing day |
| Top Store | CA_3 (1,310 units/day) | Prioritize for promotions |
| Highest Risk | CA_4–FOODS_1 (96.4%) | Daily monitoring needed |

## Methodology
1. **Data Processing**: Cleaned and optimized 10.9M rows with memory-efficient pipelines; stored in SQLite database
2. **Feature Engineering**: Created 15 temporal, event, and demand features with leak-proof rolling statistics
3. **SQL Analytics**: Revenue ranking (RANK), department tiers (NTILE), trend analysis (LAG), SNAP/weekend impact (CASE WHEN)
4. **Predictive Analytics**: 28-day forecasts with 67.8% error reduction over baseline; 95% prediction intervals
5. **Explainability**: SHAP analysis revealing top demand drivers (lag features = 86.6% importance)
6. **Dashboard**: 4-page interactive Streamlit app deployed via GitHub integration

## Results

| Metric | Value |
|--------|-------|
| Forecast Error Reduction | **67.8%** over baseline |
| Prediction Interval Coverage | **95%** |
| Annual Impact Estimate | **$650K – $1M** |
| Dashboard Pages | 4 (Overview, Forecasts, Events, Recommendations) |

## Tech Stack
Python · SQL (SQLite) · Pandas · Streamlit · SHAP · Plotly · Scikit-learn

## Deliverables
- ✅ 4-page interactive dashboard ([Live Link](https://retail-demand-forecasting-hariharan9597.streamlit.app/))
- ✅ SQL analytics with 6 production-quality queries
- ✅ Business recommendations document with dollar-impact projections
- ✅ 28-day demand forecasts with confidence intervals

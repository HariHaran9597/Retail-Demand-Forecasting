# How to Run the Project

## Quick Start Guide

### Prerequisites
- Python 3.9 or higher
- 4GB+ RAM
- M5 dataset downloaded (already in `data/raw/`)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Analysis Pipeline

Execute each day's analysis in order:

```bash
# Day 1: Data Understanding (5-10 minutes)
python run_day1.py

# Day 2: Feature Engineering (10-15 minutes)
python run_day2_optimized.py

# Day 3: Exploratory Data Analysis (5-10 minutes)
python run_day3.py

# Day 4: Modeling (15-20 minutes)
python run_day4.py

# Day 5: SHAP & Business Translation (10-15 minutes)
python run_day5.py
```

### Step 3: Launch the Dashboard

**Windows:**
```bash
run_dashboard.bat
```

**Linux/Mac:**
```bash
bash run_dashboard.sh
```

**Manual:**
```bash
cd app
streamlit run streamlit_app.py
```

The dashboard will open automatically at: **http://localhost:8501**

---

## Dashboard Features

### Page 1: Overview
- Key metrics (total sales, avg daily sales, stores, departments)
- Sales trend visualization
- Store performance comparison
- Weekly seasonality analysis
- Key insights cards (SNAP impact, weekend effect, top store)

### Page 2: Forecast Explorer
- Interactive store and department selection
- 28-day forecast with prediction intervals
- Performance metrics (MAE, accuracy)
- Next 7-day recommendations
- SNAP and weekend alerts

### Page 3: Event Impact Analyzer
- SNAP day analysis with lift percentages
- Weekend vs weekday comparison
- Event type impact (sporting, cultural, national, religious)
- Store-level SNAP impact breakdown

### Page 4: Business Recommendations
- Model performance summary
- 4 recommendation tabs:
  - Inventory optimization
  - Promotion planning
  - Staffing recommendations
  - Risk management
- Financial impact projections
- Implementation timeline

---

## Troubleshooting

### Issue: Module not found
```bash
pip install -r requirements.txt
```

### Issue: Data file not found
Ensure M5 dataset files are in `data/raw/`:
- calendar.csv
- sales_train_validation.csv
- sell_prices.csv

### Issue: Memory error
The scripts are already optimized for memory. If you still face issues:
- Close other applications
- Use the aggregated dataset (already implemented)
- Reduce sample sizes in analysis scripts

### Issue: Dashboard won't start
```bash
# Check if streamlit is installed
pip install streamlit plotly

# Try manual start
cd app
python -m streamlit run streamlit_app.py
```

### Issue: Port already in use
```bash
# Use a different port
streamlit run streamlit_app.py --server.port 8502
```

---

## Expected Output

### After Day 1:
- `data/processed/m5_data.db` - SQLite database
- `data/processed/ca_foods_merged.parquet` - Merged dataset
- `outputs/plots/day1_overview.png` - Visualization

### After Day 2:
- `data/processed/ca_foods_features.parquet` - Feature-engineered dataset
- `data/processed/ca_foods_store_dept_agg.parquet` - Aggregated dataset
- `outputs/plots/day2_features.png` - Feature visualizations

### After Day 3:
- 6 EDA visualizations in `outputs/plots/`
- Comprehensive analysis of trends, seasonality, events

### After Day 4:
- `outputs/models/prophet_models.pkl` - Prophet models
- `outputs/models/xgboost_model.json` - XGBoost model
- Model comparison and feature importance plots

### After Day 5:
- `outputs/plots/day5_shap_analysis.png` - SHAP visualizations
- `outputs/plots/day5_forecast_uncertainty.png` - Uncertainty analysis
- `outputs/BUSINESS_RECOMMENDATIONS.md` - Business recommendations

### Dashboard:
- Interactive web interface at http://localhost:8501
- 4 pages with visualizations and insights
- Real-time forecast exploration

---

## Performance Notes

### Execution Times (Approximate):
- Day 1: 5-10 minutes
- Day 2: 10-15 minutes
- Day 3: 5-10 minutes
- Day 4: 15-20 minutes (Prophet training)
- Day 5: 10-15 minutes (SHAP computation)
- **Total**: ~45-70 minutes

### Memory Usage:
- Peak: ~2-3 GB RAM
- Dashboard: ~500 MB RAM
- Optimized for standard laptops

---

## Tips for Best Experience

1. **Run in Order**: Execute Day 1 → Day 2 → ... → Day 5 sequentially
2. **Check Outputs**: Verify each day's outputs before proceeding
3. **Review Visualizations**: Open PNG files in `outputs/plots/` to see results
4. **Read Documentation**: Check `BUSINESS_RECOMMENDATIONS.md` for insights
5. **Explore Dashboard**: Spend time with each page to understand features

---

## For Deployment

### Deploy to Streamlit Cloud:

1. **Push to GitHub**:
```bash
git add .
git commit -m "Complete retail forecasting project"
git push origin main
```

2. **Deploy**:
- Go to https://streamlit.io/cloud
- Sign in with GitHub
- Click "New app"
- Select your repository
- Set main file: `app/streamlit_app.py`
- Click "Deploy"

3. **Share**:
- Get public URL
- Add to README
- Share in portfolio

---

## Support

If you encounter issues:
1. Check this guide first
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify data files are in correct locations

---

**Ready to run!** Start with `python run_day1.py` and follow the steps above.

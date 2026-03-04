"""
Day 4 - Modeling (Prophet & XGBoost)
With Walk-Forward Cross-Validation and Hyperparameter Tuning
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
import pickle
import json
import warnings
import time
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

print("=" * 70)
print("DAY 4 - MODELING (PROPHET & XGBOOST)")
print("  + Walk-Forward Cross-Validation")
print("  + Hyperparameter Tuning")
print("=" * 70)

# ──────────────── STEP 1: LOAD DATA ────────────────

print("\n[Step 1] Loading Feature-Engineered Data...")
df = pd.read_parquet('data/processed/ca_foods_store_dept_agg.parquet')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['store_id', 'dept_id', 'date'])

print(f"✓ Loaded shape: {df.shape}")
print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")

# ──────────────── STEP 2: TRAIN/TEST SPLIT ────────────────

print("\n[Step 2] Defining Forecast Horizon and Train/Test Split")
FORECAST_HORIZON = 28
TRAIN_END_DATE = df['date'].max() - pd.Timedelta(days=FORECAST_HORIZON)

train = df[df['date'] <= TRAIN_END_DATE].copy()
test = df[df['date'] > TRAIN_END_DATE].copy()

print(f"✓ Forecast horizon: {FORECAST_HORIZON} days")
print(f"✓ Train period: {train['date'].min()} to {train['date'].max()}")
print(f"✓ Test period: {test['date'].min()} to {test['date'].max()}")
print(f"✓ Train size: {len(train)} rows")
print(f"✓ Test size: {len(test)} rows")

# ──────────────── STEP 3: BASELINE MODEL ────────────────

print("\n[Step 3] Building Baseline Model (Seasonal Naive)")
print("Predicting next week's sales as last week's sales...")

baseline_predictions = []
for (store, dept), group in test.groupby(['store_id', 'dept_id']):
    train_group = train[(train['store_id'] == store) & (train['dept_id'] == dept)]
    
    for idx, row in group.iterrows():
        lag_date = row['date'] - pd.Timedelta(days=7)
        lag_sales = train_group[train_group['date'] == lag_date]['sales'].values
        
        if len(lag_sales) > 0:
            pred = lag_sales[0]
        else:
            pred = train_group['sales'].mean()
        
        baseline_predictions.append({
            'store_id': store, 'dept_id': dept,
            'date': row['date'], 'actual': row['sales'],
            'predicted': pred
        })

baseline_df = pd.DataFrame(baseline_predictions)
baseline_rmse = np.sqrt(mean_squared_error(baseline_df['actual'], baseline_df['predicted']))
baseline_mae = mean_absolute_error(baseline_df['actual'], baseline_df['predicted'])
baseline_r2 = r2_score(baseline_df['actual'], baseline_df['predicted'])

print(f"✓ Baseline RMSE: {baseline_rmse:.2f}")
print(f"✓ Baseline MAE: {baseline_mae:.2f}")
print(f"✓ Baseline R²: {baseline_r2:.4f}")

# ──────────────── STEP 4: PROPHET MODEL ────────────────

print("\n[Step 4] Building Prophet Model")
print("Training Prophet on store-department level...")

prophet_predictions = []
prophet_models = {}

for (store, dept), group in train.groupby(['store_id', 'dept_id']):
    print(f"  Training Prophet for {store} - {dept}...")
    
    prophet_df = group[['date', 'sales']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['snap'] = group['snap_CA'].values
    prophet_df['is_weekend'] = group['is_weekend'].values
    
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.add_regressor('snap')
    model.add_regressor('is_weekend')
    model.fit(prophet_df)
    prophet_models[f"{store}_{dept}"] = model
    
    test_group = test[(test['store_id'] == store) & (test['dept_id'] == dept)]
    future_df = test_group[['date', 'snap_CA', 'is_weekend']].copy()
    future_df.columns = ['ds', 'snap', 'is_weekend']
    
    forecast = model.predict(future_df)
    
    for idx, row in test_group.iterrows():
        pred_row = forecast[forecast['ds'] == row['date']]
        if len(pred_row) > 0:
            prophet_predictions.append({
                'store_id': store, 'dept_id': dept,
                'date': row['date'], 'actual': row['sales'],
                'predicted': max(0, pred_row['yhat'].values[0])
            })

prophet_pred_df = pd.DataFrame(prophet_predictions)
prophet_rmse = np.sqrt(mean_squared_error(prophet_pred_df['actual'], prophet_pred_df['predicted']))
prophet_mae = mean_absolute_error(prophet_pred_df['actual'], prophet_pred_df['predicted'])
prophet_r2 = r2_score(prophet_pred_df['actual'], prophet_pred_df['predicted'])

print(f"✓ Prophet RMSE: {prophet_rmse:.2f}")
print(f"✓ Prophet MAE: {prophet_mae:.2f}")
print(f"✓ Prophet R²: {prophet_r2:.4f}")

with open('outputs/models/prophet_models.pkl', 'wb') as f:
    pickle.dump(prophet_models, f)
print("✓ Prophet models saved")

# ──────────────── STEP 5: XGBOOST — FEATURE SETUP ────────────────

print("\n[Step 5] Building XGBoost Model (Global Approach)")
print("Training single XGBoost model across all store-departments...")

feature_cols = [
    'day_of_week', 'week_of_month', 'month', 'quarter', 'year',
    'is_weekend', 'snap_CA', 'has_event', 'is_sporting', 'is_cultural',
    'is_national', 'is_religious', 'is_promotion', 'sell_price',
    'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
    'sales_rolling_mean_7', 'sales_rolling_mean_28',
    'sales_rolling_std_7', 'sales_rolling_std_28'
]

train['store_num'] = train['store_id'].str.extract(r'(\d+)').astype(int)
train['dept_num'] = train['dept_id'].str.extract(r'(\d+)').astype(int)
test['store_num'] = test['store_id'].str.extract(r'(\d+)').astype(int)
test['dept_num'] = test['dept_id'].str.extract(r'(\d+)').astype(int)

feature_cols.extend(['store_num', 'dept_num'])

train_clean = train.dropna(subset=feature_cols)
test_clean = test.dropna(subset=feature_cols)

print(f"  Training samples: {len(train_clean)}")
print(f"  Test samples: {len(test_clean)}")

X_train = train_clean[feature_cols]
y_train = train_clean['sales']
X_test = test_clean[feature_cols]
y_test = test_clean['sales']

# ──────────────── STEP 5a: WALK-FORWARD CROSS-VALIDATION ────────────────

print("\n[Step 5a] Walk-Forward Cross-Validation (3 folds)")
print("=" * 60)

tscv = TimeSeriesSplit(n_splits=3)

cv_results = []
fold_num = 0

for train_idx, val_idx in tscv.split(X_train):
    fold_num += 1
    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    cv_model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    )
    cv_model.fit(X_cv_train, y_cv_train, verbose=False)
    cv_preds = np.maximum(cv_model.predict(X_cv_val), 0)
    
    cv_rmse = np.sqrt(mean_squared_error(y_cv_val, cv_preds))
    cv_mae = mean_absolute_error(y_cv_val, cv_preds)
    cv_r2 = r2_score(y_cv_val, cv_preds)
    
    cv_results.append({
        'fold': fold_num,
        'train_size': len(X_cv_train),
        'val_size': len(X_cv_val),
        'rmse': cv_rmse,
        'mae': cv_mae,
        'r2': cv_r2
    })
    
    print(f"  Fold {fold_num}: Train={len(X_cv_train)}, Val={len(X_cv_val)} | "
          f"RMSE={cv_rmse:.2f}, MAE={cv_mae:.2f}, R²={cv_r2:.4f}")

cv_df = pd.DataFrame(cv_results)
print(f"\n  CV Summary: RMSE={cv_df['rmse'].mean():.2f} ± {cv_df['rmse'].std():.2f}")
print(f"              MAE={cv_df['mae'].mean():.2f} ± {cv_df['mae'].std():.2f}")
print(f"              R²={cv_df['r2'].mean():.4f} ± {cv_df['r2'].std():.4f}")

# ──────────────── STEP 5b: HYPERPARAMETER TUNING ────────────────

print("\n[Step 5b] Hyperparameter Tuning (Grid Search)")
print("=" * 60)

param_grid = [
    {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
    {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8},
    {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.7},
    {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.08, 'subsample': 0.9, 'colsample_bytree': 0.9},
    {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.7},
    {'n_estimators': 250, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.85, 'colsample_bytree': 0.85},
]

tuning_results = []

# Use last fold of time series split for tuning validation
train_idx_tune, val_idx_tune = list(tscv.split(X_train))[-1]
X_tune_train, X_tune_val = X_train.iloc[train_idx_tune], X_train.iloc[val_idx_tune]
y_tune_train, y_tune_val = y_train.iloc[train_idx_tune], y_train.iloc[val_idx_tune]

for i, params in enumerate(param_grid):
    start_time = time.time()
    
    tune_model = xgb.XGBRegressor(
        **params, random_state=42, n_jobs=-1
    )
    tune_model.fit(X_tune_train, y_tune_train, verbose=False)
    tune_preds = np.maximum(tune_model.predict(X_tune_val), 0)
    
    tune_rmse = np.sqrt(mean_squared_error(y_tune_val, tune_preds))
    tune_mae = mean_absolute_error(y_tune_val, tune_preds)
    tune_r2 = r2_score(y_tune_val, tune_preds)
    elapsed = time.time() - start_time
    
    tuning_results.append({
        'config': i + 1,
        'params': str(params),
        'rmse': tune_rmse,
        'mae': tune_mae,
        'r2': tune_r2,
        'time_sec': elapsed,
        **params
    })
    
    print(f"  Config {i+1}: RMSE={tune_rmse:.2f}, MAE={tune_mae:.2f}, R²={tune_r2:.4f} ({elapsed:.1f}s)")

tuning_df = pd.DataFrame(tuning_results)
best_config = tuning_df.loc[tuning_df['rmse'].idxmin()]
print(f"\n  ★ Best Config #{int(best_config['config'])}: RMSE={best_config['rmse']:.2f}")
print(f"    Params: n_estimators={int(best_config['n_estimators'])}, "
      f"max_depth={int(best_config['max_depth'])}, "
      f"lr={best_config['learning_rate']}, "
      f"subsample={best_config['subsample']}")

# ──────────────── STEP 6: TRAIN FINAL MODEL WITH BEST PARAMS ────────────────

print("\n[Step 6] Training Final XGBoost with Best Hyperparameters")

best_params = {
    'n_estimators': int(best_config['n_estimators']),
    'max_depth': int(best_config['max_depth']),
    'learning_rate': best_config['learning_rate'],
    'subsample': best_config['subsample'],
    'colsample_bytree': best_config['colsample_bytree'],
    'random_state': 42,
    'n_jobs': -1
}

print(f"  Best params: {best_params}")

xgb_model = xgb.XGBRegressor(**best_params)

print("  Training XGBoost on full training set...")
xgb_model.fit(X_train, y_train, verbose=False)

# Predictions
xgb_predictions = np.maximum(xgb_model.predict(X_test), 0)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_r2 = r2_score(y_test, xgb_predictions)

print(f"✓ XGBoost RMSE: {xgb_rmse:.2f}")
print(f"✓ XGBoost MAE: {xgb_mae:.2f}")
print(f"✓ XGBoost R²: {xgb_r2:.4f}")

# Save model
xgb_model.save_model('outputs/models/xgboost_model.json')
print("✓ XGBoost model saved")

# ──────────────── STEP 7: MODEL COMPARISON ────────────────

print("\n[Step 7] Model Comparison")
print("=" * 60)

comparison_df = pd.DataFrame({
    'Model': ['Baseline (Seasonal Naive)', 'Prophet', 'XGBoost (Tuned)'],
    'RMSE': [baseline_rmse, prophet_rmse, xgb_rmse],
    'MAE': [baseline_mae, prophet_mae, xgb_mae],
    'R²': [baseline_r2, prophet_r2, xgb_r2]
})

print(comparison_df.to_string(index=False))

prophet_improvement = (baseline_rmse - prophet_rmse) / baseline_rmse * 100
xgb_improvement = (baseline_rmse - xgb_rmse) / baseline_rmse * 100

print(f"\nImprovement over Baseline:")
print(f"  Prophet: {prophet_improvement:+.1f}%")
print(f"  XGBoost: {xgb_improvement:+.1f}%")

# ──────────────── STEP 8: VISUALIZATIONS ────────────────

print("\n[Step 8] Creating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Model comparison
ax = axes[0, 0]
x = np.arange(len(comparison_df))
width = 0.35
bars1 = ax.bar(x - width/2, comparison_df['RMSE'], width, label='RMSE', color='steelblue')
bars2 = ax.bar(x + width/2, comparison_df['MAE'], width, label='MAE', color='coral')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Error')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=9)

# 2. Actual vs Predicted
ax = axes[0, 1]
sample_size = min(1000, len(y_test))
sample_idx = np.random.choice(len(y_test), sample_size, replace=False)
ax.scatter(y_test.iloc[sample_idx], xgb_predictions[sample_idx], alpha=0.3, s=10)
ax.plot([0, y_test.max()], [0, y_test.max()], 'r--', linewidth=2, label='Perfect')
ax.set_title('XGBoost: Actual vs Predicted', fontsize=14, fontweight='bold')
ax.set_xlabel('Actual Sales')
ax.set_ylabel('Predicted Sales')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Time series comparison
ax = axes[1, 0]
sample_store = 'CA_1'
sample_dept = 'FOODS_1'
test_sample = test_clean[(test_clean['store_id'] == sample_store) & 
                         (test_clean['dept_id'] == sample_dept)].copy()
test_sample['xgb_pred'] = xgb_model.predict(test_sample[feature_cols])
prophet_sample = prophet_pred_df[(prophet_pred_df['store_id'] == sample_store) & 
                                  (prophet_pred_df['dept_id'] == sample_dept)]
ax.plot(test_sample['date'], test_sample['sales'], 'o-', label='Actual', linewidth=2, markersize=4)
ax.plot(test_sample['date'], test_sample['xgb_pred'], 's-', label='XGBoost', linewidth=2, markersize=4, alpha=0.7)
if len(prophet_sample) > 0:
    ax.plot(prophet_sample['date'], prophet_sample['predicted'], '^-', label='Prophet', linewidth=2, markersize=4, alpha=0.7)
ax.set_title(f'Forecast: {sample_store} - {sample_dept}', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend()
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

# 4. Residual distribution
ax = axes[1, 1]
residuals = y_test - xgb_predictions
ax.hist(residuals, bins=50, color='lightblue', edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax.set_title('XGBoost Residual Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Residual (Actual - Predicted)')
ax.set_ylabel('Frequency')
ax.text(0.05, 0.95, f'Mean: {residuals.mean():.2f}\nStd: {residuals.std():.2f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('outputs/plots/day4_model_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/plots/day4_model_comparison.png")

# Feature Importance
print("\n[Step 9] Feature Importance Analysis")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
print(feature_importance.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 8))
top_features = feature_importance.head(15)
ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'])
ax.set_xlabel('Importance')
ax.set_title('Top 15 Feature Importances (XGBoost Tuned)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/plots/day4_feature_importance.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/plots/day4_feature_importance.png")

# CV visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.bar(cv_df['fold'], cv_df['rmse'], color='steelblue', alpha=0.8)
ax.axhline(y=cv_df['rmse'].mean(), color='red', linestyle='--', label=f"Mean: {cv_df['rmse'].mean():.2f}")
ax.set_title('Walk-Forward CV: RMSE per Fold', fontsize=14, fontweight='bold')
ax.set_xlabel('Fold')
ax.set_ylabel('RMSE')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
configs = [f"C{r['config']}" for _, r in tuning_df.iterrows()]
colors = ['green' if r['config'] == best_config['config'] else 'steelblue' for _, r in tuning_df.iterrows()]
ax.bar(configs, tuning_df['rmse'], color=colors, alpha=0.8)
ax.set_title('Hyperparameter Tuning: RMSE by Config', fontsize=14, fontweight='bold')
ax.set_xlabel('Config')
ax.set_ylabel('RMSE')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/plots/day4_cv_tuning.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/plots/day4_cv_tuning.png")

# ──────────────── STEP 10: ERROR ANALYSIS ────────────────

print("\n[Step 10] Error Analysis")

test_clean = test_clean.copy()
test_clean['xgb_pred'] = xgb_predictions
test_clean['error'] = test_clean['sales'] - test_clean['xgb_pred']
test_clean['abs_error'] = np.abs(test_clean['error'])
test_clean['pct_error'] = (test_clean['error'] / (test_clean['sales'] + 1)) * 100

print("\nError by Store:")
store_errors = test_clean.groupby('store_id').agg({
    'abs_error': 'mean', 'pct_error': 'mean'
}).round(2)
print(store_errors)

print("\nError by Department:")
dept_errors = test_clean.groupby('dept_id').agg({
    'abs_error': 'mean', 'pct_error': 'mean'
}).round(2)
print(dept_errors)

# ──────────────── STEP 11: SAVE ALL METRICS ────────────────

print("\n[Step 11] Saving All Metrics...")

all_metrics = {
    'models': {
        'baseline': {'rmse': round(baseline_rmse, 2), 'mae': round(baseline_mae, 2), 'r2': round(baseline_r2, 4)},
        'prophet': {'rmse': round(prophet_rmse, 2), 'mae': round(prophet_mae, 2), 'r2': round(prophet_r2, 4)},
        'xgboost_tuned': {'rmse': round(xgb_rmse, 2), 'mae': round(xgb_mae, 2), 'r2': round(xgb_r2, 4)}
    },
    'improvements': {
        'prophet_vs_baseline': round(prophet_improvement, 1),
        'xgboost_vs_baseline': round(xgb_improvement, 1)
    },
    'cross_validation': {
        'folds': cv_df.to_dict('records'),
        'mean_rmse': round(cv_df['rmse'].mean(), 2),
        'std_rmse': round(cv_df['rmse'].std(), 2),
        'mean_mae': round(cv_df['mae'].mean(), 2),
        'mean_r2': round(cv_df['r2'].mean(), 4)
    },
    'best_hyperparameters': {
        'n_estimators': int(best_config['n_estimators']),
        'max_depth': int(best_config['max_depth']),
        'learning_rate': float(best_config['learning_rate']),
        'subsample': float(best_config['subsample']),
        'colsample_bytree': float(best_config['colsample_bytree'])
    },
    'tuning_results': tuning_df[['config', 'rmse', 'mae', 'r2']].to_dict('records'),
    'feature_importance': feature_importance.head(15).to_dict('records'),
    'error_analysis': {
        'mean_residual': round(residuals.mean(), 2),
        'std_residual': round(residuals.std(), 2),
        'by_store': store_errors.to_dict(),
        'by_department': dept_errors.to_dict()
    }
}

with open('outputs/models/metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2, default=str)
print("✓ Metrics saved to: outputs/models/metrics.json")

# Summary
print("\n" + "=" * 70)
print("DAY 4 COMPLETE!")
print("=" * 70)

print(f"\nModel Performance Summary:")
print(f"1. Baseline: RMSE={baseline_rmse:.2f}, MAE={baseline_mae:.2f}, R²={baseline_r2:.4f}")
print(f"2. Prophet:  RMSE={prophet_rmse:.2f}, MAE={prophet_mae:.2f}, R²={prophet_r2:.4f} ({prophet_improvement:+.1f}%)")
print(f"3. XGBoost:  RMSE={xgb_rmse:.2f}, MAE={xgb_mae:.2f}, R²={xgb_r2:.4f} ({xgb_improvement:+.1f}%)")

print(f"\nCross-Validation: RMSE={cv_df['rmse'].mean():.2f} ± {cv_df['rmse'].std():.2f}")
print(f"Best Hyperparameters: n_est={int(best_config['n_estimators'])}, "
      f"depth={int(best_config['max_depth'])}, lr={best_config['learning_rate']}")

print(f"\nTop 3 features: {', '.join(feature_importance.head(3)['feature'].tolist())}")
print(f"Mean residual: {residuals.mean():.2f}, Std: {residuals.std():.2f}")

print("\nSaved:")
print("  - outputs/models/prophet_models.pkl")
print("  - outputs/models/xgboost_model.json")
print("  - outputs/models/metrics.json")
print("  - outputs/plots/day4_model_comparison.png")
print("  - outputs/plots/day4_feature_importance.png")
print("  - outputs/plots/day4_cv_tuning.png")

print("\nNext: Day 5 - Business Translation & SHAP Analysis")
print("=" * 70)

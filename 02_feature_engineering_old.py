"""
Day 2 - Data Cleaning & Feature Engineering
Execute this script to complete Day 2 analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')
from features import FeatureEngineer

# Set display options
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

print("=" * 70)
print("DAY 2 - DATA CLEANING & FEATURE ENGINEERING")
print("=" * 70)

# Load processed data from Day 1
print("\n[Step 1] Loading Day 1 processed data...")
df = pd.read_parquet('data/processed/ca_foods_merged.parquet')
print(f"✓ Loaded shape: {df.shape}")
print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")

# Data quality check
print("\n[Step 2] Data Quality Assessment")
print("\nMissing values by column:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)
print(f"\nTotal missing: {df.isnull().sum().sum()} values")

# Handle missing prices
print("\n[Step 3] Handling Missing Prices")
print(f"Missing prices before: {df['sell_price'].isna().sum()} ({df['sell_price'].isna().sum() / len(df) * 100:.1f}%)")

# Strategy: Forward fill within each item-store combination, then backward fill
df['sell_price'] = df.groupby(['store_id', 'item_id'])['sell_price'].ffill().bfill()

# If still missing, fill with category median
if df['sell_price'].isna().sum() > 0:
    cat_median = df.groupby('cat_id')['sell_price'].transform('median')
    df['sell_price'] = df['sell_price'].fillna(cat_median)

print(f"Missing prices after: {df['sell_price'].isna().sum()}")

# Convert date to datetime
print("\n[Step 4] Converting Date Column")
df['date'] = pd.to_datetime(df['date'])
print(f"✓ Date converted to datetime format")

# Feature Engineering
print("\n[Step 5] Feature Engineering")
engineer = FeatureEngineer(df)
df_features = engineer.create_all_features()

# Check for NaN in features
print("\n[Step 6] Feature Quality Check")
feature_cols = engineer.get_feature_names()
print(f"\nEngineered features ({len(feature_cols)}):")
for col in feature_cols:
    print(f"  - {col}")

nan_counts = df_features[feature_cols].isnull().sum()
if nan_counts.sum() > 0:
    print("\nFeatures with NaN values:")
    print(nan_counts[nan_counts > 0])

# Feature statistics
print("\n[Step 7] Feature Statistics")
print("\nLag feature correlations with sales:")
lag_cols = [col for col in feature_cols if 'lag' in col]
if lag_cols:
    lag_corr = df_features[lag_cols + ['sales']].corr()['sales'].drop('sales').sort_values(ascending=False)
    print(lag_corr)

print("\nRolling feature statistics:")
rolling_cols = [col for col in feature_cols if 'rolling' in col]
if rolling_cols:
    print(df_features[rolling_cols].describe())

# Visualizations
print("\n[Step 8] Creating Feature Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Sales by day of week
ax = axes[0, 0]
df_features.groupby('weekday')['sales'].mean().plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('Average Sales by Day of Week')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Average Sales')
ax.tick_params(axis='x', rotation=45)

# 2. SNAP impact
ax = axes[0, 1]
snap_data = df_features.groupby('snap_CA')['sales'].mean()
snap_data.plot(kind='bar', ax=ax, color=['coral', 'green'])
ax.set_title('SNAP Day Impact on Sales')
ax.set_xlabel('SNAP Day (0=No, 1=Yes)')
ax.set_ylabel('Average Sales')
ax.set_xticklabels(['Non-SNAP', 'SNAP'], rotation=0)

# 3. Event impact
ax = axes[0, 2]
event_data = df_features.groupby('has_event')['sales'].mean()
event_data.plot(kind='bar', ax=ax, color=['lightblue', 'orange'])
ax.set_title('Event Day Impact on Sales')
ax.set_xlabel('Has Event (0=No, 1=Yes)')
ax.set_ylabel('Average Sales')
ax.set_xticklabels(['No Event', 'Event'], rotation=0)

# 4. Price vs Sales scatter (sample)
ax = axes[1, 0]
sample = df_features.sample(min(10000, len(df_features)))
ax.scatter(sample['sell_price'], sample['sales'], alpha=0.1, s=1)
ax.set_title('Price vs Sales Relationship')
ax.set_xlabel('Sell Price ($)')
ax.set_ylabel('Sales')
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)

# 5. Promotion impact
ax = axes[1, 1]
promo_data = df_features.groupby('is_promotion')['sales'].mean()
promo_data.plot(kind='bar', ax=ax, color=['gray', 'red'])
ax.set_title('Promotion Impact on Sales')
ax.set_xlabel('Is Promotion (0=No, 1=Yes)')
ax.set_ylabel('Average Sales')
ax.set_xticklabels(['Regular Price', 'Promotion'], rotation=0)

# 6. Weekend vs Weekday
ax = axes[1, 2]
weekend_data = df_features.groupby('is_weekend')['sales'].mean()
weekend_data.plot(kind='bar', ax=ax, color=['skyblue', 'purple'])
ax.set_title('Weekend vs Weekday Sales')
ax.set_xlabel('Is Weekend (0=No, 1=Yes)')
ax.set_ylabel('Average Sales')
ax.set_xticklabels(['Weekday', 'Weekend'], rotation=0)

plt.tight_layout()
plt.savefig('outputs/plots/day2_features.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved to: outputs/plots/day2_features.png")

# Save feature-engineered dataset
print("\n[Step 9] Saving Feature-Engineered Dataset...")

# Save full dataset
output_path = Path('data/processed/ca_foods_features.parquet')
df_features.to_parquet(output_path, index=False)
print(f"✓ Full dataset saved to: {output_path}")

# Create and save aggregated dataset (store-department level)
print("\n[Step 10] Creating Store-Department Aggregated Dataset...")
agg_features = {
    'sales': 'sum',
    'sell_price': 'mean',
    'snap_CA': 'first',
    'has_event': 'max',
    'is_sporting': 'max',
    'is_cultural': 'max',
    'is_national': 'max',
    'is_religious': 'max',
    'is_weekend': 'first',
    'is_promotion': 'mean',
    'day_of_week': 'first',
    'week_of_month': 'first',
    'month': 'first',
    'quarter': 'first',
    'year': 'first'
}

df_agg = df_features.groupby(['store_id', 'dept_id', 'date']).agg(agg_features).reset_index()

# Add lag and rolling features at aggregated level
df_agg = df_agg.sort_values(['store_id', 'dept_id', 'date'])

for lag in [7, 14, 28]:
    df_agg[f'sales_lag_{lag}'] = df_agg.groupby(['store_id', 'dept_id'])['sales'].shift(lag)

for window in [7, 28]:
    df_agg[f'sales_rolling_mean_{window}'] = (
        df_agg.groupby(['store_id', 'dept_id'])['sales']
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    df_agg[f'sales_rolling_std_{window}'] = (
        df_agg.groupby(['store_id', 'dept_id'])['sales']
        .transform(lambda x: x.rolling(window, min_periods=1).std())
    )

agg_output_path = Path('data/processed/ca_foods_store_dept_agg.parquet')
df_agg.to_parquet(agg_output_path, index=False)
print(f"✓ Aggregated dataset saved to: {agg_output_path}")
print(f"  Shape: {df_agg.shape}")
print(f"  Store-Department combinations: {df_agg.groupby(['store_id', 'dept_id']).ngroups}")

# Summary statistics
print("\n" + "=" * 70)
print("DAY 2 COMPLETE!")
print("=" * 70)
print("\nKey Accomplishments:")
print(f"1. Cleaned missing values (prices filled using forward/backward fill)")
print(f"2. Created {len(feature_cols)} engineered features")
print(f"3. Built store-department aggregated dataset for modeling")
print(f"4. Validated feature quality and correlations")

print("\nKey Feature Insights:")
print(f"- SNAP days: {((df_features[df_features['snap_CA']==1]['sales'].mean() / df_features[df_features['snap_CA']==0]['sales'].mean()) - 1) * 100:.1f}% sales lift")
print(f"- Weekend days: {((df_features[df_features['is_weekend']==1]['sales'].mean() / df_features[df_features['is_weekend']==0]['sales'].mean()) - 1) * 100:.1f}% sales lift")
print(f"- Event days: {((df_features[df_features['has_event']==1]['sales'].mean() / df_features[df_features['has_event']==0]['sales'].mean()) - 1) * 100:.1f}% sales lift")
print(f"- Promotion days: {((df_features[df_features['is_promotion']==1]['sales'].mean() / df_features[df_features['is_promotion']==0]['sales'].mean()) - 1) * 100:.1f}% sales lift")

print("\nNext: Day 3 - Exploratory Data Analysis")
print("=" * 70)

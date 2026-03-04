"""
Day 2 - Data Cleaning & Feature Engineering (Memory Optimized)
Execute this script to complete Day 2 analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gc

# Set display options
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

print("=" * 70)
print("DAY 2 - DATA CLEANING & FEATURE ENGINEERING (OPTIMIZED)")
print("=" * 70)

# Load processed data from Day 1 - use only necessary columns
print("\n[Step 1] Loading Day 1 processed data (optimized)...")
cols_to_load = ['store_id', 'item_id', 'dept_id', 'cat_id', 'date', 'sales', 
                'sell_price', 'snap_CA', 'weekday', 'month', 'year', 'wm_yr_wk',
                'event_name_1', 'event_type_1']

df = pd.read_parquet('data/processed/ca_foods_merged.parquet', columns=cols_to_load)
print(f"✓ Loaded shape: {df.shape}")

# Convert date
df['date'] = pd.to_datetime(df['date'])
print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")

# Data quality check
print("\n[Step 2] Data Quality Assessment")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("\nMissing values by column:")
print(missing)

# Handle missing prices
print("\n[Step 3] Handling Missing Prices")
print(f"Missing prices before: {df['sell_price'].isna().sum()}")
df['sell_price'] = df.groupby(['store_id', 'item_id'])['sell_price'].ffill().bfill()
print(f"Missing prices after: {df['sell_price'].isna().sum()}")

# Feature Engineering - In place to save memory
print("\n[Step 4] Feature Engineering (in-place)")

# Calendar features
print("  - Calendar features...")
df['day_of_week'] = df['date'].dt.dayofweek
df['week_of_month'] = (df['date'].dt.day - 1) // 7 + 1
df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
df['quarter'] = df['date'].dt.quarter.astype('int8')

# Event features
print("  - Event features...")
df['has_event'] = (~df['event_name_1'].isna()).astype('int8')
df['is_sporting'] = (df['event_type_1'] == 'Sporting').astype('int8')
df['is_cultural'] = (df['event_type_1'] == 'Cultural').astype('int8')
df['is_national'] = (df['event_type_1'] == 'National').astype('int8')
df['is_religious'] = (df['event_type_1'] == 'Religious').astype('int8')

# Price features
print("  - Price features...")
df = df.sort_values(['store_id', 'item_id', 'date'])
df['price_change'] = df.groupby(['store_id', 'item_id'])['sell_price'].diff()
df['is_promotion'] = (df['price_change'] < 0).astype('int8')

# Lag features (only most important ones)
print("  - Lag features...")
for lag in [7, 28]:
    df[f'sales_lag_{lag}'] = df.groupby(['store_id', 'item_id'])['sales'].shift(lag)

# Rolling features (shifted to prevent data leakage - exclude current day)
print("  - Rolling features (leak-proof)...")
for window in [7, 28]:
    df[f'sales_rolling_mean_{window}'] = (
        df.groupby(['store_id', 'item_id'])['sales']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

print(f"\n✓ Feature engineering complete")
print(f"  Final shape: {df.shape}")

# Save feature-engineered dataset
print("\n[Step 5] Saving Feature-Engineered Dataset...")
output_path = Path('data/processed/ca_foods_features.parquet')
df.to_parquet(output_path, index=False)
print(f"✓ Saved to: {output_path}")

# Create aggregated dataset (store-department level)
print("\n[Step 6] Creating Store-Department Aggregated Dataset...")

agg_dict = {
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

df_agg = df.groupby(['store_id', 'dept_id', 'date']).agg(agg_dict).reset_index()
df_agg = df_agg.sort_values(['store_id', 'dept_id', 'date'])

# Add lag and rolling features at aggregated level
print("  - Adding aggregated lag features...")
for lag in [7, 14, 28]:
    df_agg[f'sales_lag_{lag}'] = df_agg.groupby(['store_id', 'dept_id'])['sales'].shift(lag)

print("  - Adding aggregated rolling features...")
for window in [7, 28]:
    df_agg[f'sales_rolling_mean_{window}'] = (
        df_agg.groupby(['store_id', 'dept_id'])['sales']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    df_agg[f'sales_rolling_std_{window}'] = (
        df_agg.groupby(['store_id', 'dept_id'])['sales']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
    )

agg_output_path = Path('data/processed/ca_foods_store_dept_agg.parquet')
df_agg.to_parquet(agg_output_path, index=False)
print(f"✓ Aggregated dataset saved to: {agg_output_path}")
print(f"  Shape: {df_agg.shape}")
print(f"  Store-Department combinations: {df_agg.groupby(['store_id', 'dept_id']).ngroups}")

# Visualizations
print("\n[Step 7] Creating Feature Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Sales by day of week
ax = axes[0, 0]
dow_sales = df.groupby('weekday')['sales'].mean()
dow_sales.plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('Average Sales by Day of Week')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Average Sales')
ax.tick_params(axis='x', rotation=45)

# 2. SNAP impact
ax = axes[0, 1]
snap_data = df.groupby('snap_CA')['sales'].mean()
snap_data.plot(kind='bar', ax=ax, color=['coral', 'green'])
ax.set_title('SNAP Day Impact on Sales')
ax.set_xlabel('SNAP Day (0=No, 1=Yes)')
ax.set_ylabel('Average Sales')
ax.set_xticklabels(['Non-SNAP', 'SNAP'], rotation=0)

# 3. Event impact
ax = axes[0, 2]
event_data = df.groupby('has_event')['sales'].mean()
event_data.plot(kind='bar', ax=ax, color=['lightblue', 'orange'])
ax.set_title('Event Day Impact on Sales')
ax.set_xlabel('Has Event (0=No, 1=Yes)')
ax.set_ylabel('Average Sales')
ax.set_xticklabels(['No Event', 'Event'], rotation=0)

# 4. Price vs Sales scatter (sample)
ax = axes[1, 0]
sample = df.sample(min(10000, len(df)))
ax.scatter(sample['sell_price'], sample['sales'], alpha=0.1, s=1)
ax.set_title('Price vs Sales Relationship')
ax.set_xlabel('Sell Price ($)')
ax.set_ylabel('Sales')
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)

# 5. Promotion impact
ax = axes[1, 1]
promo_data = df.groupby('is_promotion')['sales'].mean()
promo_data.plot(kind='bar', ax=ax, color=['gray', 'red'])
ax.set_title('Promotion Impact on Sales')
ax.set_xlabel('Is Promotion (0=No, 1=Yes)')
ax.set_ylabel('Average Sales')
ax.set_xticklabels(['Regular Price', 'Promotion'], rotation=0)

# 6. Weekend vs Weekday
ax = axes[1, 2]
weekend_data = df.groupby('is_weekend')['sales'].mean()
weekend_data.plot(kind='bar', ax=ax, color=['skyblue', 'purple'])
ax.set_title('Weekend vs Weekday Sales')
ax.set_xlabel('Is Weekend (0=No, 1=Yes)')
ax.set_ylabel('Average Sales')
ax.set_xticklabels(['Weekday', 'Weekend'], rotation=0)

plt.tight_layout()
plt.savefig('outputs/plots/day2_features.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved to: outputs/plots/day2_features.png")

# Summary statistics
print("\n" + "=" * 70)
print("DAY 2 COMPLETE!")
print("=" * 70)

print("\nKey Accomplishments:")
print(f"1. Cleaned missing values (prices filled using forward/backward fill)")
print(f"2. Created {df.shape[1] - len(cols_to_load)} engineered features")
print(f"3. Built store-department aggregated dataset for modeling")
print(f"4. Validated feature quality with visualizations")

print("\nKey Feature Insights:")
snap_lift = ((df[df['snap_CA']==1]['sales'].mean() / df[df['snap_CA']==0]['sales'].mean()) - 1) * 100
weekend_lift = ((df[df['is_weekend']==1]['sales'].mean() / df[df['is_weekend']==0]['sales'].mean()) - 1) * 100
event_lift = ((df[df['has_event']==1]['sales'].mean() / df[df['has_event']==0]['sales'].mean()) - 1) * 100
promo_lift = ((df[df['is_promotion']==1]['sales'].mean() / df[df['is_promotion']==0]['sales'].mean()) - 1) * 100

print(f"- SNAP days: {snap_lift:.1f}% sales lift")
print(f"- Weekend days: {weekend_lift:.1f}% sales lift")
print(f"- Event days: {event_lift:.1f}% sales lift")
print(f"- Promotion days: {promo_lift:.1f}% sales lift")

print("\nDatasets Created:")
print(f"1. Product-level features: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"2. Store-department aggregated: {df_agg.shape[0]:,} rows, {df_agg.shape[1]} columns")

print("\nNext: Day 3 - Exploratory Data Analysis")
print("=" * 70)

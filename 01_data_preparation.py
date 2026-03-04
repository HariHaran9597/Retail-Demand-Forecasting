"""
Day 1 - Data Understanding & Environment Setup
Execute this script to complete Day 1 analysis
"""
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')
from data_pipeline import M5DataLoader
from utils import calculate_zero_sales_pct, calculate_snap_lift

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
sns.set_style('whitegrid')

print("=" * 70)
print("DAY 1 - DATA UNDERSTANDING & ENVIRONMENT SETUP")
print("=" * 70)

# Initialize data loader
loader = M5DataLoader()

# Step 1: Load raw data
print("\n[Step 1] Loading raw data files...")
sales, calendar, prices = loader.load_raw_data()
print(f"✓ Sales shape: {sales.shape}")
print(f"✓ Calendar shape: {calendar.shape}")
print(f"✓ Prices shape: {prices.shape}")

# Step 2: Inspect data structure
print("\n[Step 2] Data Structure Inspection")
print("\nSales columns (first 15):")
print(sales.columns.tolist()[:15])
print("\nCalendar columns:")
print(calendar.columns.tolist())
print("\nPrices columns:")
print(prices.columns.tolist())

# Step 3: Create SQLite database
print("\n[Step 3] Creating SQLite database...")
loader.create_database(sales, calendar, prices)

# Step 4: Query hierarchy statistics
print("\n[Step 4] Hierarchy Statistics")
hierarchy_stats = loader.get_hierarchy_stats()
print(hierarchy_stats)

# Step 5: Date range
print("\n[Step 5] Date Range Analysis")
date_query = """
SELECT 
    MIN(date) as start_date,
    MAX(date) as end_date,
    COUNT(DISTINCT date) as num_days
FROM calendar
"""
date_range = loader.query(date_query)
print(date_range)

# Step 6: Category breakdown
print("\n[Step 6] Category Breakdown")
cat_query = """
SELECT 
    cat_id,
    COUNT(DISTINCT item_id) as num_items,
    COUNT(DISTINCT dept_id) as num_departments
FROM sales
GROUP BY cat_id
"""
categories = loader.query(cat_query)
print(categories)

# Step 7: Store breakdown
print("\n[Step 7] Store Breakdown by State")
store_query = """
SELECT 
    state_id,
    COUNT(DISTINCT store_id) as num_stores,
    COUNT(DISTINCT item_id) as num_items
FROM sales
GROUP BY state_id
"""
stores = loader.query(store_query)
print(stores)

# Step 8: Filter to California Foods
print("\n[Step 8] Filtering to California Foods subset...")
ca_foods = loader.filter_subset(sales, state='CA', category='FOODS')
print(f"✓ California Foods products: {ca_foods.shape[0]}")
print(f"✓ Stores: {ca_foods['store_id'].nunique()}")
print(f"✓ Departments: {ca_foods['dept_id'].nunique()}")

# Step 9: Intermittent demand analysis
print("\n[Step 9] Intermittent Demand Analysis (sample of 1000 products)...")
day_cols = [col for col in sales.columns if col.startswith('d_')]
sample_sales = ca_foods.head(1000)
zero_pct = calculate_zero_sales_pct(sample_sales, day_cols)

print(f"Zero-sales day percentage:")
print(f"  Mean: {zero_pct.mean():.1f}%")
print(f"  Median: {zero_pct.median():.1f}%")
print(f"  Products with >50% zero days: {(zero_pct > 50).sum()}")
print(f"  Products with >30% zero days: {(zero_pct > 30).sum()}")

# Step 10: Transform to long format (memory optimized)
print("\n[Step 10] Transforming to long format (memory optimized)...")
print("Processing in chunks to manage memory...")

# Process in smaller batches
chunk_size = 1000
num_chunks = (len(ca_foods) + chunk_size - 1) // chunk_size
merged_chunks = []

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(ca_foods))
    
    if i % 5 == 0:
        print(f"  Processing chunk {i+1}/{num_chunks}...")
    
    # Process chunk
    chunk = ca_foods.iloc[start_idx:end_idx]
    chunk_long = loader.melt_sales(chunk)
    
    # Merge with calendar (smaller table)
    chunk_merged = chunk_long.merge(calendar, on='d', how='left')
    
    # Merge with prices
    chunk_merged = chunk_merged.merge(
        prices,
        on=['store_id', 'item_id', 'wm_yr_wk'],
        how='left'
    )
    
    merged_chunks.append(chunk_merged)
    
    # Clear memory
    del chunk, chunk_long

# Combine all chunks
print("  Combining chunks...")
ca_foods_merged = pd.concat(merged_chunks, ignore_index=True)
del merged_chunks

print(f"✓ Final merged shape: {ca_foods_merged.shape}")
print(f"✓ Missing prices: {ca_foods_merged['sell_price'].isna().sum()} rows ({ca_foods_merged['sell_price'].isna().sum() / len(ca_foods_merged) * 100:.1f}%)")

# Step 11: SNAP impact analysis
print("\n[Step 11] SNAP Impact Analysis")
snap_impact = ca_foods_merged.groupby('snap_CA')['sales'].agg(['mean', 'median', 'count'])
print(snap_impact)
snap_lift = calculate_snap_lift(ca_foods_merged, 'snap_CA')
print(f"\n✓ SNAP days show {snap_lift:.1f}% higher average sales")

# Step 12: Event impact analysis
print("\n[Step 12] Top 10 Events by Sales Impact")
event_impact = ca_foods_merged.groupby('event_name_1')['sales'].mean().sort_values(ascending=False).head(10)
print(event_impact)

# Step 13: Day of week analysis
print("\n[Step 13] Day of Week Analysis")
dow_impact = ca_foods_merged.groupby('weekday')['sales'].mean().sort_values(ascending=False)
print(dow_impact)

# Step 14: Save processed data
print("\n[Step 14] Saving processed data...")
output_path = Path('data/processed/ca_foods_merged.parquet')
ca_foods_merged.to_parquet(output_path, index=False)
print(f"✓ Saved to: {output_path}")

# Step 15: Create sample visualization
print("\n[Step 15] Creating sample visualization...")
plt.figure(figsize=(14, 5))

# Plot 1: Sample product time series
sample_item = ca_foods_merged['item_id'].iloc[0]
sample_data = ca_foods_merged[ca_foods_merged['item_id'] == sample_item].copy()
sample_data['date'] = pd.to_datetime(sample_data['date'])
sample_data = sample_data.sort_values('date')

plt.subplot(1, 2, 1)
plt.plot(sample_data['date'], sample_data['sales'], linewidth=0.8)
plt.title(f'Sample Product Sales Pattern\n{sample_item}')
plt.xlabel('Date')
plt.ylabel('Daily Sales')
plt.xticks(rotation=45)

# Plot 2: SNAP vs Non-SNAP sales distribution
plt.subplot(1, 2, 2)
snap_data = ca_foods_merged[ca_foods_merged['snap_CA'] == 1]['sales']
non_snap_data = ca_foods_merged[ca_foods_merged['snap_CA'] == 0]['sales']
plt.hist([non_snap_data, snap_data], bins=50, label=['Non-SNAP', 'SNAP'], alpha=0.7)
plt.title('Sales Distribution: SNAP vs Non-SNAP Days')
plt.xlabel('Daily Sales')
plt.ylabel('Frequency')
plt.legend()
plt.xlim(0, 20)

plt.tight_layout()
plt.savefig('outputs/plots/day1_overview.png', dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved to: outputs/plots/day1_overview.png")

# Summary
print("\n" + "=" * 70)
print("DAY 1 COMPLETE!")
print("=" * 70)
print("\nKey Insights:")
print(f"1. Dataset covers {date_range['num_days'].iloc[0]} days of sales data")
print(f"2. California Foods subset: {ca_foods.shape[0]} products across {ca_foods['store_id'].nunique()} stores")
print(f"3. SNAP days drive {snap_lift:.1f}% higher sales on average")
print(f"4. Intermittent demand is significant - many products have frequent zero-sales days")
print(f"5. Price data has {ca_foods_merged['sell_price'].isna().sum() / len(ca_foods_merged) * 100:.1f}% missing values to handle")
print("\nNext: Day 2 - Feature Engineering")
print("=" * 70)

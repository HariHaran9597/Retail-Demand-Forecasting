"""
Day 3 - Exploratory Data Analysis (EDA)
Execute this script to complete Day 3 analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

print("=" * 70)
print("DAY 3 - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Load aggregated data (more manageable for EDA)
print("\n[Step 1] Loading Feature-Engineered Data...")
df_agg = pd.read_parquet('data/processed/ca_foods_store_dept_agg.parquet')
df_agg['date'] = pd.to_datetime(df_agg['date'])
df_agg = df_agg.sort_values(['store_id', 'dept_id', 'date'])

print(f"✓ Loaded shape: {df_agg.shape}")
print(f"✓ Date range: {df_agg['date'].min()} to {df_agg['date'].max()}")
print(f"✓ Store-Department combinations: {df_agg.groupby(['store_id', 'dept_id']).ngroups}")

# Section 1: Overall Demand Trends
print("\n" + "="*70)
print("SECTION 1: OVERALL DEMAND TRENDS")
print("="*70)

# Total sales over time
total_sales = df_agg.groupby('date')['sales'].sum().reset_index()
total_sales['date'] = pd.to_datetime(total_sales['date'])

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Plot 1: Total sales time series
ax = axes[0]
ax.plot(total_sales['date'], total_sales['sales'], linewidth=0.8, color='steelblue')
ax.set_title('California Foods - Total Daily Sales Over Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Total Daily Sales (Units)')
ax.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(range(len(total_sales)), total_sales['sales'], 1)
p = np.poly1d(z)
ax.plot(total_sales['date'], p(range(len(total_sales))), "r--", alpha=0.8, linewidth=2, label='Trend')
ax.legend()

# Plot 2: Monthly aggregation
ax = axes[1]
total_sales['year_month'] = total_sales['date'].dt.to_period('M')
monthly_sales = total_sales.groupby('year_month')['sales'].sum()
monthly_sales.plot(kind='bar', ax=ax, color='coral', width=0.8)
ax.set_title('Monthly Total Sales', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Total Sales (Units)')
ax.tick_params(axis='x', rotation=45, labelsize=8)
# Show every 6th label
for i, label in enumerate(ax.xaxis.get_ticklabels()):
    if i % 6 != 0:
        label.set_visible(False)

plt.tight_layout()
plt.savefig('outputs/plots/day3_01_overall_trends.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/plots/day3_01_overall_trends.png")

# Calculate trend statistics
trend_slope = z[0]
avg_daily_sales = total_sales['sales'].mean()
print(f"\nOverall Trend Statistics:")
print(f"  Average daily sales: {avg_daily_sales:,.0f} units")
print(f"  Trend: {trend_slope:+.2f} units/day ({trend_slope/avg_daily_sales*100:+.3f}% per day)")
print(f"  Total period growth: {(total_sales['sales'].iloc[-1] / total_sales['sales'].iloc[0] - 1) * 100:+.1f}%")

# Section 2: Seasonality Decomposition
print("\n" + "="*70)
print("SECTION 2: SEASONALITY DECOMPOSITION")
print("="*70)

# Select one representative store-department for decomposition
sample_series = df_agg[(df_agg['store_id'] == 'CA_1') & (df_agg['dept_id'] == 'FOODS_1')].copy()
sample_series = sample_series.set_index('date')['sales']

# Perform decomposition (use additive for series with zeros)
print("Performing seasonal decomposition (additive)...")
decomposition = seasonal_decompose(sample_series, model='additive', period=7, extrapolate_trend='freq')

fig, axes = plt.subplots(4, 1, figsize=(16, 12))

# Original
axes[0].plot(decomposition.observed, linewidth=0.8)
axes[0].set_title('Original Time Series (CA_1, FOODS_1)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Sales')

# Trend
axes[1].plot(decomposition.trend, linewidth=1.5, color='red')
axes[1].set_title('Trend Component', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Trend')

# Seasonal
axes[2].plot(decomposition.seasonal, linewidth=0.8, color='green')
axes[2].set_title('Seasonal Component (Weekly Pattern)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Seasonal')

# Residual
axes[3].plot(decomposition.resid, linewidth=0.5, color='gray', alpha=0.7)
axes[3].set_title('Residual Component', fontsize=12, fontweight='bold')
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Date')

plt.tight_layout()
plt.savefig('outputs/plots/day3_02_seasonality_decomposition.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/plots/day3_02_seasonality_decomposition.png")

# Analyze weekly pattern
df_agg['weekday'] = df_agg['date'].dt.day_name()
weekly_pattern = df_agg.groupby('weekday')['sales'].mean()
# Sort by day of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_pattern = weekly_pattern.reindex(day_order)
print(f"\nWeekly Seasonality Pattern:")
for day, sales in weekly_pattern.items():
    print(f"  {day}: {sales:,.1f} units ({sales/weekly_pattern.mean()*100-100:+.1f}% vs avg)")

# Section 3: Event Impact Analysis
print("\n" + "="*70)
print("SECTION 3: EVENT IMPACT ANALYSIS")
print("="*70)

# Load product-level data for event analysis
print("Loading product-level data for detailed event analysis...")
df_events = pd.read_parquet('data/processed/ca_foods_features.parquet', 
                             columns=['date', 'sales', 'snap_CA', 'has_event', 
                                     'is_sporting', 'is_cultural', 'is_national', 'is_religious'])

# SNAP impact
snap_comparison = df_events.groupby('snap_CA')['sales'].agg(['mean', 'median', 'std', 'count'])
snap_lift = (snap_comparison.loc[1, 'mean'] / snap_comparison.loc[0, 'mean'] - 1) * 100

print(f"\nSNAP Day Impact:")
print(snap_comparison)
print(f"  Lift: {snap_lift:+.1f}%")

# Event type impact
event_types = ['is_sporting', 'is_cultural', 'is_national', 'is_religious']
event_impacts = {}

for event_type in event_types:
    event_avg = df_events[df_events[event_type] == 1]['sales'].mean()
    no_event_avg = df_events[df_events[event_type] == 0]['sales'].mean()
    lift = (event_avg / no_event_avg - 1) * 100
    event_impacts[event_type.replace('is_', '')] = lift

print(f"\nEvent Type Impacts:")
for event, lift in event_impacts.items():
    print(f"  {event.capitalize()}: {lift:+.1f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# SNAP impact
ax = axes[0, 0]
snap_data = df_events.groupby('snap_CA')['sales'].mean()
bars = ax.bar(['Non-SNAP', 'SNAP'], snap_data.values, color=['lightcoral', 'lightgreen'])
ax.set_title(f'SNAP Day Impact: {snap_lift:+.1f}% Lift', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Sales per Product')
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{snap_data.values[i]:.2f}',
            ha='center', va='bottom')

# Event types
ax = axes[0, 1]
event_names = list(event_impacts.keys())
event_lifts = list(event_impacts.values())
colors = ['green' if x > 0 else 'red' for x in event_lifts]
bars = ax.barh(event_names, event_lifts, color=colors, alpha=0.7)
ax.set_title('Event Type Impact on Sales', fontsize=12, fontweight='bold')
ax.set_xlabel('Sales Lift (%)')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:+.1f}%',
            ha='left' if width > 0 else 'right', va='center', fontsize=10)

# SNAP + Event interaction
ax = axes[1, 0]
interaction = df_events.groupby(['snap_CA', 'has_event'])['sales'].mean().unstack()
interaction.plot(kind='bar', ax=ax, color=['lightblue', 'orange'])
ax.set_title('SNAP × Event Interaction', fontsize=12, fontweight='bold')
ax.set_xlabel('SNAP Day')
ax.set_ylabel('Average Sales')
ax.set_xticklabels(['Non-SNAP', 'SNAP'], rotation=0)
ax.legend(['No Event', 'Has Event'])

# Sales distribution by SNAP
ax = axes[1, 1]
snap_sales = df_events[df_events['snap_CA'] == 1]['sales']
non_snap_sales = df_events[df_events['snap_CA'] == 0]['sales']
ax.hist([non_snap_sales, snap_sales], bins=50, alpha=0.6, label=['Non-SNAP', 'SNAP'], 
        range=(0, 10), density=True)
ax.set_title('Sales Distribution: SNAP vs Non-SNAP', fontsize=12, fontweight='bold')
ax.set_xlabel('Daily Sales per Product')
ax.set_ylabel('Density')
ax.legend()

plt.tight_layout()
plt.savefig('outputs/plots/day3_03_event_impact.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/plots/day3_03_event_impact.png")

# Section 4: Store Comparison
print("\n" + "="*70)
print("SECTION 4: STORE COMPARISON ANALYSIS")
print("="*70)

# Store-level statistics
store_stats = df_agg.groupby('store_id').agg({
    'sales': ['mean', 'std', 'sum'],
    'sell_price': 'mean',
    'is_promotion': 'mean'
}).round(2)

print("\nStore Performance Comparison:")
print(store_stats)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Average sales by store
ax = axes[0, 0]
store_avg = df_agg.groupby('store_id')['sales'].mean().sort_values(ascending=False)
bars = ax.bar(store_avg.index, store_avg.values, color='steelblue')
ax.set_title('Average Daily Sales by Store', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Sales')
ax.set_xlabel('Store')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:,.0f}',
            ha='center', va='bottom')

# Sales volatility (coefficient of variation)
ax = axes[0, 1]
store_cv = (df_agg.groupby('store_id')['sales'].std() / 
            df_agg.groupby('store_id')['sales'].mean()).sort_values(ascending=False)
bars = ax.bar(store_cv.index, store_cv.values, color='coral')
ax.set_title('Sales Volatility by Store (CV)', fontsize=12, fontweight='bold')
ax.set_ylabel('Coefficient of Variation')
ax.set_xlabel('Store')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom')

# Department mix by store
ax = axes[1, 0]
dept_mix = df_agg.groupby(['store_id', 'dept_id'])['sales'].sum().unstack()
dept_mix.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
ax.set_title('Department Sales Mix by Store', fontsize=12, fontweight='bold')
ax.set_ylabel('Total Sales')
ax.set_xlabel('Store')
ax.legend(title='Department', bbox_to_anchor=(1.05, 1))
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# Time series by store
ax = axes[1, 1]
for store in df_agg['store_id'].unique():
    store_ts = df_agg[df_agg['store_id'] == store].groupby('date')['sales'].sum()
    ax.plot(store_ts.index, store_ts.values, label=store, alpha=0.7, linewidth=1)
ax.set_title('Sales Trends by Store', fontsize=12, fontweight='bold')
ax.set_ylabel('Daily Sales')
ax.set_xlabel('Date')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/plots/day3_04_store_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/plots/day3_04_store_comparison.png")

# Section 5: Intermittent Demand Analysis
print("\n" + "="*70)
print("SECTION 5: INTERMITTENT DEMAND ANALYSIS")
print("="*70)

# Load product-level data
print("Analyzing intermittent demand patterns...")
df_products = pd.read_parquet('data/processed/ca_foods_features.parquet',
                               columns=['item_id', 'store_id', 'date', 'sales'])

# Calculate zero-sales percentage per product
zero_pct = df_products.groupby(['store_id', 'item_id'])['sales'].apply(
    lambda x: (x == 0).sum() / len(x) * 100
).reset_index()
zero_pct.columns = ['store_id', 'item_id', 'zero_pct']

print(f"\nIntermittent Demand Statistics:")
print(f"  Mean zero-sales percentage: {zero_pct['zero_pct'].mean():.1f}%")
print(f"  Median zero-sales percentage: {zero_pct['zero_pct'].median():.1f}%")
print(f"  Products with >50% zero days: {(zero_pct['zero_pct'] > 50).sum()} ({(zero_pct['zero_pct'] > 50).sum() / len(zero_pct) * 100:.1f}%)")
print(f"  Products with >70% zero days: {(zero_pct['zero_pct'] > 70).sum()} ({(zero_pct['zero_pct'] > 70).sum() / len(zero_pct) * 100:.1f}%)")

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution of zero-sales percentage
ax = axes[0]
ax.hist(zero_pct['zero_pct'], bins=50, color='skyblue', edgecolor='black')
ax.set_title('Distribution of Zero-Sales Percentage', fontsize=12, fontweight='bold')
ax.set_xlabel('Zero-Sales Days (%)')
ax.set_ylabel('Number of Products')
ax.axvline(x=50, color='red', linestyle='--', label='50% threshold')
ax.legend()

# Categorize products
ax = axes[1]
categories = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
bins = [0, 20, 40, 60, 80, 100]
zero_pct['category'] = pd.cut(zero_pct['zero_pct'], bins=bins, labels=categories)
category_counts = zero_pct['category'].value_counts().sort_index()
bars = ax.bar(categories, category_counts.values, color='coral')
ax.set_title('Products by Intermittency Level', fontsize=12, fontweight='bold')
ax.set_xlabel('Zero-Sales Days (%)')
ax.set_ylabel('Number of Products')
ax.tick_params(axis='x', rotation=45)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom')

plt.tight_layout()
plt.savefig('outputs/plots/day3_05_intermittent_demand.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/plots/day3_05_intermittent_demand.png")

# Section 6: Price Elasticity Exploration
print("\n" + "="*70)
print("SECTION 6: PRICE ELASTICITY EXPLORATION")
print("="*70)

# Load data with prices
print("Analyzing price-sales relationship...")
df_price = pd.read_parquet('data/processed/ca_foods_features.parquet',
                            columns=['item_id', 'store_id', 'date', 'sales', 'sell_price', 'price_change'])

# Remove outliers and missing values
df_price = df_price[(df_price['sell_price'] > 0) & (df_price['sell_price'] < 50)]
df_price = df_price[df_price['sales'] < df_price['sales'].quantile(0.99)]

# Calculate correlation
price_sales_corr = df_price[['sell_price', 'sales']].corr().iloc[0, 1]
print(f"\nPrice-Sales Correlation: {price_sales_corr:.3f}")

# Simple elasticity estimate (% change in quantity / % change in price)
# Group by item and calculate
elasticity_data = []
for item in df_price['item_id'].unique()[:100]:  # Sample 100 items
    item_data = df_price[df_price['item_id'] == item].copy()
    if len(item_data) > 50 and item_data['sell_price'].std() > 0:
        # Calculate percentage changes
        item_data = item_data.sort_values('date')
        item_data['price_pct_change'] = item_data['sell_price'].pct_change()
        item_data['sales_pct_change'] = item_data['sales'].pct_change()
        
        # Remove infinities and NaNs
        item_data = item_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(item_data) > 10:
            # Simple elasticity: correlation of % changes
            if item_data['price_pct_change'].std() > 0:
                elasticity = item_data[['price_pct_change', 'sales_pct_change']].corr().iloc[0, 1]
                elasticity_data.append({
                    'item_id': item,
                    'elasticity': elasticity,
                    'avg_price': item_data['sell_price'].mean(),
                    'avg_sales': item_data['sales'].mean()
                })

elasticity_df = pd.DataFrame(elasticity_data)
print(f"\nElasticity Analysis (sample of {len(elasticity_df)} products):")
print(f"  Mean elasticity: {elasticity_df['elasticity'].mean():.3f}")
print(f"  Median elasticity: {elasticity_df['elasticity'].median():.3f}")
print(f"  Products with negative elasticity: {(elasticity_df['elasticity'] < 0).sum()} ({(elasticity_df['elasticity'] < 0).sum() / len(elasticity_df) * 100:.1f}%)")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Price vs Sales scatter
ax = axes[0, 0]
sample = df_price.sample(min(10000, len(df_price)))
ax.scatter(sample['sell_price'], sample['sales'], alpha=0.1, s=1)
ax.set_title(f'Price vs Sales Relationship (r={price_sales_corr:.3f})', fontsize=12, fontweight='bold')
ax.set_xlabel('Sell Price ($)')
ax.set_ylabel('Sales (Units)')
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)

# Add trend line
z = np.polyfit(sample['sell_price'], sample['sales'], 1)
p = np.poly1d(z)
ax.plot(sorted(sample['sell_price']), p(sorted(sample['sell_price'])), "r-", linewidth=2, label='Trend')
ax.legend()

# Price distribution
ax = axes[0, 1]
ax.hist(df_price['sell_price'], bins=50, color='lightgreen', edgecolor='black')
ax.set_title('Price Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Sell Price ($)')
ax.set_ylabel('Frequency')

# Elasticity distribution
ax = axes[1, 0]
if len(elasticity_df) > 0:
    ax.hist(elasticity_df['elasticity'], bins=30, color='lightcoral', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero elasticity')
    ax.set_title('Price Elasticity Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Elasticity')
    ax.set_ylabel('Number of Products')
    ax.legend()

# Price change impact
ax = axes[1, 1]
df_price['price_change_bin'] = pd.cut(df_price['price_change'], 
                                       bins=[-np.inf, -1, -0.1, 0.1, 1, np.inf],
                                       labels=['Large Drop', 'Small Drop', 'No Change', 'Small Increase', 'Large Increase'])
price_change_impact = df_price.groupby('price_change_bin')['sales'].mean()
bars = ax.bar(range(len(price_change_impact)), price_change_impact.values, 
              color=['green', 'lightgreen', 'gray', 'lightcoral', 'red'])
ax.set_title('Sales by Price Change Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Sales')
ax.set_xticks(range(len(price_change_impact)))
ax.set_xticklabels(price_change_impact.index, rotation=45, ha='right')
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom')

plt.tight_layout()
plt.savefig('outputs/plots/day3_06_price_elasticity.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/plots/day3_06_price_elasticity.png")

# Final Summary
print("\n" + "=" * 70)
print("DAY 3 COMPLETE!")
print("=" * 70)

print("\nKey Insights Summary:")
print("\n1. OVERALL TRENDS:")
print(f"   - Average daily sales: {avg_daily_sales:,.0f} units")
print(f"   - Trend: {trend_slope:+.2f} units/day")

print("\n2. SEASONALITY:")
print(f"   - Strong weekly pattern detected")
weekend_days = ['Saturday', 'Sunday']
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
print(f"   - Weekend sales: {weekly_pattern[weekend_days].mean():,.1f} units/day")
print(f"   - Weekday sales: {weekly_pattern[weekdays].mean():,.1f} units/day")

print("\n3. EVENT IMPACT:")
print(f"   - SNAP days: {snap_lift:+.1f}% sales lift")
for event, lift in event_impacts.items():
    print(f"   - {event.capitalize()} events: {lift:+.1f}% impact")

print("\n4. STORE COMPARISON:")
print(f"   - Highest performing store: {store_avg.index[0]} ({store_avg.values[0]:,.0f} units/day)")
print(f"   - Most volatile store: {store_cv.index[0]} (CV: {store_cv.values[0]:.2f})")

print("\n5. INTERMITTENT DEMAND:")
print(f"   - {(zero_pct['zero_pct'] > 50).sum() / len(zero_pct) * 100:.1f}% of products have >50% zero-sales days")
print(f"   - Requires specialized forecasting approaches")

print("\n6. PRICE ELASTICITY:")
print(f"   - Overall price-sales correlation: {price_sales_corr:.3f}")
if len(elasticity_df) > 0:
    print(f"   - Mean elasticity: {elasticity_df['elasticity'].mean():.3f}")
    print(f"   - {(elasticity_df['elasticity'] < 0).sum() / len(elasticity_df) * 100:.1f}% of products show negative elasticity")

print("\nVisualizations Created:")
print("  1. outputs/plots/day3_01_overall_trends.png")
print("  2. outputs/plots/day3_02_seasonality_decomposition.png")
print("  3. outputs/plots/day3_03_event_impact.png")
print("  4. outputs/plots/day3_04_store_comparison.png")
print("  5. outputs/plots/day3_05_intermittent_demand.png")
print("  6. outputs/plots/day3_06_price_elasticity.png")

print("\nNext: Day 4 - Modeling (Prophet & XGBoost)")
print("=" * 70)

"""
06 - SQL Analytics & Reporting
SQL-based demand analysis on the SQLite database
Demonstrates: Window functions, CTEs, CASE WHEN, GROUP BY, aggregations
"""
import sqlite3
import pandas as pd
from pathlib import Path

print("=" * 70)
print("SQL ANALYTICS & REPORTING")
print("=" * 70)

# Create output directory
Path('outputs/sql_results').mkdir(parents=True, exist_ok=True)

# Connect to database
db_path = 'data/processed/m5_data.db'
conn = sqlite3.connect(db_path)
print(f"✓ Connected to: {db_path}")

# Also load the aggregated parquet for additional SQL analysis
df_agg = pd.read_parquet('data/processed/ca_foods_store_dept_agg.parquet')
df_agg.to_sql('store_dept_daily', conn, if_exists='replace', index=False)
print(f"✓ Loaded aggregated data into SQL: {len(df_agg)} rows")

# ──────────────── QUERY 1: REVENUE BY STORE (RANKED) ────────────────

print("\n" + "─" * 60)
print("QUERY 1: Store Revenue Ranking (Window Functions)")
print("─" * 60)

query1 = """
WITH store_metrics AS (
    SELECT 
        store_id,
        ROUND(SUM(sales), 0) AS total_sales,
        ROUND(AVG(sales), 1) AS avg_daily_sales,
        COUNT(DISTINCT date) AS days_tracked,
        ROUND(AVG(CASE WHEN is_weekend = 1 THEN sales END), 1) AS avg_weekend_sales,
        ROUND(AVG(CASE WHEN is_weekend = 0 THEN sales END), 1) AS avg_weekday_sales
    FROM store_dept_daily
    GROUP BY store_id
)
SELECT 
    store_id,
    total_sales,
    avg_daily_sales,
    avg_weekend_sales,
    avg_weekday_sales,
    ROUND((avg_weekend_sales - avg_weekday_sales) / avg_weekday_sales * 100, 1) AS weekend_lift_pct,
    RANK() OVER (ORDER BY total_sales DESC) AS sales_rank
FROM store_metrics
ORDER BY sales_rank
"""

df1 = pd.read_sql_query(query1, conn)
print(df1.to_string(index=False))
df1.to_csv('outputs/sql_results/01_store_ranking.csv', index=False)
print("✓ Saved: outputs/sql_results/01_store_ranking.csv")

# ──────────────── QUERY 2: SNAP IMPACT ANALYSIS ────────────────

print("\n" + "─" * 60)
print("QUERY 2: SNAP Day Impact Analysis (CASE WHEN)")
print("─" * 60)

query2 = """
SELECT 
    store_id,
    ROUND(AVG(CASE WHEN snap_CA = 1 THEN sales END), 1) AS avg_snap_sales,
    ROUND(AVG(CASE WHEN snap_CA = 0 THEN sales END), 1) AS avg_nonsnap_sales,
    ROUND(
        (AVG(CASE WHEN snap_CA = 1 THEN sales END) - 
         AVG(CASE WHEN snap_CA = 0 THEN sales END)) / 
        AVG(CASE WHEN snap_CA = 0 THEN sales END) * 100, 1
    ) AS snap_lift_pct,
    SUM(CASE WHEN snap_CA = 1 THEN 1 ELSE 0 END) AS snap_days,
    SUM(CASE WHEN snap_CA = 0 THEN 1 ELSE 0 END) AS nonsnap_days
FROM store_dept_daily
GROUP BY store_id
ORDER BY snap_lift_pct DESC
"""

df2 = pd.read_sql_query(query2, conn)
print(df2.to_string(index=False))
df2.to_csv('outputs/sql_results/02_snap_impact.csv', index=False)
print("✓ Saved: outputs/sql_results/02_snap_impact.csv")

# ──────────────── QUERY 3: DEPARTMENT REVENUE CONCENTRATION ────────────────

print("\n" + "─" * 60)
print("QUERY 3: Department Revenue Concentration (NTILE)")
print("─" * 60)

query3 = """
WITH dept_revenue AS (
    SELECT 
        store_id,
        dept_id,
        ROUND(SUM(sales), 0) AS total_sales,
        ROUND(AVG(sales), 1) AS avg_daily_sales,
        NTILE(4) OVER (ORDER BY SUM(sales) DESC) AS revenue_quartile
    FROM store_dept_daily
    GROUP BY store_id, dept_id
)
SELECT 
    store_id,
    dept_id,
    total_sales,
    avg_daily_sales,
    revenue_quartile,
    CASE revenue_quartile
        WHEN 1 THEN 'Top 25% (High Revenue)'
        WHEN 2 THEN '25-50% (Above Average)'
        WHEN 3 THEN '50-75% (Below Average)'
        WHEN 4 THEN 'Bottom 25% (Low Revenue)'
    END AS revenue_tier
FROM dept_revenue
ORDER BY total_sales DESC
"""

df3 = pd.read_sql_query(query3, conn)
print(df3.to_string(index=False))
df3.to_csv('outputs/sql_results/03_dept_revenue_tiers.csv', index=False)
print("✓ Saved: outputs/sql_results/03_dept_revenue_tiers.csv")

# ──────────────── QUERY 4: MONTHLY TREND WITH GROWTH RATE ────────────────

print("\n" + "─" * 60)
print("QUERY 4: Monthly Sales Trend with Growth Rate (LAG)")
print("─" * 60)

query4 = """
WITH monthly AS (
    SELECT 
        SUBSTR(date, 1, 7) AS month,
        ROUND(SUM(sales), 0) AS total_sales,
        ROUND(AVG(sales), 1) AS avg_daily_sales,
        COUNT(DISTINCT date) AS trading_days
    FROM store_dept_daily
    GROUP BY SUBSTR(date, 1, 7)
),
with_lag AS (
    SELECT 
        month,
        total_sales,
        avg_daily_sales,
        trading_days,
        LAG(total_sales) OVER (ORDER BY month) AS prev_month_sales
    FROM monthly
)
SELECT 
    month,
    total_sales,
    avg_daily_sales,
    trading_days,
    prev_month_sales,
    CASE 
        WHEN prev_month_sales IS NOT NULL 
        THEN ROUND((total_sales - prev_month_sales) * 1.0 / prev_month_sales * 100, 1)
        ELSE NULL 
    END AS mom_growth_pct
FROM with_lag
ORDER BY month
"""

df4 = pd.read_sql_query(query4, conn)
print(df4.to_string(index=False))
df4.to_csv('outputs/sql_results/04_monthly_trends.csv', index=False)
print("✓ Saved: outputs/sql_results/04_monthly_trends.csv")

# ──────────────── QUERY 5: DAY-OF-WEEK ANALYSIS ────────────────

print("\n" + "─" * 60)
print("QUERY 5: Day-of-Week Demand Pattern")
print("─" * 60)

query5 = """
WITH daily_pattern AS (
    SELECT 
        day_of_week,
        CASE day_of_week
            WHEN 0 THEN 'Monday'
            WHEN 1 THEN 'Tuesday'
            WHEN 2 THEN 'Wednesday'
            WHEN 3 THEN 'Thursday'
            WHEN 4 THEN 'Friday'
            WHEN 5 THEN 'Saturday'
            WHEN 6 THEN 'Sunday'
        END AS day_name,
        ROUND(AVG(sales), 1) AS avg_sales,
        ROUND(MIN(sales), 1) AS min_sales,
        ROUND(MAX(sales), 1) AS max_sales
    FROM store_dept_daily
    GROUP BY day_of_week
),
overall AS (
    SELECT ROUND(AVG(sales), 1) AS overall_avg FROM store_dept_daily
)
SELECT 
    dp.day_name,
    dp.avg_sales,
    dp.min_sales,
    dp.max_sales,
    ROUND((dp.avg_sales - o.overall_avg) / o.overall_avg * 100, 1) AS vs_average_pct,
    CASE 
        WHEN dp.avg_sales > o.overall_avg * 1.1 THEN '↑ Increase Staff'
        WHEN dp.avg_sales < o.overall_avg * 0.9 THEN '↓ Reduce Staff'
        ELSE '→ Standard'
    END AS staffing_recommendation
FROM daily_pattern dp, overall o
ORDER BY dp.day_of_week
"""

df5 = pd.read_sql_query(query5, conn)
print(df5.to_string(index=False))
df5.to_csv('outputs/sql_results/05_dayofweek_pattern.csv', index=False)
print("✓ Saved: outputs/sql_results/05_dayofweek_pattern.csv")

# ──────────────── QUERY 6: HIGH-RISK STORE-DEPARTMENTS ────────────────

print("\n" + "─" * 60)
print("QUERY 6: High-Risk Store-Departments (Volatility)")
print("─" * 60)

query6 = """
WITH dept_stats AS (
    SELECT 
        store_id,
        dept_id,
        ROUND(AVG(sales), 1) AS avg_sales,
        ROUND(AVG(sales) + 2 * 
            (SUM(sales * sales) / COUNT(*) - AVG(sales) * AVG(sales)), 1
        ) AS upper_threshold,
        COUNT(*) AS total_days,
        SUM(CASE WHEN is_weekend = 1 THEN sales ELSE 0 END) AS weekend_total,
        SUM(CASE WHEN snap_CA = 1 THEN sales ELSE 0 END) AS snap_total
    FROM store_dept_daily
    GROUP BY store_id, dept_id
)
SELECT 
    store_id,
    dept_id,
    avg_sales,
    ROUND(weekend_total * 1.0 / total_days, 1) AS weekend_daily_avg,
    ROUND(snap_total * 1.0 / total_days, 1) AS snap_daily_avg,
    CASE 
        WHEN avg_sales > 800 THEN 'HIGH VOLUME'
        WHEN avg_sales > 400 THEN 'MEDIUM VOLUME'
        ELSE 'LOW VOLUME'
    END AS volume_tier,
    RANK() OVER (ORDER BY avg_sales DESC) AS volume_rank
FROM dept_stats
ORDER BY avg_sales DESC
"""

df6 = pd.read_sql_query(query6, conn)
print(df6.to_string(index=False))
df6.to_csv('outputs/sql_results/06_store_dept_risk.csv', index=False)
print("✓ Saved: outputs/sql_results/06_store_dept_risk.csv")

# ──────────────── SUMMARY ────────────────

conn.close()

print("\n" + "=" * 70)
print("SQL ANALYTICS COMPLETE!")
print("=" * 70)

print("\nQueries Executed:")
print("  1. Store Revenue Ranking        → Window functions (RANK)")
print("  2. SNAP Impact Analysis          → CASE WHEN aggregations")
print("  3. Dept Revenue Concentration    → NTILE quartiles")
print("  4. Monthly Trends + Growth       → LAG window function")
print("  5. Day-of-Week Demand Pattern    → CTE + CASE WHEN")
print("  6. Store-Dept Volume Tiers       → RANK + CASE WHEN")

print("\nAll results saved to: outputs/sql_results/")
print("=" * 70)

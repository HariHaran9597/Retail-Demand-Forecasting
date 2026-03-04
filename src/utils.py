"""
Utility functions for the forecasting project
"""
import pandas as pd
import numpy as np
from typing import List, Tuple


def calculate_zero_sales_pct(df: pd.DataFrame, day_cols: List[str]) -> pd.Series:
    """Calculate percentage of zero-sales days for each product"""
    return (df[day_cols] == 0).sum(axis=1) / len(day_cols) * 100


def get_date_range(calendar: pd.DataFrame) -> Tuple[str, str, int]:
    """Get start date, end date, and number of days"""
    return (
        calendar['date'].min(),
        calendar['date'].max(),
        calendar['date'].nunique()
    )


def calculate_snap_lift(df: pd.DataFrame, snap_col: str = 'snap_CA') -> float:
    """Calculate percentage lift in sales on SNAP days"""
    snap_avg = df[df[snap_col] == 1]['sales'].mean()
    non_snap_avg = df[df[snap_col] == 0]['sales'].mean()
    return ((snap_avg / non_snap_avg) - 1) * 100


def get_top_events(df: pd.DataFrame, event_col: str = 'event_name_1', top_n: int = 10) -> pd.Series:
    """Get top N events by average sales impact"""
    return df.groupby(event_col)['sales'].mean().sort_values(ascending=False).head(top_n)


def create_train_test_split(df: pd.DataFrame, train_days: int = 1800) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets based on day number"""
    df['day_num'] = df['d'].str.replace('d_', '').astype(int)
    train = df[df['day_num'] <= train_days].copy()
    test = df[df['day_num'] > train_days].copy()
    return train, test

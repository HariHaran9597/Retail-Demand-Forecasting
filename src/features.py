"""
Feature engineering utilities for demand forecasting
"""
import pandas as pd
import numpy as np
from typing import List


class FeatureEngineer:
    """Create time series features for forecasting"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def add_lag_features(self, lags: List[int] = [7, 14, 28]) -> pd.DataFrame:
        """Add lag features for sales"""
        print(f"Adding lag features: {lags}")
        
        for lag in lags:
            self.df[f'sales_lag_{lag}'] = self.df.groupby(['store_id', 'item_id'])['sales'].shift(lag)
        
        return self.df
    
    def add_rolling_features(self, windows: List[int] = [7, 28]) -> pd.DataFrame:
        """Add rolling mean and std features (leak-proof: excludes current day)"""
        print(f"Adding rolling features with windows: {windows} (shifted to prevent leakage)")
        
        for window in windows:
            # Rolling mean (shifted by 1 to exclude current day)
            self.df[f'sales_rolling_mean_{window}'] = (
                self.df.groupby(['store_id', 'item_id'])['sales']
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
            
            # Rolling std (shifted by 1 to exclude current day)
            self.df[f'sales_rolling_std_{window}'] = (
                self.df.groupby(['store_id', 'item_id'])['sales']
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
            )
        
        return self.df
    
    def add_calendar_features(self) -> pd.DataFrame:
        """Add calendar-based features"""
        print("Adding calendar features")
        
        # Convert date to datetime
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Day of week (already have weekday, but add numeric)
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        
        # Week of month
        self.df['week_of_month'] = (self.df['date'].dt.day - 1) // 7 + 1
        
        # Is weekend
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        # Month and quarter (already have month, add quarter)
        self.df['quarter'] = self.df['date'].dt.quarter
        
        return self.df
    
    def add_event_features(self) -> pd.DataFrame:
        """Add event-related features"""
        print("Adding event features")
        
        # Has any event
        self.df['has_event'] = (~self.df['event_name_1'].isna()).astype(int)
        
        # Event type flags
        self.df['is_sporting'] = (self.df['event_type_1'] == 'Sporting').astype(int)
        self.df['is_cultural'] = (self.df['event_type_1'] == 'Cultural').astype(int)
        self.df['is_national'] = (self.df['event_type_1'] == 'National').astype(int)
        self.df['is_religious'] = (self.df['event_type_1'] == 'Religious').astype(int)
        
        return self.df
    
    def add_price_features(self) -> pd.DataFrame:
        """Add price-related features"""
        print("Adding price features")
        
        # Fill missing prices with forward fill then backward fill
        self.df['sell_price'] = self.df.groupby(['store_id', 'item_id'])['sell_price'].ffill().bfill()
        
        # Price change from previous week
        self.df['price_change'] = (
            self.df.groupby(['store_id', 'item_id'])['sell_price'].diff()
        )
        
        # Is on promotion (price decreased)
        self.df['is_promotion'] = (self.df['price_change'] < 0).astype(int)
        
        # Price relative to item average
        self.df['price_vs_avg'] = (
            self.df.groupby('item_id')['sell_price']
            .transform(lambda x: x / x.mean() if x.mean() > 0 else 1)
        )
        
        # Price momentum (lag of price change)
        self.df['price_momentum'] = (
            self.df.groupby(['store_id', 'item_id'])['price_change'].shift(1)
        )
        
        return self.df
    
    def add_aggregated_features(self) -> pd.DataFrame:
        """Add store and department level aggregations"""
        print("Adding aggregated features")
        
        # Store-day total sales
        store_day_sales = self.df.groupby(['store_id', 'date'])['sales'].transform('sum')
        self.df['store_day_total'] = store_day_sales
        
        # Department-day total sales
        dept_day_sales = self.df.groupby(['dept_id', 'date'])['sales'].transform('sum')
        self.df['dept_day_total'] = dept_day_sales
        
        return self.df
    
    def create_all_features(self) -> pd.DataFrame:
        """Create all features in sequence"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        self.add_calendar_features()
        self.add_event_features()
        self.add_price_features()
        self.add_lag_features()
        self.add_rolling_features()
        self.add_aggregated_features()
        
        print("\n✓ All features created")
        print(f"Final shape: {self.df.shape}")
        print(f"Total features: {len(self.df.columns)}")
        
        return self.df
    
    def get_feature_names(self) -> List[str]:
        """Get list of engineered feature names"""
        base_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
                     'd', 'date', 'sales', 'wm_yr_wk', 'weekday', 'wday', 'month', 
                     'year', 'event_name_1', 'event_type_1', 'event_name_2', 
                     'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI', 'sell_price']
        
        return [col for col in self.df.columns if col not in base_cols]

"""
Data loading and SQL query utilities for M5 dataset
"""
import pandas as pd
import sqlite3
from pathlib import Path
from typing import Tuple, Optional


class M5DataLoader:
    """Handle M5 dataset loading and SQL operations"""
    
    def __init__(self, data_dir: str = 'data/raw', db_path: str = 'data/processed/m5_data.db'):
        self.data_dir = Path(data_dir)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the three core M5 files"""
        sales = pd.read_csv(self.data_dir / 'sales_train_validation.csv')
        calendar = pd.read_csv(self.data_dir / 'calendar.csv')
        prices = pd.read_csv(self.data_dir / 'sell_prices.csv')
        return sales, calendar, prices
    
    def create_database(self, sales: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
        """Load dataframes into SQLite database"""
        conn = sqlite3.connect(self.db_path)
        sales.to_sql('sales', conn, if_exists='replace', index=False)
        calendar.to_sql('calendar', conn, if_exists='replace', index=False)
        prices.to_sql('prices', conn, if_exists='replace', index=False)
        conn.close()
        print(f"✓ Database created at {self.db_path}")
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return results"""
        conn = sqlite3.connect(self.db_path)
        result = pd.read_sql_query(sql, conn)
        conn.close()
        return result
    
    def get_hierarchy_stats(self) -> pd.DataFrame:
        """Get counts of hierarchy levels"""
        query = """
        SELECT 
            COUNT(DISTINCT state_id) as num_states,
            COUNT(DISTINCT store_id) as num_stores,
            COUNT(DISTINCT cat_id) as num_categories,
            COUNT(DISTINCT dept_id) as num_departments,
            COUNT(DISTINCT item_id) as num_items
        FROM sales
        """
        return self.query(query)
    
    def filter_subset(self, sales: pd.DataFrame, state: str = 'CA', category: str = 'FOODS') -> pd.DataFrame:
        """Filter to specific state and category"""
        return sales[(sales['state_id'] == state) & (sales['cat_id'] == category)].copy()
    
    def melt_sales(self, sales: pd.DataFrame) -> pd.DataFrame:
        """Transform sales from wide to long format"""
        id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        day_cols = [col for col in sales.columns if col.startswith('d_')]
        
        return sales.melt(
            id_vars=id_cols,
            value_vars=day_cols,
            var_name='d',
            value_name='sales'
        )
    
    def merge_all(self, sales_long: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """Merge sales with calendar and prices"""
        # Merge with calendar
        merged = sales_long.merge(calendar, on='d', how='left')
        
        # Merge with prices
        merged = merged.merge(
            prices,
            on=['store_id', 'item_id', 'wm_yr_wk'],
            how='left'
        )
        
        return merged

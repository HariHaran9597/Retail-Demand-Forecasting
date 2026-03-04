"""
Simple test to verify Streamlit and data loading
"""
import streamlit as st
import pandas as pd
from pathlib import Path

st.title("🏪 Retail Demand Forecasting - Test")

st.write("Testing data loading...")

# Try to load data
try:
    # Get the correct path
    app_dir = Path(__file__).parent
    data_path = app_dir.parent / 'data' / 'processed' / 'ca_foods_store_dept_agg.parquet'
    
    st.write(f"App directory: {app_dir}")
    st.write(f"Looking for data at: {data_path}")
    st.write(f"File exists: {data_path.exists()}")
    
    if data_path.exists():
        df = pd.read_parquet(data_path)
        st.success(f"✅ Data loaded successfully!")
        st.write(f"Shape: {df.shape}")
        st.write(f"Columns: {list(df.columns)}")
        st.dataframe(df.head())
    else:
        st.error("❌ Data file not found!")
        st.write("Checking parent directory...")
        st.write(f"Parent exists: {data_path.parent.exists()}")
        if data_path.parent.exists():
            st.write(f"Files in directory: {list(data_path.parent.glob('*'))}")
        
except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())

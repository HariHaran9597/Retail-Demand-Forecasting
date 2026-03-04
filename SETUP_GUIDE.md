# Setup Guide - Retail Demand Forecasting Project

## Prerequisites
- Python 3.9 or higher
- Kaggle account (to download M5 dataset)
- 4GB+ RAM recommended
- 2GB+ free disk space

## Step-by-Step Setup

### 1. Clone or Download Project
```bash
# If using git
git clone <your-repo-url>
cd retail-demand-forecasting

# Or download and extract ZIP
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download M5 Dataset

#### Option A: Using Kaggle API (Recommended)
```bash
# Install kaggle CLI
pip install kaggle

# Set up Kaggle credentials
# 1. Go to https://www.kaggle.com/account
# 2. Click "Create New API Token"
# 3. Place kaggle.json in:
#    Windows: C:\Users\<username>\.kaggle\kaggle.json
#    Mac/Linux: ~/.kaggle/kaggle.json

# Download dataset
kaggle competitions download -c m5-forecasting-accuracy
```

#### Option B: Manual Download
1. Go to https://www.kaggle.com/c/m5-forecasting-accuracy/data
2. Download these files:
   - sales_train_validation.csv
   - calendar.csv
   - sell_prices.csv
3. Place them in `data/raw/` folder

### 5. Create Required Directories
```bash
mkdir -p data/raw data/processed outputs/models outputs/plots
```

### 6. Verify Setup
```bash
# Start Jupyter
jupyter notebook

# Open notebooks/day1_data_understanding.ipynb
# Run the first cell to verify all imports work
```

## Troubleshooting

### Prophet Installation Issues
If Prophet fails to install:
```bash
# Windows
pip install pystan==2.19.1.1
pip install prophet

# Mac with M1/M2
conda install -c conda-forge prophet
```

### Memory Issues
If you run out of memory:
- Reduce the subset size in notebooks
- Use `chunksize` parameter when reading CSVs
- Close other applications

### SQLite Issues
If database creation fails:
- Ensure `data/processed/` directory exists
- Check write permissions
- Try deleting existing .db file and recreating

## Next Steps
Once setup is complete:
1. Open `notebooks/day1_data_understanding.ipynb`
2. Follow the notebook instructions
3. Proceed through Day 1 → Day 6

## Getting Help
- Check notebook markdown cells for detailed instructions
- Review error messages carefully
- Ensure all data files are in correct locations

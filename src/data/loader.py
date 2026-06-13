import pandas as pd
import yfinance as yf
from datetime import datetime

def load_wti_data(start_date="2010-01-01", end_date=None):
    """Load WTI Crude Oil prices"""
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    # WTI via Yahoo Finance (or FRED alternative)
    wti = yf.download("CL=F", start=start_date, end=end_date, progress=False)
    wti = wti[['Close']].rename(columns={'Close': 'wti_price'})
    wti = wti.resample('D').ffill()  # forward fill weekends/holidays
    return wti

def load_global_production():
    """Placeholder for global oil production data (monthly -> daily)"""
    # In next step we'll add real data source (Our World in Data / EIA)
    dates = pd.date_range(start='2010-01-01', end=datetime.today().strftime('%Y-%m-%d'), freq='D')
    df = pd.DataFrame(index=dates)
    df['global_production'] = 95000  # placeholder in kb/d - we'll replace with real data
    return df

def load_merged_data():
    """Merge WTI + Global Production"""
    wti = load_wti_data()
    prod = load_global_production()
    df = wti.join(prod, how='left')
    df = df.ffill()
    return df

if __name__ == "__main__":
    df = load_merged_data()
    print(f"Data loaded: {df.shape[0]} days")
    print(df.tail())
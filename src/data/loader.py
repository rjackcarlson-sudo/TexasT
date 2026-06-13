import pandas as pd
import yfinance as yf
from datetime import datetime

def load_wti_data(start_date="2010-01-01", end_date=None):
    """Load daily WTI Crude Oil prices"""
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    wti = yf.download("CL=F", start=start_date, end=end_date, progress=False)
    wti = wti[['Close']].rename(columns={'Close': 'wti_price'})
    wti = wti.resample('D').ffill()
    return wti

def load_global_production():
    """Placeholder — we'll replace with real continent-level data (EIA/OWID CSV)"""
    dates = pd.date_range(start='2010-01-01', end=datetime.today().strftime('%Y-%m-%d'), freq='D')
    df = pd.DataFrame(index=dates)
    df['global_production_mbpd'] = 95000  # kb/d placeholder
    df['us_production_mbpd'] = 13000      # US share placeholder
    df['non_us_production_mbpd'] = df['global_production_mbpd'] - df['us_production_mbpd']
    return df

def load_merged_data():
    """Merge WTI + production (expandable to consumption, storage, etc.)"""
    wti = load_wti_data()
    prod = load_global_production()
    df = wti.join(prod, how='left').ffill()
    return df
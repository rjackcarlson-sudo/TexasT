import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-informed features including user's requested signals"""
    df = df.copy()
    
    # === Technical / Price Features ===
    for lag in [1, 2, 3, 5, 7, 14, 30]:
        df[f'lag_{lag}'] = df['wti_price'].shift(lag)
    
    for window in [7, 14, 30]:
        df[f'roll_mean_{window}'] = df['wti_price'].rolling(window=window).mean()
        df[f'roll_std_{window}'] = df['wti_price'].rolling(window=window).std()
    
    df['momentum_7'] = df['wti_price'] - df['wti_price'].shift(7)
    df['pct_change_1'] = df['wti_price'].pct_change()
    
    # === Production by Region (Your Key Intuition) ===
    if 'us_production_mbpd' in df.columns:
        df['us_vs_non_us_ratio'] = df['us_production_mbpd'] / (df['non_us_production_mbpd'] + 1)
        df['us_prod_change_30'] = df['us_production_mbpd'].pct_change(30)
    
    # === Consumption, Storage, Tankers, Weather Placeholders ===
    # These will be populated from real CSVs / APIs in next iterations
    df['global_consumption_mbpd'] = 103000  # placeholder
    df['storage_utilization'] = 0.75        # global storage fill rate placeholder
    df['tanker_availability_index'] = 1.0   # lower = tighter shipping
    df['heating_demand_index'] = 1.0        # from major metro temps
    
    # Target
    df['target'] = df['wti_price'].shift(-1)
    df['target_direction'] = (df['target'] > df['wti_price']).astype(int)
    
    return df.dropna()
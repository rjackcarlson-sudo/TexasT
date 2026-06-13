import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from src.data.loader import load_merged_data
from src.features.engineer import engineer_features

def train_base_model(save_model=True):
    """Train XGBoost with your domain-informed features"""
    df_raw = load_merged_data()
    df = engineer_features(df_raw)
    
    # Use all engineered features (your real-world signals)
    exclude_cols = ['target', 'target_direction', 'wti_price']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target']
    
    # TimeSeries CV (critical — no future leakage)
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(
            objective='reg:absoluteerror',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        maes.append(mean_absolute_error(y_val, pred))
    
    print(f"Average MAE (TimeSeries CV): ${sum(maes)/len(maes):.2f}")
    
    # Final model
    final_model = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    final_model.fit(X, y)
    
    if save_model:
        joblib.dump(final_model, 'models/base_xgboost.pkl')
        joblib.dump(feature_cols, 'models/feature_cols.pkl')
        print("✅ Model saved to models/ folder")
    
    return final_model, feature_cols

if __name__ == "__main__":
    train_base_model()
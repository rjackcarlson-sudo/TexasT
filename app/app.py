import streamlit as st
import pandas as pd
import joblib
from src.data.loader import load_merged_data
from src.features.engineer import engineer_features
from src.models.train import train_base_model

st.set_page_config(page_title="TexasT — Oil Price Predictor", layout="wide")
st.title("🛢️ TexasT — WTI Oil Price Predictor")
st.markdown("**Hybrid ML + LLM | Domain-Driven Next-Day Forecast**")

# Data Pipeline
with st.spinner("Loading data & engineering features..."):
    df_raw = load_merged_data()
    df = engineer_features(df_raw)

st.subheader("Recent WTI Price Trend")
st.line_chart(df['wti_price'].tail(180))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Latest WTI", f"${df['wti_price'].iloc[-1]:.2f}")
with col2:
    st.metric("7d Volatility", f"{df['roll_std_7'].iloc[-1]:.2f}")
with col3:
    st.metric("Engineered Features", len(df.columns)-3)

if st.button("🚀 Train Base Model"):
    with st.spinner("Training XGBoost with your domain features..."):
        model, features = train_base_model()
    st.success("✅ Base model trained successfully!")

st.info("""**Your real-world intuition built in**:
- US vs Non-US production split
- Consumption, storage, tanker availability, weather demand proxies
- More real EIA/IEA data loaders coming next""")

if st.button("Show Feature List"):
    st.write(df.columns.tolist())
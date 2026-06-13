import streamlit as st
import pandas as pd
import joblib
from src.data.loader import load_merged_data
from src.features.engineer import engineer_features
from src.models.train import train_base_model
from src.llm.analyzer import generate_llm_analysis

st.set_page_config(page_title="TexasT — Oil Price Predictor", layout="wide")
st.title("🛢️ TexasT — WTI Oil Price Predictor")
st.markdown("**Hybrid ML + LLM | Domain-Driven Next-Day Forecast**")

# Load data
with st.spinner("Loading data & features..."):
    df_raw = load_merged_data()
    df = engineer_features(df_raw)

st.subheader("Recent WTI Trend")
st.line_chart(df['wti_price'].tail(180))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Latest WTI", f"${df['wti_price'].iloc[-1]:.2f}")
with col2:
    st.metric("7d Volatility", f"{df['roll_std_7'].iloc[-1]:.2f}")
with col3:
    st.metric("Features", len(df.columns)-3)

if st.button("🚀 Train / Load Base Model & Predict"):
    with st.spinner("Training & generating prediction..."):
        model, feature_cols = train_base_model(save_model=True)
        
        # Latest prediction
        latest_features = df[feature_cols].iloc[-1:].copy()
        prediction = model.predict(latest_features)[0]
        current_price = df['wti_price'].iloc[-1]
        
        # LLM analysis
        analysis = generate_llm_analysis(prediction, current_price, latest_features.iloc[0].to_dict())
    
    st.success(f"**Tomorrow's WTI Forecast: ${prediction:.2f}**")
    st.markdown(analysis)

st.info("""**Your domain intuition is now in the model**:
- US/Non-US production split
- Consumption, storage, tanker, weather proxies
- Next: Real EIA data loaders + full Grok LLM integration""")
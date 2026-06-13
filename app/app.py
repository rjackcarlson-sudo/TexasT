import streamlit as st
import pandas as pd
from src.data.loader import load_merged_data

st.set_page_config(page_title="TexasT — Oil Price Predictor", layout="wide")
st.title("🛢️ TexasT — WTI Oil Price Predictor")
st.markdown("Hybrid ML + LLM system for next-day WTI Crude Oil forecasting")

# Load data
with st.spinner("Loading data..."):
    df = load_merged_data()

st.subheader("Recent WTI Prices")
st.line_chart(df['wti_price'].tail(180))

st.info("Next: We'll add feature engineering, model training, and LLM analysis layer.")

if st.button("Run Basic Prediction (Placeholder)"):
    st.success("Base model prediction coming in next iteration!")
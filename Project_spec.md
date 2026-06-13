# Oil Price Predictor Project Specification

**Project Goal**  
Build a professional-grade, interactive web application that predicts next-day WTI Crude Oil prices using a **hybrid ML + LLM system**.  
- Base ML model handles quantifiable signals (price history, global production, engineered features).  
- LLM layer provides contextual analysis of hard-to-quantify world events (geopolitics, OPEC decisions, demand shocks, etc.).  
- Focused on **one specific task**: Next-day WTI price forecast.  
- Designed as a portfolio piece demonstrating full ML engineering practices.

## Core Requirements
- Clean, modular repo structure (professional ML engineering standard).
- Reproducible data pipeline with global oil production integrated.
- Strong emphasis on time-series best practices (no data leakage, proper validation).
- Interactive Streamlit web app for users to input data and receive predictions + LLM analysis.
- Deployable (Streamlit Cloud or similar) and embeddable on a personal website.

## Data Sources
- **WTI Crude Oil Daily Prices**: FRED (DCOILWTICO) or EIA public CSVs.
- **Global Oil Production**: Our World in Data (World aggregate, monthly, forward-filled to daily) or EIA/IEA sources (in Mb/d or equivalent).
- Future extensions: Natural gas prices, USD index, inventories, rig counts.

## Feature Strategy (Optimal 8–15 features)
- Price lags: lag_1, lag_7, lag_14.
- Rolling statistics: 7-day mean, 7-day std (volatility).
- Exogenous: Global Production (TWh or Mb/d, ffilled).
- Additional candidates: More lags, rolling windows, external indicators (selected via importance/correlation).
- Linear algebra relevance: Used in deep models (matrix ops in linear layers, attention), feature reduction (PCA/SVD), and underlying LLM transformer.

## Modeling Approach
- **Base Model**: XGBoost/LightGBM (tabular) or PyTorch neural net / LSTM / Transformer (to explicitly showcase linear algebra: matrix multiplication, gradients, attention mechanisms).
- Training: Time-series split / walk-forward validation.
- Metrics: MAE, RMSE, directional accuracy, backtesting.
- Artifacts: Saved model + feature list.

## Hybrid LLM Layer
- Base model outputs numerical prediction + confidence.
- LLM (Grok API, OpenAI, or local Ollama) receives: ML prediction, recent features, news/context summary.
- Output: Natural language analysis explaining world events, risks, upside/downside scenarios, and narrative forecast.

## Repo Structure (Professional Skeleton)
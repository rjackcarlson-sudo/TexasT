# Oil Price Predictor (TT App) - Project Specification

**Project Goal**  
Build a professional-grade hybrid ML + LLM web app for next-day WTI Crude Oil price prediction.

## Core Features
- Time-series ML base model (XGBoost / LightGBM / LSTM)
- Walk-forward validation + full evaluation suite
- LLM layer for contextual analysis (geopolitics, OPEC, etc.)
- Interactive Streamlit dashboard

## Repo Structure
- `PROJECT_SPEC.md`
- `src/data/` — data loading & merging (WTI + global production)
- `src/features/` — feature engineering
- `src/models/` — training + evaluation (walk-forward)
- `src/llm/` — LLM analysis
- `app/` — Streamlit interface
- `data/` & `models/` — gitignored

## Evaluation (Current Priority)
- Walk-forward validation
- Metrics: MAE, RMSE, MAPE, R², Directional Accuracy
- Visuals: Actual vs Predicted + residuals

**Status**: In active Grok build — June 2026

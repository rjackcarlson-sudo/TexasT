# TexasT — Oil Price Predictor

Professional hybrid ML + LLM system for next-day WTI Crude Oil price forecasting.

**Features**
- Time-series ML base model (XGBoost/LightGBM + engineered features)
- Global oil production integration
- LLM contextual analysis layer (geopolitics, news, OPEC, etc.)
- Interactive Streamlit web app

**Tech Stack**
- Python, Pandas, XGBoost/LightGBM
- Streamlit (frontend)
- Grok / OpenAI / Ollama (LLM layer)

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app/app.py
import os
from dotenv import load_dotenv

load_dotenv()

def generate_llm_analysis(prediction: float, current_price: float, features: dict):
    """Placeholder for LLM contextual analysis (Grok/OpenAI/Ollama)"""
    change = prediction - current_price
    direction = "UP" if change > 0 else "DOWN"
    
    analysis = f"""
**TexasT Hybrid Forecast Analysis**

**Base Model Prediction:** Tomorrow's WTI ≈ **${prediction:.2f}** ({direction} ${abs(change):.2f})

**Key Drivers (from your domain features):**
- US vs Non-US production split
- Global consumption signals
- Storage utilization & tanker availability
- Weather-driven demand in major metros

**LLM Interpretation:** The model sees moderate upward pressure likely driven by regional production dynamics and potential storage tightness. Watch OPEC+ decisions and geopolitical events in key producing regions for confirmation.
"""
    return analysis.strip()
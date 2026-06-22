import numpy as np
from xgboost import XGBRegressor
from src.models.evaluate import walk_forward_validation, plot_actual_vs_pred

# Dummy data for testing (replace later with real WTI data)
np.random.seed(42)
n_samples = 500
X = np.random.randn(n_samples, 10)  # 10 dummy features
y = np.random.randn(n_samples).cumsum() + 70  # fake price series around $70

print("Testing evaluation module...")

model = XGBRegressor(n_estimators=50, random_state=42)
metrics, results_df = walk_forward_validation(model, X, y, n_test=30)

print("\n=== Model Performance Metrics ===")
print(metrics)

# Plot (will show in Codespaces if possible, or save)
plot_actual_vs_pred(results_df)
print("\nEvaluation test complete!")

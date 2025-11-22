# survival-reasoning/survival_cox_vs_neural_net.py
# AI Tutor Project: Critiquing Statistical Reasoning in Survival Models
# Dataset: UCI Heart Disease (time-to-event simulation from clinical features)

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter  # pip install lifelines if needed (but use your env)
from  sklearn.neural_network import MLPRegressor  # For simple survival approx
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset (uses heart.csv or processed.cleveland.data)
try:
    df = pd.read_csv('heart.csv')  # Kaggle version
    # Map to survival sim: 'time' as simulated survival time, 'target' as event
    np.random.seed(42)
    df['time'] = 1000 * np.exp(-0.05 * df['age'] + 0.1 * df['target']) + np.random.exponential(500, len(df))
    df['event'] = np.random.binomial(1, 0.3 + 0.2 * df['target'], len(df))
except:
    # Fallback to raw UCI
    df = pd.read_csv('processed.cleveland.data', header=None, na_values='?')
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = df.dropna()
    # Simulate survival as above
    np.random.seed(42)
    df['time'] = 1000 * np.exp(-0.05 * df['age'] + 0.1 * df['target']) + np.random.exponential(500, len(df))
    df['event'] = np.random.binomial(1, 0.3 + 0.2 * df['target'], len(df))

# Features for modeling
features = ['age', 'cp', 'trestbps', 'chol']
X = df[features]
y_time = df['time']
y_event = df['event']

# 1. Cox PH Model (Traditional Biostats)
cph = CoxPHFitter()
cph.fit(df[['time', 'event'] + features], duration_col='time', event_col='event')
cph.print_summary()  # Statistical reasoning: Check HR, p-values, assumptions

# 2. Neural Net Approximation (AI/ML Extension)
X_train, X_test, y_train, y_test = train_test_split(X, y_time, test_size=0.2)
nn = MLPRegressor(hidden_layer_sizes=(50, 30), max_iter=500)
nn.fit(X_train, y_train)
preds = nn.predict(X_test)
mse = mean_squared_error(y_test, preds)

print(f"Cox PH: Concorde-Young C-index = {cph.concordance_index_:.3f}")
print(f"Neural Net MSE = {mse:.2f}")

# 3. AI Reasoning Critique (What the Tutor Role Does)
def critique_survival_reasoning(model_type, metrics):
    feedback = []
    if model_type == 'Cox':
        if cph.concordance_index_ < 0.7:
            feedback.append("Low C-index suggests poor discrimination; consider adding interactions (e.g., age*cp).")
        if any(p < 0.05 for p in cph.summary['p'][1:]):
            feedback.append("Significant predictors valid, but check proportional hazards assumption (Schoenfeld residuals).")
    elif model_type == 'Neural Net':
        if mse > 100:
            feedback.append("High MSE indicates overfitting; add regularization or use proper survival loss (e.g., negative log-likelihood).")
        feedback.append("NN lacks interpretability vs. Cox HRs; use SHAP for feature importance to bridge reasoning gap.")
    return {"issues": feedback, "recommendations": "Hybrid: Cox for inference, NN for prediction."}

print("\nAI Model Critique:")
print(critique_survival_reasoning('Cox', cph.concordance_index_))
print(critique_survival_reasoning('Neural Net', mse))

# Plot (save for GitHub)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
cph.plot()
plt.subplot(1, 2, 2)
plt.scatter(X_test['age'], preds, alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Predicted Survival Time')
plt.savefig('survival_comparison.png')
plt.show()
#%% [markdown]
# # Golden Baseline Comparison (Fairlearn)
# This script trains a standard, unconstrained model and a fairness-constrained model
# using Microsoft's `fairlearn` library to serve as "Golden Baselines".
# We will compare our MOO Pareto frontier against these baselines to prove the
# efficacy of our approach!

#%%
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import warnings

warnings.filterwarnings('ignore')

# Re-use our exact data loading and metric functions to ensure a 1:1 perfectly fair comparison
from moo_nsga2_pymoo import (
    load_and_preprocess_compas,
    compute_fairness_metrics,
    compute_model_complexity
)

#%% [markdown]
# ## Train Baselines
# 1. Unconstrained Logistic Regression
# 2. Fairlearn Exponentiated Gradient Logistic Regression

#%%
def main():
    print("=" * 70)
    print("  TRAINING GOLDEN BASELINES (FAIRLEARN)")
    print("=" * 70)

    # 1. Load exact same data
    X_train, y_train, X_test, y_test, sens_train, sens_test, _ = load_and_preprocess_compas('compas-scores-two-years.csv')

    baselines = []

    # --- MODEL 1: STANDARD UNCONSTRAINED MODEL ---
    print("\n🚀 Training Standard Unconstrained Logistic Regression...")
    clf_standard = LogisticRegression(random_state=42, max_iter=1000)
    clf_standard.fit(X_train, y_train)
    y_pred_std = clf_standard.predict(X_test)

    acc_std, dp_std, _ = compute_fairness_metrics(y_test, y_pred_std, sens_test)
    comp_std = compute_model_complexity('LogisticRegression', 0, 0, 0, 1.0) # Default L2=1.0

    baselines.append({
        'model_name': 'Standard LR (Unconstrained)',
        'balanced_accuracy': acc_std,
        'dp_violation': dp_std,
        'model_complexity': comp_std
    })
    print(f"   Accuracy: {acc_std:.4f} | DP Viol: {dp_std:.4f} | Comp: {comp_std:.4f}")

    # --- MODEL 2: FAIRLEARN GOLDEN MODEL ---
    print("\n⚖️ Training Fairlearn Model (Demographic Parity Constraint)...")
    
    # Exponentiated Gradient Reduction is a state-of-the-art fairness algorithm
    constraint = DemographicParity()
    clf_fair = ExponentiatedGradient(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        constraints=constraint,
        eps=0.01 # Allow maximum 1% DP violation
    )
    
    # Fairlearn requires the sensitive features during training to enforce the constraint
    clf_fair.fit(X_train, y_train, sensitive_features=sens_train)
    y_pred_fair = clf_fair.predict(X_test)

    acc_fair, dp_fair, _ = compute_fairness_metrics(y_test, y_pred_fair, sens_test)
    comp_fair = compute_model_complexity('LogisticRegression', 0, 0, 0, 1.0)

    baselines.append({
        'model_name': 'Fairlearn LR (Golden Model)',
        'balanced_accuracy': acc_fair,
        'dp_violation': dp_fair,
        'model_complexity': comp_fair
    })
    print(f"   Accuracy: {acc_fair:.4f} | DP Viol: {dp_fair:.4f} | Comp: {comp_fair:.4f}")

    # --- SAVE RESULTS ---
    os.makedirs('results', exist_ok=True)
    df_base = pd.DataFrame(baselines)
    df_base.to_csv('results/golden_baselines.csv', index=False)
    
    print("\n✅ Baselines saved to results/golden_baselines.csv")
    print("   We can now overlay these onto our 2D Pareto graphs for qualitative analysis!")

#%%
if __name__ == '__main__':
    main()

"""
Multi-Objective Optimization for Fairness in ML — Optuna Comparison
====================================================================
Comparison Algorithm: Optuna with NSGAIISampler (native multi-objective)
Dataset: COMPAS Recidivism

Same 3 objectives as NSGA-II:
  1. Minimize (1 - balanced_accuracy)
  2. Minimize DP violation
  3. Minimize EO violation

This script provides a direct comparison against the pymoo NSGA-II results.
"""

import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Re-use data loader and fairness metrics from the main script
from moo_nsga2_pymoo import (
    load_and_preprocess_compas,
    compute_fairness_metrics,
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# 1. OPTUNA OBJECTIVE (MULTI-OBJECTIVE)
# ──────────────────────────────────────────────────────────────────────────────

def optuna_objective(trial, X_train, y_train, X_val, y_val, sensitive_val):
    """
    Optuna objective returning 3 values for native multi-objective optimization.
    """
    # ── Decision variables ──
    model_type = trial.suggest_categorical('model_type', ['LogisticRegression', 'RandomForest', 'XGBoost'])
    regularization = trial.suggest_float('regularization', 1e-4, 10.0, log=True)
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 2, 15)
    class_weight_ratio = trial.suggest_float('class_weight_ratio', 1.0, 5.0)
    threshold = trial.suggest_float('threshold', 0.3, 0.7)
    learning_rate = trial.suggest_float('learning_rate', 0.005, 0.3, log=True)

    # ── Build model ──
    sample_weight = np.ones(len(y_train))
    sample_weight[y_train == 1] = class_weight_ratio

    try:
        if model_type == 'LogisticRegression':
            model = LogisticRegression(
                C=1.0 / regularization,
                max_iter=500,
                solver='lbfgs',
                random_state=42
            )
            model.fit(X_train, y_train, sample_weight=sample_weight)

        elif model_type == 'RandomForest':
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=5,
                class_weight={0: 1.0, 1: class_weight_ratio},
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

        else:  # XGBoost
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                reg_lambda=regularization,
                scale_pos_weight=class_weight_ratio,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
                verbosity=0
            )
            model.fit(X_train, y_train, verbose=False)

        # ── Predict and evaluate ──
        proba = model.predict_proba(X_val)[:, 1]
        y_pred = (proba >= threshold).astype(int)

        bal_acc, dp_viol, eo_viol = compute_fairness_metrics(
            y_val, y_pred, sensitive_val
        )

        return 1.0 - bal_acc, dp_viol, eo_viol

    except Exception:
        return 1.0, 1.0, 1.0


# ──────────────────────────────────────────────────────────────────────────────
# 2. EVALUATE PARETO FRONT ON TEST SET
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_optuna_pareto(best_trials, X_train_full, y_train_full,
                            X_test, y_test, sensitive_test):
    """Re-train each Pareto-optimal trial on full training data and test."""
    results = []

    for i, trial in enumerate(best_trials):
        params = trial.params
        model_type = params['model_type']
        reg = params['regularization']
        n_est = params['n_estimators']
        depth = params['max_depth']
        cw = params['class_weight_ratio']
        threshold = params['threshold']
        lr = params['learning_rate']

        sample_weight = np.ones(len(y_train_full))
        sample_weight[y_train_full == 1] = cw

        try:
            if model_type == 'LogisticRegression':
                model = LogisticRegression(
                    C=1.0 / reg, max_iter=500, solver='lbfgs', random_state=42
                )
                model.fit(X_train_full, y_train_full, sample_weight=sample_weight)
            elif model_type == 'RandomForest':
                model = RandomForestClassifier(
                    n_estimators=n_est, max_depth=depth, min_samples_leaf=5,
                    class_weight={0: 1.0, 1: cw}, random_state=42, n_jobs=-1
                )
                model.fit(X_train_full, y_train_full)
            else:
                model = xgb.XGBClassifier(
                    n_estimators=n_est, max_depth=depth, learning_rate=lr,
                    reg_lambda=reg, scale_pos_weight=cw, subsample=0.8,
                    colsample_bytree=0.8, objective='binary:logistic',
                    eval_metric='logloss', use_label_encoder=False,
                    random_state=42, verbosity=0
                )
                model.fit(X_train_full, y_train_full, verbose=False)

            proba = model.predict_proba(X_test)[:, 1]
            y_pred = (proba >= threshold).astype(int)

            bal_acc, dp_viol, eo_viol = compute_fairness_metrics(
                y_test, y_pred, sensitive_test
            )
        except Exception:
            bal_acc, dp_viol, eo_viol = 0.0, 1.0, 1.0

        results.append({
            'solution_id': i + 1,
            'model_type': model_type,
            'regularization': reg,
            'learning_rate': lr,
            'n_estimators': n_est,
            'max_depth': depth,
            'class_weight_ratio': cw,
            'threshold': threshold,
            'balanced_accuracy': bal_acc,
            'dp_violation': dp_viol,
            'eo_violation': eo_viol,
            'demographic_parity': 1.0 - dp_viol,
            'equalized_odds': 1.0 - eo_viol,
            'obj1_min': 1.0 - bal_acc,
            'obj2_min': dp_viol,
            'obj3_min': eo_viol,
        })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# 3. MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  MULTI-OBJECTIVE OPTIMIZATION FOR FAIRNESS IN ML")
    print("  Algorithm: Optuna NSGA-II (Comparison)")
    print("  Dataset: COMPAS Recidivism")
    print("=" * 70)

    # ── Load data ──
    data = load_and_preprocess_compas()
    X_train, y_train, X_test, y_test, sens_train, sens_test, preprocessor = data

    # ── Train/val split ──
    X_tr, X_val, y_tr, y_val, sens_tr_idx, sens_val_idx = train_test_split(
        X_train, y_train, range(len(y_train)),
        test_size=0.25, random_state=42, stratify=y_train
    )
    sens_val = sens_train.iloc[sens_val_idx].reset_index(drop=True)
    y_tr = y_tr.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    # ── Create multi-objective study ──
    N_TRIALS = 400

    sampler = optuna.samplers.NSGAIISampler(seed=42)
    study = optuna.create_study(
        directions=['minimize', 'minimize', 'minimize'],
        sampler=sampler,
        study_name='fairness_moo_optuna'
    )

    print(f"\n🚀 Starting Optuna NSGA-II optimization ({N_TRIALS} trials)...\n")
    start_time = time.time()

    study.optimize(
        lambda trial: optuna_objective(
            trial, X_tr, y_tr, X_val, y_val, sens_val
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    elapsed = time.time() - start_time
    print(f"\n✅ Optuna completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Extract Pareto front ──
    best_trials = study.best_trials
    print(f"   Pareto-optimal trials: {len(best_trials)}")

    # ── Evaluate on test set ──
    print("\n📊 Evaluating Pareto front on test set...")
    pareto_df = evaluate_optuna_pareto(
        best_trials, X_train, y_train, X_test, y_test, sens_test
    )

    # ── Display ──
    print("\n" + "=" * 70)
    print("  PARETO-OPTIMAL SOLUTIONS — Optuna NSGA-II (Test Set)")
    print("=" * 70)

    display_cols = [
        'solution_id', 'model_type', 'balanced_accuracy',
        'demographic_parity', 'equalized_odds',
        'threshold', 'class_weight_ratio'
    ]
    print(pareto_df[display_cols].to_string(index=False, float_format='%.4f'))

    # ── Save ──
    os.makedirs('results', exist_ok=True)
    pareto_df.to_csv('results/optuna_pareto_front.csv', index=False)
    print(f"\n💾 Saved to results/optuna_pareto_front.csv")

    # Save trial history for visualization
    trial_history = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            trial_history.append({
                'trial': t.number,
                'obj1': t.values[0],
                'obj2': t.values[1],
                'obj3': t.values[2],
                **t.params
            })

    history_df = pd.DataFrame(trial_history)
    history_data = {
        'pareto_df': pareto_df,
        'trial_history': history_df,
        'elapsed_time': elapsed,
        'n_trials': N_TRIALS,
        'algorithm': 'Optuna-NSGA-II',
    }

    with open('results/optuna_results.pkl', 'wb') as f:
        pickle.dump(history_data, f)
    print(f"💾 Full results saved to results/optuna_results.pkl")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY STATISTICS")
    print("=" * 70)
    print(f"  Balanced Accuracy range: [{pareto_df['balanced_accuracy'].min():.4f}, {pareto_df['balanced_accuracy'].max():.4f}]")
    print(f"  DP Violation range:      [{pareto_df['dp_violation'].min():.4f}, {pareto_df['dp_violation'].max():.4f}]")
    print(f"  EO Violation range:      [{pareto_df['eo_violation'].min():.4f}, {pareto_df['eo_violation'].max():.4f}]")

    return pareto_df, history_data


if __name__ == '__main__':
    pareto_df, history_data = main()

"""
Multi-Objective Optimization for Fairness in Machine Learning
=============================================================
Primary Algorithm: NSGA-II via pymoo
Dataset: COMPAS Recidivism (compas-scores-two-years.csv)

3 Objectives (all minimized internally, reported as maximization):
  1. Maximize Balanced Accuracy  → minimize (1 - balanced_accuracy)
  2. Maximize Demographic Parity → minimize |P(Ŷ=1|AA) - P(Ŷ=1|C)|
  3. Maximize Equalized Odds     → minimize max(|TPR_AA - TPR_C|, |FPR_AA - FPR_C|)

Decision Variables:
  - model_type: {0: LogisticRegression, 1: RandomForest, 2: XGBoost}
  - regularization: [1e-4, 10]
  - learning_rate: [0.005, 0.3]  (used for XGBoost)
  - n_estimators: [50, 500]
  - max_depth: [2, 15]
  - class_weight_ratio: [1.0, 5.0]
  - threshold: [0.3, 0.7]

Authors: Niranjan Gopal, Divyam Sareen
"""

import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import xgboost as xgb

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def load_and_preprocess_compas(data_path='compas-scores-two-years.csv'):
    """
    Load the COMPAS dataset and apply standard ProPublica filtering.
    Returns processed train/test splits along with sensitive attribute series.
    """
    df = pd.read_csv(data_path)

    # ProPublica's standard filtering criteria
    df = df[
        (df['days_b_screening_arrest'] <= 30) &
        (df['days_b_screening_arrest'] >= -30) &
        (df['is_recid'] != -1) &
        (df['c_charge_degree'] != 'O') &
        (df['score_text'] != 'N/A')
    ].copy()

    # Focus on African-American vs Caucasian for binary fairness analysis
    df = df[(df['race'] == 'African-American') | (df['race'] == 'Caucasian')].copy()

    # Feature selection
    feature_cols = [
        'sex', 'age', 'age_cat', 'race',
        'juv_fel_count', 'juv_misd_count', 'juv_other_count',
        'priors_count', 'c_charge_degree', 'days_b_screening_arrest'
    ]
    target_col = 'two_year_recid'

    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].fillna(
        df['days_b_screening_arrest'].median()
    )

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Identify column types
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # Ensure 'race' is treated as categorical
    if 'race' in numerical_features:
        numerical_features.remove('race')
    if 'race' not in categorical_features:
        categorical_features.append('race')
    X['race'] = X['race'].astype('category')

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Store sensitive attribute before encoding
    sensitive_train = X_train['race'].copy().reset_index(drop=True)
    sensitive_test = X_test['race'].copy().reset_index(drop=True)

    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print(f"Dataset loaded: {len(df)} records after filtering")
    print(f"  Train: {X_train_processed.shape[0]}, Test: {X_test_processed.shape[0]}")
    print(f"  Race distribution (train): {sensitive_train.value_counts().to_dict()}")
    print(f"  Recidivism rate: {y.mean():.3f}")

    return (X_train_processed, y_train, X_test_processed, y_test,
            sensitive_train, sensitive_test, preprocessor)


# ──────────────────────────────────────────────────────────────────────────────
# 2. FAIRNESS METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_fairness_metrics(y_true, y_pred, sensitive,
                              privileged='Caucasian',
                              unprivileged='African-American'):
    """
    Compute all three objectives given true labels, predictions, and sensitive attribute.

    Returns:
        balanced_acc: Balanced accuracy (higher is better)
        dp_violation: Demographic parity violation |P(Ŷ=1|AA) - P(Ŷ=1|C)| (lower is better)
        eo_violation: Equalized odds violation max(|ΔTPR|, |ΔFPR|) (lower is better)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sensitive = np.array(sensitive)

    # --- Objective 1: Balanced Accuracy ---
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # --- Masks for protected groups ---
    mask_priv = (sensitive == privileged)
    mask_unpriv = (sensitive == unprivileged)

    # --- Objective 2: Demographic Parity Violation ---
    ppr_priv = y_pred[mask_priv].mean() if mask_priv.sum() > 0 else 0.0
    ppr_unpriv = y_pred[mask_unpriv].mean() if mask_unpriv.sum() > 0 else 0.0
    dp_violation = abs(ppr_unpriv - ppr_priv)

    # --- Objective 3: Equalized Odds Violation ---
    def group_rates(mask):
        yt = y_true[mask]
        yp = y_pred[mask]
        # TPR = TP / (TP + FN)
        pos = (yt == 1)
        neg = (yt == 0)
        tpr = yp[pos].mean() if pos.sum() > 0 else 0.0
        fpr = yp[neg].mean() if neg.sum() > 0 else 0.0
        return tpr, fpr

    tpr_priv, fpr_priv = group_rates(mask_priv)
    tpr_unpriv, fpr_unpriv = group_rates(mask_unpriv)

    eo_violation = max(abs(tpr_unpriv - tpr_priv), abs(fpr_unpriv - fpr_priv))

    return balanced_acc, dp_violation, eo_violation


# ──────────────────────────────────────────────────────────────────────────────
# 3. MODEL BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def build_and_evaluate(x_vec, X_train, y_train, X_val, y_val, sensitive_val):
    """
    Build a model from the decision-variable vector, train, predict, and
    return the 3 objective values (all to be minimized).

    Decision variable vector x_vec (7 elements):
      [0] model_type        ∈ [0, 2]  → int → {0:LR, 1:RF, 2:XGB}
      [1] regularization    ∈ [1e-4, 10]  (log scale internally)
      [2] learning_rate     ∈ [0.005, 0.3]
      [3] n_estimators      ∈ [50, 500] → int
      [4] max_depth         ∈ [2, 15]   → int
      [5] class_weight_ratio ∈ [1.0, 5.0]
      [6] threshold         ∈ [0.3, 0.7]
    """
    # Decode decision variables
    model_type = int(np.clip(np.round(x_vec[0]), 0, 2))
    reg_strength = float(np.clip(x_vec[1], 1e-4, 10.0))
    lr = float(np.clip(x_vec[2], 0.005, 0.3))
    n_est = int(np.clip(np.round(x_vec[3]), 50, 500))
    depth = int(np.clip(np.round(x_vec[4]), 2, 15))
    cw_ratio = float(np.clip(x_vec[5], 1.0, 5.0))
    threshold = float(np.clip(x_vec[6], 0.3, 0.7))

    # Build class weights dict
    # class 0 gets weight 1.0, class 1 gets cw_ratio
    sample_weight = np.ones(len(y_train))
    sample_weight[y_train == 1] = cw_ratio

    try:
        if model_type == 0:
            # Logistic Regression — C = 1/regularization
            model = LogisticRegression(
                C=1.0 / reg_strength,
                max_iter=500,
                solver='lbfgs',
                random_state=42
            )
            model.fit(X_train, y_train, sample_weight=sample_weight)

        elif model_type == 1:
            # Random Forest
            model = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=depth,
                min_samples_leaf=5,
                class_weight={0: 1.0, 1: cw_ratio},
                random_state=42,
                n_jobs=2
            )
            model.fit(X_train, y_train)

        else:
            # XGBoost
            model = xgb.XGBClassifier(
                n_estimators=n_est,
                max_depth=depth,
                learning_rate=lr,
                reg_lambda=reg_strength,
                scale_pos_weight=cw_ratio,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
                verbosity=0
            )
            model.fit(X_train, y_train, verbose=False)

        # Predict probabilities and apply threshold
        proba = model.predict_proba(X_val)[:, 1]
        y_pred = (proba >= threshold).astype(int)

        # Compute objectives
        bal_acc, dp_viol, eo_viol = compute_fairness_metrics(
            y_val, y_pred, sensitive_val
        )

        # Return as minimization objectives: (1-bal_acc, dp_viol, eo_viol)
        return 1.0 - bal_acc, dp_viol, eo_viol, model, threshold

    except Exception as e:
        # Return worst-case objectives on failure
        return 1.0, 1.0, 1.0, None, threshold


# ──────────────────────────────────────────────────────────────────────────────
# 4. PYMOO PROBLEM DEFINITION
# ──────────────────────────────────────────────────────────────────────────────

class FairnessMOOProblem(Problem):
    """
    pymoo Problem with 7 decision variables and 3 objectives.
    All objectives are minimized.
    """

    def __init__(self, X_train, y_train, X_val, y_val, sensitive_val):
        # 7 decision variables
        xl = np.array([0.0, 1e-4, 0.005, 50.0, 2.0, 1.0, 0.3])
        xu = np.array([2.0, 10.0, 0.3, 500.0, 15.0, 5.0, 0.7])

        super().__init__(
            n_var=7,
            n_obj=3,
            n_constr=0,
            xl=xl,
            xu=xu
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.sensitive_val = sensitive_val
        self._eval_count = 0

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate a population of solutions."""
        F = np.zeros((X.shape[0], 3))

        for i in range(X.shape[0]):
            obj1, obj2, obj3, _, _ = build_and_evaluate(
                X[i], self.X_train, self.y_train,
                self.X_val, self.y_val, self.sensitive_val
            )
            F[i, 0] = obj1  # 1 - balanced_accuracy
            F[i, 1] = obj2  # DP violation
            F[i, 2] = obj3  # EO violation

        self._eval_count += X.shape[0]
        if self._eval_count % 200 == 0:
            print(f"  [pymoo] {self._eval_count} evaluations completed...")

        out["F"] = F


# ──────────────────────────────────────────────────────────────────────────────
# 5. PARETO FRONT EXTRACTION & EVALUATION ON TEST SET
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(pareto_X, X_train_full, y_train_full,
                     X_test, y_test, sensitive_test):
    """
    Re-train each Pareto-optimal configuration on the full training set
    and evaluate on the held-out test set.
    """
    results = []
    model_type_names = {0: 'LogisticRegression', 1: 'RandomForest', 2: 'XGBoost'}

    for idx, x_vec in enumerate(pareto_X):
        obj1, obj2, obj3, model, threshold = build_and_evaluate(
            x_vec, X_train_full, y_train_full,
            X_test, y_test, sensitive_test
        )

        mt = int(np.clip(np.round(x_vec[0]), 0, 2))
        results.append({
            'solution_id': idx + 1,
            'model_type': model_type_names[mt],
            'model_type_int': mt,
            'regularization': float(x_vec[1]),
            'learning_rate': float(x_vec[2]),
            'n_estimators': int(np.round(x_vec[3])),
            'max_depth': int(np.round(x_vec[4])),
            'class_weight_ratio': float(x_vec[5]),
            'threshold': float(x_vec[6]),
            'balanced_accuracy': 1.0 - obj1,
            'dp_violation': obj2,
            'eo_violation': obj3,
            'demographic_parity': 1.0 - obj2,
            'equalized_odds': 1.0 - obj3,
            'obj1_min': obj1,
            'obj2_min': obj2,
            'obj3_min': obj3,
        })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# 6. MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  MULTI-OBJECTIVE OPTIMIZATION FOR FAIRNESS IN ML")
    print("  Algorithm: NSGA-II (pymoo)")
    print("  Dataset: COMPAS Recidivism")
    print("  Objectives: Balanced Accuracy, Demographic Parity, Equalized Odds")
    print("=" * 70)

    # ── Load data ──
    data = load_and_preprocess_compas()
    X_train, y_train, X_test, y_test, sens_train, sens_test, preprocessor = data

    # ── Create train/validation split for optimization ──
    X_tr, X_val, y_tr, y_val, sens_tr_idx, sens_val_idx = train_test_split(
        X_train, y_train, range(len(y_train)),
        test_size=0.25, random_state=42, stratify=y_train
    )
    sens_val = sens_train.iloc[sens_val_idx].reset_index(drop=True)
    y_tr = y_tr.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    # ── Define the MOO problem ──
    problem = FairnessMOOProblem(X_tr, y_tr, X_val, y_val, sens_val)

    # ── Configure NSGA-II ──
    algorithm = NSGA2(
        pop_size=60,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 40)

    print("\n🚀 Starting NSGA-II optimization...")
    print(f"   Population: 60, Generations: 40")
    print(f"   Total evaluations: ~2,400\n")

    start_time = time.time()

    res = pymoo_minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=True,
        save_history=True
    )

    elapsed = time.time() - start_time
    print(f"\n✅ NSGA-II completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"   Total evaluations: {problem._eval_count}")
    print(f"   Pareto-optimal solutions found: {len(res.F)}")

    # ── Evaluate Pareto front on test set ──
    print("\n📊 Evaluating Pareto front on test set...")
    pareto_df = evaluate_on_test(
        res.X, X_train, y_train, X_test, y_test, sens_test
    )

    # ── Display Pareto table ──
    print("\n" + "=" * 70)
    print("  PARETO-OPTIMAL SOLUTIONS (Test Set)")
    print("=" * 70)

    display_cols = [
        'solution_id', 'model_type', 'balanced_accuracy',
        'demographic_parity', 'equalized_odds',
        'threshold', 'class_weight_ratio'
    ]
    print(pareto_df[display_cols].to_string(index=False, float_format='%.4f'))

    # ── Save results ──
    os.makedirs('results', exist_ok=True)

    pareto_df.to_csv('results/nsga2_pareto_front.csv', index=False)
    print(f"\n💾 Pareto front saved to results/nsga2_pareto_front.csv")

    # Save full optimization history for analysis
    history_data = {
        'pareto_X': res.X,
        'pareto_F': res.F,
        'pareto_df': pareto_df,
        'elapsed_time': elapsed,
        'n_evaluations': problem._eval_count,
        'algorithm': 'NSGA-II',
        'history_F': [gen.pop.get("F") for gen in res.history] if res.history else None,
        'history_X': [gen.pop.get("X") for gen in res.history] if res.history else None,
    }

    with open('results/nsga2_results.pkl', 'wb') as f:
        pickle.dump(history_data, f)
    print(f"💾 Full results saved to results/nsga2_results.pkl")

    # ── Summary stats ──
    print("\n" + "=" * 70)
    print("  SUMMARY STATISTICS")
    print("=" * 70)
    print(f"  Balanced Accuracy range: [{pareto_df['balanced_accuracy'].min():.4f}, {pareto_df['balanced_accuracy'].max():.4f}]")
    print(f"  Demographic Parity range: [{pareto_df['demographic_parity'].min():.4f}, {pareto_df['demographic_parity'].max():.4f}]")
    print(f"  Equalized Odds range:     [{pareto_df['equalized_odds'].min():.4f}, {pareto_df['equalized_odds'].max():.4f}]")
    print(f"  DP Violation range:       [{pareto_df['dp_violation'].min():.4f}, {pareto_df['dp_violation'].max():.4f}]")
    print(f"  EO Violation range:       [{pareto_df['eo_violation'].min():.4f}, {pareto_df['eo_violation'].max():.4f}]")

    # ── Highlight a key trade-off ──
    best_acc_idx = pareto_df['balanced_accuracy'].idxmax()
    best_fair_idx = pareto_df['dp_violation'].idxmin()

    print("\n  📌 KEY TRADE-OFF:")
    ba = pareto_df.loc[best_acc_idx]
    bf = pareto_df.loc[best_fair_idx]
    print(f"  Highest accuracy solution: BalAcc={ba['balanced_accuracy']:.4f}, "
          f"DP={ba['dp_violation']:.4f}, EO={ba['eo_violation']:.4f}")
    print(f"  Fairest solution:          BalAcc={bf['balanced_accuracy']:.4f}, "
          f"DP={bf['dp_violation']:.4f}, EO={bf['eo_violation']:.4f}")

    acc_drop = ba['balanced_accuracy'] - bf['balanced_accuracy']
    dp_gain = ba['dp_violation'] - bf['dp_violation']
    print(f"  → Dropping {acc_drop:.4f} in accuracy yields {dp_gain:.4f} improvement in DP fairness")

    return pareto_df, history_data


if __name__ == '__main__':
    pareto_df, history_data = main()

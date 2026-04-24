# Multi-Objective Optimization for Fairness in Machine Learning

## COMPAS Recidivism — Pareto Front Analysis

This project implements multi-objective optimization (MOO) to simultaneously balance **predictive performance** and **fairness** in a recidivism prediction model using the COMPAS dataset.

### Three Objectives (Optimized Simultaneously)
1. **Maximize Balanced Accuracy** — robust to class imbalance
2. **Maximize Demographic Parity** — equal positive prediction rates across racial groups
3. **Maximize Equalized Odds** — equal TPR and FPR across racial groups

### Decision Variables
| Variable | Range | Description |
|---|---|---|
| Model Type | {LR, RF, XGB} | Classifier architecture |
| Regularization | [1e-4, 10] | L2/lambda strength |
| Learning Rate | [0.005, 0.3] | XGBoost learning rate |
| N Estimators | [50, 500] | Trees / iterations |
| Max Depth | [2, 15] | Tree depth |
| Class Weight Ratio | [1.0, 5.0] | Minority class weight |
| Threshold | [0.3, 0.7] | Classification threshold |

### Algorithms Compared
- **NSGA-II** (pymoo) — evolutionary multi-objective optimization
- **Optuna NSGA-II** — Bayesian-guided multi-objective sampling

---

## Setup

```bash
pip install -r requirements.txt
```

Ensure `compas-scores-two-years.csv` is in the project directory.   
Download from: https://www.kaggle.com/datasets/danofer/compass

## Run

```bash
# 1. Run NSGA-II optimization (primary)
python moo_nsga2_pymoo.py

# 2. Run Optuna comparison
python moo_optuna_comparison.py

# 3. Generate all visualizations & metrics
python visualize_results.py
```

Results are saved in the `results/` directory.

## Output
- `results/nsga2_pareto_front.csv` — Pareto-optimal solutions (NSGA-II)
- `results/optuna_pareto_front.csv` — Pareto-optimal solutions (Optuna)
- `results/pareto_2d_*.png` — 2D Pareto scatter plots
- `results/pareto_3d_scatter.png` — 3D Pareto visualization
- `results/parallel_coordinates.png` — Parallel coordinates plot
- `results/hv_convergence.png` — Hypervolume convergence
- `results/algorithm_comparison.png` — NSGA-II vs Optuna comparison
- `results/tradeoff_heatmap.png` — Correlation heatmap

## Authors
Niranjan Gopal, Divyam Sareen

## Acknowledgments
Dataset: ProPublica COMPAS via Kaggle

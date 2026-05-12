# Multi-Objective Optimization for Fairness & Complexity in ML

This project implements advanced multi-objective optimization (MOO) to simultaneously balance **predictive performance**, **algorithmic fairness**, and **model complexity** in a recidivism prediction model using the COMPAS dataset. 

Instead of traditional hyperparameter tuning that optimizes a single metric, this project maps out the entire **Pareto Frontier**—finding the absolute best trade-offs across three inherently conflicting objectives.

---

## Three Conflicting Objectives
All objectives are mathematically minimized by the algorithms:

1. **Maximize Balanced Accuracy** 📈
   - Minimized as `(1 - balanced_accuracy)`
   - Ensures the model remains robust against class imbalance.
2. **Maximize Demographic Parity** 
   - Minimized as `|P(Ŷ=1|African-American) - P(Ŷ=1|Caucasian)|`
   - Ensures the positive prediction rate is equal across racial groups.
3. **Minimize Model Complexity** 
   - A normalized metric `[0, 1]` based on tree depth, number of estimators, and model type.
   - Ensures the chosen model remains as simple, interpretable, and computationally cheap as possible.

---

## Decision Variables (The Search Space)
The algorithms search across 7 discrete and continuous variables simultaneously:

| Variable | Range | Description |
|---|---|---|
| **Model Type** | `{LR, RF, XGB}` | Logistic Regression, Random Forest, or XGBoost |
| **Regularization** | `[1e-4, 10]` | L2/lambda regularization strength |
| **Learning Rate** | `[0.005, 0.3]` | Step size (used only for XGBoost) |
| **N Estimators** | `[50, 500]` | Number of trees / boosting rounds |
| **Max Depth** | `[2, 15]` | Maximum tree depth |
| **Class Weight** | `[1.0, 5.0]` | Weight applied to the minority class |
| **Threshold** | `[0.3, 0.7]` | Probability threshold for classification |

---

## Setup & Execution

### 1. Installation
Ensure you have a Python virtual environment set up, then install the dependencies:
```bash
pip install -r requirements.txt
```
*Note: Make sure `compas-scores-two-years.csv` is located in the root project directory.*

### 2. Interactive Notebook Execution (Recommended)
The Python scripts in this project are formatted with `#%%` markers. If you open them in **VS Code** (with the Jupyter extension installed), you can execute the code blocks interactively by clicking the **Run Cell** buttons. This allows you to see the outputs and graphs inline, block-by-block.

### 3. Standard Execution
If you prefer running them from the terminal, execute them in this exact order:

```bash
# 1. Run Pymoo NSGA-II optimization (~30,000 evaluations)
python moo_nsga2_pymoo.py

# 2. Run Optuna NSGA-II comparison (2,500 trials)
python moo_optuna_comparison.py

# 3. Generate all visualizations & aggregate metrics
python visualize_results.py
```

---

## 📂 Expected Output Architecture

Running `visualize_results.py` will generate a rich set of analysis files in the `results/` folder:

### Global Aggregate Summaries (Root `results/` folder)
- `algorithm_comparison.png` — Head-to-head bar charts (Hypervolume, Spacing) comparing Pymoo vs Optuna.
- `hv_convergence.png` — Line graph tracking Hypervolume optimization over generations.
- `parallel_coordinates.png` — Tracks the hyperparameter DNA of every single Pareto-optimal solution.
- `model_distribution.png` — Bar chart showing which model types dominated the Pareto front.
- `tradeoff_heatmap.png` — Correlation grid exposing exactly how Accuracy fights against Fairness.

### Model-Specific Projections (`results/pymoo/` and `results/optuna/`)
Inside each algorithm's subfolder, results are further broken down by model type (e.g., `results/pymoo/RandomForest/`):
- **`interactive_pareto_3d.html`** — A fully interactive Plotly 3D scatter plot. Open this in your browser to rotate the 3D Pareto frontier and hover over points to see exact hyperparameters!
- **`pareto_2d_*.png`** — Three static 2D projection views of the 3D space. **Dominated points are visible as a dimmed cloud in the background** to provide visual context for the Pareto frontier drawn across them.

---

## Authors
Daksh Rajesh, Mohammad Owais

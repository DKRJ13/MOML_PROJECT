#%% [markdown]
# # Visualization and Analysis
# Generating interactive plots for multi-objective optimization results.

#%%
"""
Visualization & Metrics for MOO Fairness & Complexity Results
=============================================================
Generates:
  - 2D Pareto scatter plots with 2D-dominance highlighting
  - 3D Pareto scatter plot
  - Parallel coordinates plot
  - Hypervolume indicator computation
  - Spacing metric
  - Generational distance (NSGA-II vs Optuna)
  - Pareto points tabulation
  - Algorithm comparison summary
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import seaborn as sns
from tabulate import tabulate
import plotly.graph_objects as go

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11


# ──────────────────────────────────────────────────────────────────────────────
# 1. QUALITY METRICS
# ──────────────────────────────────────────────────────────────────────────────

#%% [markdown]
# ## Helper Functions
# Mathematical operations to compute Pareto dominance and metrics.

#%%
def compute_hypervolume(pareto_F, ref_point):
    """
    Compute hypervolume indicator for a Pareto front.
    All objectives are MINIMIZED, ref_point should dominate all solutions.

    Uses pymoo's built-in HV computation.
    """
    try:
        from pymoo.indicators.hv import HV
        hv = HV(ref_point=np.array(ref_point))
        return hv(np.array(pareto_F))
    except ImportError:
        # Fallback: approximate 2D HV for first two objectives
        print("  [WARNING] pymoo not available for HV computation, using fallback")
        F = np.array(pareto_F)[:, :2]
        sorted_F = F[F[:, 0].argsort()]
        hv_val = 0.0
        for i in range(len(sorted_F)):
            if i == 0:
                width = ref_point[0] - sorted_F[i, 0]
            else:
                width = sorted_F[i-1, 0] - sorted_F[i, 0]
            height = ref_point[1] - sorted_F[i, 1]
            hv_val += max(0, width) * max(0, height)
        return hv_val


def compute_spacing(pareto_F):
    """
    Compute spacing metric (Schott's spacing).
    Measures uniformity of the Pareto front distribution.
    Lower is better (more uniform).
    """
    F = np.array(pareto_F)
    n = len(F)
    if n <= 1:
        return 0.0

    # Compute minimum distance from each point to any other
    min_dists = []
    for i in range(n):
        dists = []
        for j in range(n):
            if i != j:
                dists.append(np.sum(np.abs(F[i] - F[j])))  # L1 distance
        min_dists.append(min(dists))

    d_bar = np.mean(min_dists)
    spacing = np.sqrt(np.sum([(d - d_bar) ** 2 for d in min_dists]) / n)
    return spacing


def compute_generational_distance(front_A, front_B):
    """
    Compute Generational Distance from front_A to front_B.
    GD measures how far front_A is from front_B.
    Lower means front_A is closer to front_B.
    """
    A = np.array(front_A)
    B = np.array(front_B)

    if len(A) == 0 or len(B) == 0:
        return float('inf')

    distances = []
    for a in A:
        min_dist = min(np.linalg.norm(a - b) for b in B)
        distances.append(min_dist)

    return np.mean(distances)


# ──────────────────────────────────────────────────────────────────────────────
# 2. 2D PARETO SCATTER PLOTS
# ──────────────────────────────────────────────────────────────────────────────

def _is_2d_pareto(x_vals, y_vals, x_minimize=True, y_minimize=True):
    """
    Find indices of points that are Pareto-optimal in 2D projection.
    Points dominated in 2D but kept in 3D Pareto set will be marked non-optimal.
    """
    n = len(x_vals)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            xb = (x_vals[j] < x_vals[i]) if x_minimize else (x_vals[j] > x_vals[i])
            xe = (x_vals[j] == x_vals[i]) if x_minimize else (x_vals[j] == x_vals[i])
            yb = (y_vals[j] < y_vals[i]) if y_minimize else (y_vals[j] > y_vals[i])
            ye = (y_vals[j] == y_vals[i]) if y_minimize else (y_vals[j] == y_vals[i])
            if (xb or xe) and (yb or ye) and (xb or yb):
                is_pareto[i] = False
                break
    return is_pareto


#%% [markdown]
# ## Plotting Functions
# Generates model-specific distributions, convergence charts, and aggregate comparisons.

#%%
def plot_model_specific_visualizations(df, algo_name, save_dir='results'):
    """
    Generate model-specific 2D and 3D Pareto plots in subfolders.
    Shows dominated points dimmed out in 2D, and generates an interactive 3D HTML plot.
    """
    if df is None or len(df) == 0:
        return
        
    algo_dir = os.path.join(save_dir, algo_name)
    os.makedirs(algo_dir, exist_ok=True)

    pairs = [
        ('balanced_accuracy', 'dp_violation',
         'Balanced Accuracy ↑', 'DP Violation ↓',
         'Accuracy vs. Demographic Parity', False, True),
        ('balanced_accuracy', 'model_complexity',
         'Balanced Accuracy ↑', 'Model Complexity ↓',
         'Accuracy vs. Model Complexity', False, True),
        ('dp_violation', 'model_complexity',
         'DP Violation ↓', 'Model Complexity ↓',
         'Demographic Parity vs. Model Complexity', True, True),
    ]

    color_map = {
        'LogisticRegression': '#2196F3',
        'RandomForest': '#4CAF50',
        'XGBoost': '#FF9800'
    }

    for model_type in df['model_type'].unique():
        model_dir = os.path.join(algo_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        df_model = df[df['model_type'] == model_type].copy()
        if len(df_model) == 0:
            continue
            
        color = color_map.get(model_type, '#9E9E9E')
        
        # --- 1. Interactive 3D Pareto Plot (Plotly) ---
        hover_text = []
        for _, row in df_model.iterrows():
            text = (f"Model: {model_type}<br>"
                    f"Balanced Acc: {row['balanced_accuracy']:.4f}<br>"
                    f"DP Violation: {row['dp_violation']:.4f}<br>"
                    f"Complexity: {row['model_complexity']:.4f}<br>"
                    f"Threshold: {row['threshold']:.3f}<br>"
                    f"CW Ratio: {row['class_weight_ratio']:.2f}<br>"
                    f"Reg: {row.get('regularization', 0):.4f}<br>"
                    f"Est: {row.get('n_estimators', 0)}<br>"
                    f"Depth: {row.get('max_depth', 0)}")
            hover_text.append(text)
            
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=df_model['balanced_accuracy'],
            y=df_model['dp_violation'],
            z=df_model['model_complexity'],
            mode='markers',
            marker=dict(
                size=8,
                color=color,
                line=dict(width=1, color='DarkSlateGrey'),
                opacity=0.8
            ),
            text=hover_text,
            hoverinfo='text'
        )])

        fig_3d.update_layout(
            title=f"Interactive 3D Pareto Front: {model_type} ({algo_name})",
            scene=dict(
                xaxis_title='Balanced Accuracy ↑',
                yaxis_title='DP Violation ↓',
                zaxis_title='Model Complexity ↓'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        fig_3d.write_html(os.path.join(model_dir, 'interactive_pareto_3d.html'))

        # --- 2. 2D Pareto Plots with Dimmed Dominated Points ---
        for x_col, y_col, x_label, y_label, title, x_min, y_min in pairs:
            x_vals = df_model[x_col].values
            y_vals = df_model[y_col].values
            
            # Find points on the 2D Pareto frontier for this model type
            pareto_mask = _is_2d_pareto(x_vals, y_vals, x_minimize=x_min, y_minimize=y_min)
            df_pareto = df_model[pareto_mask]
            df_dominated = df_model[~pareto_mask]
            
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Plot the dominated points in the background (dimmed)
            if len(df_dominated) > 0:
                ax.scatter(df_dominated[x_col], df_dominated[y_col], c='gray', s=60,
                           edgecolors='gray', linewidths=0.5, alpha=0.3, 
                           label=f'{model_type} (Dominated)')
            
            # Plot only the 2D Pareto optimal points
            if len(df_pareto) > 0:
                ax.scatter(df_pareto[x_col], df_pareto[y_col], c=color, s=140,
                           edgecolors='black', linewidths=1.0, alpha=0.9, 
                           label=f'{model_type} (Pareto)')
                
                # Connect the Pareto points
                sorted_par = df_pareto.sort_values(x_col)
                ax.plot(sorted_par[x_col], sorted_par[y_col],
                        linestyle='-', color=color, alpha=0.6, linewidth=2)

            ax.set_xlabel(x_label, fontsize=13)
            ax.set_ylabel(y_label, fontsize=13)
            ax.set_title(f"2D Pareto: {title}\n{model_type} ({algo_name})", fontsize=14, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.4)
            fig.tight_layout()
            
            fname = f"pareto_2d_{x_col}_vs_{y_col}.png"
            fig.savefig(os.path.join(model_dir, fname), dpi=150, bbox_inches='tight')
            plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 4. PARALLEL COORDINATES PLOT
# ──────────────────────────────────────────────────────────────────────────────

def plot_parallel_coordinates(df_nsga, save_dir='results'):
    """
    Parallel coordinates plot showing objectives and key decision variables.
    Each line is one Pareto-optimal solution.
    """
    # Normalize all columns to [0, 1] for visualization
    plot_cols = [
        'balanced_accuracy', 'demographic_parity', 'model_complexity',
        'threshold', 'class_weight_ratio'
    ]
    # Ensure columns exist
    available = [c for c in plot_cols if c in df_nsga.columns]
    df_plot = df_nsga[available + ['model_type']].copy()

    # Normalize
    for col in available:
        rng = df_plot[col].max() - df_plot[col].min()
        if rng > 0:
            df_plot[col + '_norm'] = (df_plot[col] - df_plot[col].min()) / rng
        else:
            df_plot[col + '_norm'] = 0.5

    fig, ax = plt.subplots(figsize=(14, 7))

    norm_cols = [c + '_norm' for c in available]
    x_pos = range(len(available))

    color_map = {
        'LogisticRegression': '#2196F3',
        'RandomForest': '#4CAF50',
        'XGBoost': '#FF9800'
    }

    for _, row in df_plot.iterrows():
        vals = [row[c] for c in norm_cols]
        color = color_map.get(row['model_type'], 'gray')
        ax.plot(x_pos, vals, color=color, alpha=0.5, linewidth=1.5)

    # Axis labels
    ax.set_xticks(x_pos)
    label_map = {
        'balanced_accuracy': 'Balanced\nAccuracy ↑',
        'demographic_parity': 'Demographic\nParity ↑',
        'model_complexity': 'Model\nComplexity ↓',
        'threshold': 'Threshold',
        'class_weight_ratio': 'Class Weight\nRatio',
    }
    labels = [label_map.get(c, c) for c in available]
    ax.set_xticklabels(labels, fontsize=10)

    # Add value labels on axes
    for i, col in enumerate(available):
        lo, hi = df_nsga[col].min(), df_nsga[col].max()
        ax.annotate(f'{lo:.3f}', (i, -0.05), fontsize=8, ha='center', color='gray')
        ax.annotate(f'{hi:.3f}', (i, 1.05), fontsize=8, ha='center', color='gray')

    # Vertical axis lines
    for i in x_pos:
        ax.axvline(i, color='gray', linewidth=0.5, alpha=0.3)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=c, linewidth=2, label=mt)
        for mt, c in color_map.items()
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_ylim(-0.1, 1.15)
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.set_title('Parallel Coordinates: Pareto-Optimal Solutions', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    fig.savefig(os.path.join(save_dir, 'parallel_coordinates.png'), dpi=150, bbox_inches='tight')
    print("  📈 Saved parallel_coordinates.png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 5. HYPERVOLUME CONVERGENCE (NSGA-II)
# ──────────────────────────────────────────────────────────────────────────────

def plot_hv_convergence(history_F, ref_point, save_dir='results'):
    """Plot hypervolume convergence over generations."""
    if history_F is None:
        print("  [SKIP] No generation history available for HV convergence plot")
        return

    from pymoo.indicators.hv import HV
    hv_calc = HV(ref_point=np.array(ref_point))

    hv_values = []
    for gen_F in history_F:
        # Filter to only valid solutions (not inf/nan)
        valid = gen_F[~np.any(np.isinf(gen_F) | np.isnan(gen_F), axis=1)]
        if len(valid) > 0:
            hv_values.append(hv_calc(valid))
        else:
            hv_values.append(0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(hv_values) + 1), hv_values,
            color='#1565C0', linewidth=2)
    ax.fill_between(range(1, len(hv_values) + 1), hv_values,
                     alpha=0.15, color='#1565C0')
    ax.set_xlabel('Generation', fontsize=13)
    ax.set_ylabel('Hypervolume', fontsize=13)
    ax.set_title('NSGA-II: Hypervolume Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()

    fig.savefig(os.path.join(save_dir, 'hv_convergence.png'), dpi=150, bbox_inches='tight')
    print("  📈 Saved hv_convergence.png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 6. MODEL TYPE DISTRIBUTION
# ──────────────────────────────────────────────────────────────────────────────

def plot_model_distribution(df_nsga, df_optuna=None, save_dir='results'):
    """Bar chart showing model type distribution in Pareto front."""
    fig, axes = plt.subplots(1, 2 if df_optuna is not None else 1,
                              figsize=(12 if df_optuna is not None else 7, 5))

    if df_optuna is None:
        axes = [axes]

    colors = ['#2196F3', '#4CAF50', '#FF9800']

    # NSGA-II
    counts_nsga = df_nsga['model_type'].value_counts()
    axes[0].bar(counts_nsga.index, counts_nsga.values, color=colors[:len(counts_nsga)])
    axes[0].set_title('NSGA-II: Model Types in Pareto Front', fontweight='bold')
    axes[0].set_ylabel('Count')

    for i, (mt, cnt) in enumerate(counts_nsga.items()):
        axes[0].annotate(str(cnt), (i, cnt), ha='center', va='bottom', fontweight='bold')

    if df_optuna is not None and len(df_optuna) > 0:
        counts_opt = df_optuna['model_type'].value_counts()
        axes[1].bar(counts_opt.index, counts_opt.values, color=colors[:len(counts_opt)])
        axes[1].set_title('Optuna: Model Types in Pareto Front', fontweight='bold')
        axes[1].set_ylabel('Count')
        for i, (mt, cnt) in enumerate(counts_opt.items()):
            axes[1].annotate(str(cnt), (i, cnt), ha='center', va='bottom', fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'model_distribution.png'), dpi=150, bbox_inches='tight')
    print("  📈 Saved model_distribution.png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 7. TRADE-OFF ANALYSIS HEATMAP
# ──────────────────────────────────────────────────────────────────────────────

def plot_tradeoff_heatmap(df_nsga, save_dir='results'):
    """Correlation heatmap between objectives and decision variables."""
    cols = ['balanced_accuracy', 'dp_violation', 'model_complexity',
            'threshold', 'class_weight_ratio', 'n_estimators', 'max_depth']
    available_cols = [c for c in cols if c in df_nsga.columns]

    if len(available_cols) < 3:
        print("  [SKIP] Not enough columns for heatmap")
        return

    corr = df_nsga[available_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax, vmin=-1, vmax=1,
                xticklabels=[c.replace('_', '\n') for c in available_cols],
                yticklabels=[c.replace('_', '\n') for c in available_cols])
    ax.set_title('Correlation: Objectives vs Decision Variables', fontsize=13, fontweight='bold')
    fig.tight_layout()

    fig.savefig(os.path.join(save_dir, 'tradeoff_heatmap.png'), dpi=150, bbox_inches='tight')
    print("  📈 Saved tradeoff_heatmap.png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 8. ALGORITHM COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

def compare_algorithms(nsga_data, optuna_data, ref_point, save_dir='results'):
    """Print and plot comparison between NSGA-II and Optuna."""
    df_nsga = nsga_data['pareto_df']
    df_optuna = optuna_data['pareto_df']

    nsga_F = df_nsga[['obj1_min', 'obj2_min', 'obj3_min']].values
    optuna_F = df_optuna[['obj1_min', 'obj2_min', 'obj3_min']].values

    # Metrics
    hv_nsga = compute_hypervolume(nsga_F, ref_point)
    hv_optuna = compute_hypervolume(optuna_F, ref_point)

    sp_nsga = compute_spacing(nsga_F)
    sp_optuna = compute_spacing(optuna_F)

    gd_nsga_to_opt = compute_generational_distance(nsga_F, optuna_F)
    gd_opt_to_nsga = compute_generational_distance(optuna_F, nsga_F)

    print("\n" + "=" * 70)
    print("  ALGORITHM COMPARISON: NSGA-II vs Optuna NSGA-II")
    print("=" * 70)

    comparison = [
        ['Metric', 'NSGA-II (pymoo)', 'Optuna NSGA-II'],
        ['Pareto solutions', len(df_nsga), len(df_optuna)],
        ['Hypervolume ↑', f'{hv_nsga:.6f}', f'{hv_optuna:.6f}'],
        ['Spacing ↓', f'{sp_nsga:.6f}', f'{sp_optuna:.6f}'],
        ['GD (to other) ↓', f'{gd_nsga_to_opt:.6f}', f'{gd_opt_to_nsga:.6f}'],
        ['Runtime (s)', f'{nsga_data.get("elapsed_time", 0):.1f}',
         f'{optuna_data.get("elapsed_time", 0):.1f}'],
        ['Best Bal. Accuracy', f'{df_nsga["balanced_accuracy"].max():.4f}',
         f'{df_optuna["balanced_accuracy"].max():.4f}'],
        ['Best DP (min viol.)', f'{df_nsga["dp_violation"].min():.4f}',
         f'{df_optuna["dp_violation"].min():.4f}'],
        ['Best Complexity (min)', f'{df_nsga["model_complexity"].min():.4f}',
         f'{df_optuna["model_complexity"].min():.4f}'],
    ]
    print(tabulate(comparison, headers='firstrow', tablefmt='grid'))

    # Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Hypervolume
    bars = axes[0].bar(['NSGA-II', 'Optuna'], [hv_nsga, hv_optuna],
                        color=['#1565C0', '#E91E63'])
    axes[0].set_title('Hypervolume ↑', fontweight='bold')
    axes[0].set_ylabel('HV')
    for bar, val in zip(bars, [hv_nsga, hv_optuna]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # Spacing
    bars = axes[1].bar(['NSGA-II', 'Optuna'], [sp_nsga, sp_optuna],
                        color=['#1565C0', '#E91E63'])
    axes[1].set_title('Spacing ↓', fontweight='bold')
    axes[1].set_ylabel('Spacing')
    for bar, val in zip(bars, [sp_nsga, sp_optuna]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # Solutions count
    bars = axes[2].bar(['NSGA-II', 'Optuna'], [len(df_nsga), len(df_optuna)],
                        color=['#1565C0', '#E91E63'])
    axes[2].set_title('Pareto Solutions ↑', fontweight='bold')
    axes[2].set_ylabel('Count')
    for bar, val in zip(bars, [len(df_nsga), len(df_optuna)]):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.suptitle('Algorithm Comparison: NSGA-II vs Optuna', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'algorithm_comparison.png'), dpi=150, bbox_inches='tight')
    print("  📈 Saved algorithm_comparison.png")
    plt.close(fig)

    return {
        'hv_nsga': hv_nsga, 'hv_optuna': hv_optuna,
        'sp_nsga': sp_nsga, 'sp_optuna': sp_optuna,
        'gd_nsga_to_opt': gd_nsga_to_opt, 'gd_opt_to_nsga': gd_opt_to_nsga,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 9. PARETO TABULATION
# ──────────────────────────────────────────────────────────────────────────────

def print_pareto_table(df, algorithm_name='NSGA-II'):
    """Pretty-print the Pareto front table."""
    print(f"\n{'=' * 80}")
    print(f"  PARETO FRONT TABULATION — {algorithm_name}")
    print(f"{'=' * 80}")

    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            int(row['solution_id']),
            row['model_type'],
            f"{row['balanced_accuracy']:.4f}",
            f"{row.get('demographic_parity', 1.0 - row.get('dp_violation', 0)):.4f}",
            f"{row.get('model_complexity', 0):.4f}",
            f"{row['dp_violation']:.4f}",
            f"{row['threshold']:.3f}",
            f"{row['class_weight_ratio']:.2f}",
        ])

    headers = ['#', 'Model', 'Bal.Acc', 'DP↑', 'Complexity↓', 'DP Viol↓',
               'Thresh', 'CW Ratio']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))


# ──────────────────────────────────────────────────────────────────────────────
# 10. DETAILED SOLUTION ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def detailed_solution_analysis(df, save_dir='results'):
    """Analyze one Pareto-optimal solution in detail (for appendix)."""
    if len(df) == 0:
        print("  No solutions to analyze")
        return

    # Pick the "knee" solution — best balanced trade-off
    # Normalize objectives and pick the solution closest to ideal
    norm_acc = 1.0 - (df['balanced_accuracy'] - df['balanced_accuracy'].min()) / \
               max(df['balanced_accuracy'].max() - df['balanced_accuracy'].min(), 1e-9)
    norm_dp = (df['dp_violation'] - df['dp_violation'].min()) / \
              max(df['dp_violation'].max() - df['dp_violation'].min(), 1e-9)
    norm_cx = (df['model_complexity'] - df['model_complexity'].min()) / \
              max(df['model_complexity'].max() - df['model_complexity'].min(), 1e-9)

    composite = norm_acc + norm_dp + norm_cx
    knee_idx = composite.idxmin()
    sol = df.loc[knee_idx]

    print("\n" + "=" * 70)
    print("  DETAILED ANALYSIS: KNEE SOLUTION (Best Balanced Trade-off)")
    print("=" * 70)
    print(f"\n  Solution ID: {int(sol['solution_id'])}")
    print(f"\n  --- Decision Variables ---")
    print(f"  Model Type:         {sol['model_type']}")
    print(f"  Regularization:     {sol['regularization']:.6f}")
    print(f"  Learning Rate:      {sol['learning_rate']:.6f}")
    print(f"  N Estimators:       {int(sol['n_estimators'])}")
    print(f"  Max Depth:          {int(sol['max_depth'])}")
    print(f"  Class Weight Ratio: {sol['class_weight_ratio']:.4f}")
    print(f"  Threshold:          {sol['threshold']:.4f}")
    print(f"\n  --- Objective Values ---")
    print(f"  Balanced Accuracy:    {sol['balanced_accuracy']:.4f}")
    print(f"  Demographic Parity:   {sol.get('demographic_parity', 1.0 - sol.get('dp_violation', 0)):.4f} (violation: {sol['dp_violation']:.4f})")
    print(f"  Model Complexity:     {sol['model_complexity']:.4f}")
    print(f"\n  --- Interpretation ---")
    print(f"  This solution balances all three objectives.")
    if sol['dp_violation'] < 0.1:
        print(f"  ✓ Low DP violation ({sol['dp_violation']:.4f}): prediction rates are")
        print(f"    similar across racial groups.")
    if sol['model_complexity'] < 0.3:
        print(f"  ✓ Low complexity ({sol['model_complexity']:.4f}): model is relatively")
        print(f"    simple and interpretable.")
    elif sol['model_complexity'] > 0.6:
        print(f"  ⚠ High complexity ({sol['model_complexity']:.4f}): model is complex,")
        print(f"    potentially harder to interpret.")

    return sol


# ──────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

#%% [markdown]
# ## Main Execution
# Loading the CSV files, running visualization logic, and reporting statistics.

#%%
def main():
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("  VISUALIZATION & METRICS — MOO Fairness Results")
    print("=" * 70)

    # ── Load results ──
    nsga_data = None
    optuna_data = None

    if os.path.exists('results/pymoo_results.pkl'):
        with open('results/pymoo_results.pkl', 'rb') as f:
            nsga_data = pickle.load(f)
        print(f"✅ Loaded Pymoo results ({len(nsga_data['pareto_df'])} Pareto solutions)")
    else:
        print("⚠️  Pymoo results not found. Run moo_nsga2_pymoo.py first.")
        return

    if os.path.exists('results/optuna_results.pkl'):
        with open('results/optuna_results.pkl', 'rb') as f:
            optuna_data = pickle.load(f)
        print(f"✅ Loaded Optuna results ({len(optuna_data['pareto_df'])} Pareto solutions)")
    else:
        print("⚠️  Optuna results not found. Comparison will be skipped.")

    df_nsga = nsga_data['pareto_df']
    df_optuna = optuna_data['pareto_df'] if optuna_data else None

    # ── Reference point for HV (worst-case for all 3 minimization objectives) ──
    ref_point = [1.0, 1.0, 1.0]  # All objectives bounded in [0, 1]

    # ── 1. Pareto Tabulation ──
    print_pareto_table(df_nsga, 'Pymoo (NSGA-II)')
    if df_optuna is not None:
        print_pareto_table(df_optuna, 'Optuna NSGA-II')

    # ── 2. Quality Metrics ──
    nsga_F = df_nsga[['obj1_min', 'obj2_min', 'obj3_min']].values
    hv_nsga = compute_hypervolume(nsga_F, ref_point)
    sp_nsga = compute_spacing(nsga_F)

    print("\n📏 Pymoo Metrics:")
    print(f"   Hypervolume:  {hv_nsga:.6f}")
    print(f"   Spacing:      {sp_nsga:.6f}")
    print(f"   # Solutions:  {len(df_nsga)}")

    if df_optuna is not None:
        optuna_F = df_optuna[['obj1_min', 'obj2_min', 'obj3_min']].values
        hv_optuna = compute_hypervolume(optuna_F, ref_point)
        sp_optuna = compute_spacing(optuna_F)
        print(f"\n📏 Optuna Metrics:")
        print(f"   Hypervolume:  {hv_optuna:.6f}")
        print(f"   Spacing:      {sp_optuna:.6f}")
        print(f"   # Solutions:  {len(df_optuna)}")

    # ── 3. Plots ──
    print("\n📊 Generating visualizations...")
    plot_model_specific_visualizations(df_nsga, 'pymoo', save_dir)
    if df_optuna is not None:
        plot_model_specific_visualizations(df_optuna, 'optuna', save_dir)
        
    plot_parallel_coordinates(df_nsga, save_dir)
    plot_model_distribution(df_nsga, df_optuna, save_dir)
    plot_tradeoff_heatmap(df_nsga, save_dir)

    # HV convergence (needs generation history)
    if nsga_data.get('history_F') is not None:
        plot_hv_convergence(nsga_data['history_F'], ref_point, save_dir)

    # ── 4. Algorithm comparison ──
    if optuna_data is not None:
        compare_algorithms(nsga_data, optuna_data, ref_point, save_dir)

    # ── 5. Detailed solution analysis ──
    detailed_solution_analysis(df_nsga, save_dir)

    print(f"\n✅ All visualizations saved to {save_dir}/")
    print("=" * 70)


#%% [markdown]
# Execute the visualization pipeline.

#%%
if __name__ == '__main__':
    main()

"""
Visualization & Metrics for MOO Fairness Results
=================================================
Generates:
  - 2D Pareto scatter plots (3 pairwise combinations)
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

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11


# ──────────────────────────────────────────────────────────────────────────────
# 1. QUALITY METRICS
# ──────────────────────────────────────────────────────────────────────────────

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

def plot_2d_pareto(df_nsga, df_optuna=None, save_dir='results'):
    """Generate 3 pairwise 2D Pareto scatter plots."""
    pairs = [
        ('balanced_accuracy', 'dp_violation',
         'Balanced Accuracy ↑', 'Demographic Parity Violation ↓',
         'Accuracy vs. Demographic Parity'),
        ('balanced_accuracy', 'eo_violation',
         'Balanced Accuracy ↑', 'Equalized Odds Violation ↓',
         'Accuracy vs. Equalized Odds'),
        ('dp_violation', 'eo_violation',
         'DP Violation ↓', 'EO Violation ↓',
         'Demographic Parity vs. Equalized Odds'),
    ]

    for x_col, y_col, x_label, y_label, title in pairs:
        fig, ax = plt.subplots(figsize=(10, 7))

        # NSGA-II points
        colors_nsga = df_nsga['model_type'].map({
            'LogisticRegression': '#2196F3',
            'RandomForest': '#4CAF50',
            'XGBoost': '#FF9800'
        })
        ax.scatter(
            df_nsga[x_col], df_nsga[y_col],
            c=colors_nsga, s=120, edgecolors='black', linewidths=0.8,
            zorder=3, alpha=0.85, label='NSGA-II'
        )

        # Connect NSGA-II Pareto-optimal front line
        sorted_nsga = df_nsga.sort_values(x_col)
        ax.plot(sorted_nsga[x_col], sorted_nsga[y_col],
                linestyle='--', color='#1565C0', alpha=0.4, zorder=2)

        if df_optuna is not None and len(df_optuna) > 0:
            ax.scatter(
                df_optuna[x_col], df_optuna[y_col],
                c='#E91E63', s=80, marker='D', edgecolors='black',
                linewidths=0.8, zorder=2, alpha=0.7, label='Optuna NSGA-II'
            )

        # Annotations for NSGA-II top solutions
        for _, row in df_nsga.head(8).iterrows():
            ax.annotate(
                f"{row['model_type'][:3]}\n{row['threshold']:.2f}",
                (row[x_col], row[y_col]),
                textcoords="offset points", xytext=(8, 8),
                fontsize=7, ha='left',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7),
            )

        # Legend for model types
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3',
                   markersize=10, label='Logistic Regression'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50',
                   markersize=10, label='Random Forest'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9800',
                   markersize=10, label='XGBoost'),
        ]
        if df_optuna is not None:
            legend_elements.append(
                Line2D([0], [0], marker='D', color='w', markerfacecolor='#E91E63',
                       markersize=8, label='Optuna NSGA-II')
            )
        ax.legend(handles=legend_elements, loc='best', fontsize=9)

        ax.set_xlabel(x_label, fontsize=13)
        ax.set_ylabel(y_label, fontsize=13)
        ax.set_title(f"Pareto Front: {title}", fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()

        fname = f"pareto_2d_{x_col}_vs_{y_col}.png"
        fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
        print(f"  📈 Saved {fname}")
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 3. 3D PARETO SCATTER PLOT
# ──────────────────────────────────────────────────────────────────────────────

def plot_3d_pareto(df_nsga, df_optuna=None, save_dir='results'):
    """Generate 3D scatter of all three objectives."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Color by model type
    color_map = {
        'LogisticRegression': '#2196F3',
        'RandomForest': '#4CAF50',
        'XGBoost': '#FF9800'
    }

    for mt, color in color_map.items():
        mask = df_nsga['model_type'] == mt
        if mask.any():
            ax.scatter(
                df_nsga.loc[mask, 'balanced_accuracy'],
                df_nsga.loc[mask, 'dp_violation'],
                df_nsga.loc[mask, 'eo_violation'],
                c=color, s=100, edgecolors='black', linewidths=0.5,
                alpha=0.8, label=f'NSGA-II: {mt}'
            )

    if df_optuna is not None and len(df_optuna) > 0:
        ax.scatter(
            df_optuna['balanced_accuracy'],
            df_optuna['dp_violation'],
            df_optuna['eo_violation'],
            c='#E91E63', s=70, marker='D', edgecolors='black',
            linewidths=0.5, alpha=0.6, label='Optuna NSGA-II'
        )

    ax.set_xlabel('Balanced Accuracy ↑', fontsize=11, labelpad=10)
    ax.set_ylabel('DP Violation ↓', fontsize=11, labelpad=10)
    ax.set_zlabel('EO Violation ↓', fontsize=11, labelpad=10)
    ax.set_title('3D Pareto Front: All Three Objectives', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)

    # Better viewing angle
    ax.view_init(elev=25, azim=45)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'pareto_3d_scatter.png'), dpi=150, bbox_inches='tight')
    print("  📈 Saved pareto_3d_scatter.png")
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
        'balanced_accuracy', 'demographic_parity', 'equalized_odds',
        'threshold', 'class_weight_ratio'
    ]
    df_plot = df_nsga[plot_cols + ['model_type']].copy()

    # Normalize
    for col in plot_cols:
        rng = df_plot[col].max() - df_plot[col].min()
        if rng > 0:
            df_plot[col + '_norm'] = (df_plot[col] - df_plot[col].min()) / rng
        else:
            df_plot[col + '_norm'] = 0.5

    fig, ax = plt.subplots(figsize=(14, 7))

    norm_cols = [c + '_norm' for c in plot_cols]
    x_pos = range(len(plot_cols))

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
    labels = ['Balanced\nAccuracy ↑', 'Demographic\nParity ↑', 'Equalized\nOdds ↑',
              'Threshold', 'Class Weight\nRatio']
    ax.set_xticklabels(labels, fontsize=10)

    # Add value labels on axes
    for i, col in enumerate(plot_cols):
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
    cols = ['balanced_accuracy', 'dp_violation', 'eo_violation',
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
        ['Best EO (min viol.)', f'{df_nsga["eo_violation"].min():.4f}',
         f'{df_optuna["eo_violation"].min():.4f}'],
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
            f"{row['demographic_parity']:.4f}",
            f"{row['equalized_odds']:.4f}",
            f"{row['dp_violation']:.4f}",
            f"{row['eo_violation']:.4f}",
            f"{row['threshold']:.3f}",
            f"{row['class_weight_ratio']:.2f}",
        ])

    headers = ['#', 'Model', 'Bal.Acc', 'DP↑', 'EO↑', 'DP Viol↓', 'EO Viol↓',
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
    norm_eo = (df['eo_violation'] - df['eo_violation'].min()) / \
              max(df['eo_violation'].max() - df['eo_violation'].min(), 1e-9)

    composite = norm_acc + norm_dp + norm_eo
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
    print(f"  Demographic Parity:   {sol['demographic_parity']:.4f} (violation: {sol['dp_violation']:.4f})")
    print(f"  Equalized Odds:       {sol['equalized_odds']:.4f} (violation: {sol['eo_violation']:.4f})")
    print(f"\n  --- Interpretation ---")
    print(f"  This solution balances all three objectives.")
    if sol['dp_violation'] < 0.1:
        print(f"  ✓ Low DP violation ({sol['dp_violation']:.4f}): prediction rates are")
        print(f"    similar across racial groups.")
    if sol['eo_violation'] < 0.15:
        print(f"  ✓ Low EO violation ({sol['eo_violation']:.4f}): true positive and")
        print(f"    false positive rates are relatively balanced across groups.")

    return sol


# ──────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

def main():
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("  VISUALIZATION & METRICS — MOO Fairness Results")
    print("=" * 70)

    # ── Load results ──
    nsga_data = None
    optuna_data = None

    if os.path.exists('results/nsga2_results.pkl'):
        with open('results/nsga2_results.pkl', 'rb') as f:
            nsga_data = pickle.load(f)
        print(f"✅ Loaded NSGA-II results ({len(nsga_data['pareto_df'])} Pareto solutions)")
    else:
        print("⚠️  NSGA-II results not found. Run moo_nsga2_pymoo.py first.")
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
    print_pareto_table(df_nsga, 'NSGA-II (pymoo)')
    if df_optuna is not None:
        print_pareto_table(df_optuna, 'Optuna NSGA-II')

    # ── 2. Quality Metrics ──
    nsga_F = df_nsga[['obj1_min', 'obj2_min', 'obj3_min']].values
    hv_nsga = compute_hypervolume(nsga_F, ref_point)
    sp_nsga = compute_spacing(nsga_F)

    print(f"\n📏 NSGA-II Metrics:")
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
    plot_2d_pareto(df_nsga, df_optuna, save_dir)
    plot_3d_pareto(df_nsga, df_optuna, save_dir)
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


if __name__ == '__main__':
    main()

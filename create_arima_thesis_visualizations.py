#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ARIMA Thesis Visualization Generator
==================================

Creates publication-ready visualizations for ARIMA time series forecasting results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

# Create output directory
output_dir = Path("Diagrams/arima_visualizations")
output_dir.mkdir(exist_ok=True)

# Load ARIMA results
print("Loading ARIMA results...")
try:
    summary_df = pd.read_csv("Models/outputs/arima/arima_summary_by_genre.csv")
    print(f"[OK] Loaded summary: {len(summary_df)} genres")
except FileNotFoundError:
    print("[ERROR] Could not find ARIMA summary file")
    exit()

# Load individual forecasts
forecasts = {}
holdouts = {}
genres = ['pop', 'hip-hop', 'latin', 'electronic']

for genre in genres:
    try:
        forecast_file = f"Models/outputs/arima/{genre}_forecast_1to5y.csv"
        holdout_file = f"Models/outputs/arima/{genre}_holdout_2023_2024_vs_forecast.csv"

        forecasts[genre] = pd.read_csv(forecast_file)
        holdouts[genre] = pd.read_csv(holdout_file)
        print(f"[OK] Loaded {genre} forecast and holdout data")
    except FileNotFoundError:
        print(f"[ERROR] Could not find files for {genre}")

def create_arima_performance_comparison():
    """Create ARIMA vs XGBoost performance comparison."""
    print("\nCreating ARIMA vs XGBoost comparison...")

    # XGBoost results for comparison (from our fixed implementation)
    xgb_results = {
        'genre': ['pop', 'hip-hop', 'latin', 'electronic'],
        'xgb_rmse': [8.23, 4.55, 6.56, 4.37],
        'xgb_mae': [7.79, 3.61, 5.34, 3.76]
    }
    xgb_df = pd.DataFrame(xgb_results)

    # Merge with ARIMA results
    comparison_df = summary_df.merge(xgb_df, on='genre')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # RMSE comparison
    x = np.arange(len(comparison_df))
    width = 0.35

    bars1 = ax1.bar(x - width/2, comparison_df['avg_rmse_walkforward'], width,
                    label='ARIMA', color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x + width/2, comparison_df['xgb_rmse'], width,
                    label='XGBoost', color='#A23B72', alpha=0.8)

    ax1.set_xlabel('Genre', fontsize=14, fontweight='bold')
    ax1.set_ylabel('RMSE Score', fontsize=14, fontweight='bold')
    ax1.set_title('Model Performance Comparison: RMSE', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([g.title() for g in comparison_df['genre']])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    # MAE comparison
    bars3 = ax2.bar(x - width/2, comparison_df['avg_mae_walkforward'], width,
                    label='ARIMA', color='#2E86AB', alpha=0.8)
    bars4 = ax2.bar(x + width/2, comparison_df['xgb_mae'], width,
                    label='XGBoost', color='#A23B72', alpha=0.8)

    ax2.set_xlabel('Genre', fontsize=14, fontweight='bold')
    ax2.set_ylabel('MAE Score', fontsize=14, fontweight='bold')
    ax2.set_title('Model Performance Comparison: MAE', fontsize=16, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([g.title() for g in comparison_df['genre']])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('ARIMA vs XGBoost: Time Series Forecasting Performance\\n(Lower is Better)',
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'arima_vs_xgboost_comparison.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Created arima_vs_xgboost_comparison.png")

def create_arima_forecast_summary():
    """Create a summary of all ARIMA forecasts."""
    print("\nCreating forecast summary visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for i, (genre, color) in enumerate(zip(genres, colors)):
        if genre not in forecasts:
            continue

        ax = axes[i]
        forecast_df = forecasts[genre]

        # Plot forecasts
        ax.plot(forecast_df['year'], forecast_df['forecast_popularity'],
                marker='o', linewidth=3, color=color, label='Forecast')

        # Add holdout data if available
        if genre in holdouts:
            holdout_df = holdouts[genre]
            ax.scatter(holdout_df['year'], holdout_df['actual'],
                      color='red', s=100, label='Actual (2023-2024)', zorder=5)
            ax.scatter(holdout_df['year'], holdout_df['forecast'],
                      color='orange', s=100, label='Predicted (2023-2024)', zorder=5)

        # Formatting
        ax.set_title(f'{genre.title()} Genre Forecast', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Popularity Score', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add RMSE annotation
        row = summary_df[summary_df['genre'] == genre]
        if not row.empty:
            rmse = row['avg_rmse_walkforward'].iloc[0]
            ax.text(0.05, 0.95, f'RMSE: {rmse:.2f}',
                   transform=ax.transAxes, fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('ARIMA Time Series Forecasts by Genre\\n(2025-2029 Predictions)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'arima_forecasts_summary.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Created arima_forecasts_summary.png")

def create_arima_model_summary():
    """Create ARIMA model summary with (p,d,q) parameters."""
    print("\nCreating ARIMA model summary...")

    # Extract (p,d,q) parameters manually
    p_values, d_values, q_values = [], [], []
    for pdq_str in summary_df['final_pdq']:
        # Remove quotes and parentheses, then split
        clean_str = pdq_str.strip('"').strip('()').replace(' ', '')
        p, d, q = map(int, clean_str.split(','))
        p_values.append(p)
        d_values.append(d)
        q_values.append(q)

    summary_df['p'] = p_values
    summary_df['d'] = d_values
    summary_df['q'] = q_values

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Walk-forward performance
    bars1 = ax1.bar(summary_df['genre'], summary_df['avg_rmse_walkforward'],
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    ax1.set_title('Walk-Forward Validation RMSE', fontsize=14, fontweight='bold')
    ax1.set_ylabel('RMSE Score', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars1, summary_df['avg_rmse_walkforward']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # Holdout performance
    bars2 = ax2.bar(summary_df['genre'], summary_df['holdout_rmse_2023_2024'],
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    ax2.set_title('Holdout Validation RMSE (2023-2024)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RMSE Score', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars2, summary_df['holdout_rmse_2023_2024']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # Number of folds
    bars3 = ax3.bar(summary_df['genre'], summary_df['folds'],
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    ax3.set_title('Validation Folds Available', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Folds', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars3, summary_df['folds']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{int(val)}', ha='center', va='bottom', fontweight='bold')

    # ARIMA parameters visualization
    x = np.arange(len(summary_df))
    width = 0.25

    bars4a = ax4.bar(x - width, summary_df['p'], width, label='p (AR)', color='#FF6B6B', alpha=0.8)
    bars4b = ax4.bar(x, summary_df['d'], width, label='d (I)', color='#4ECDC4', alpha=0.8)
    bars4c = ax4.bar(x + width, summary_df['q'], width, label='q (MA)', color='#45B7D1', alpha=0.8)

    ax4.set_title('ARIMA Model Parameters (p,d,q)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Parameter Value', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels([g.title() for g in summary_df['genre']])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    # Format all x-axis labels
    for ax in [ax1, ax2, ax3]:
        ax.set_xticklabels([g.title() for g in summary_df['genre']])

    plt.suptitle('ARIMA Model Performance and Configuration Summary',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'arima_model_summary.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Created arima_model_summary.png")

def create_thesis_arima_dashboard():
    """Create comprehensive ARIMA dashboard for thesis."""
    print("\nCreating ARIMA thesis dashboard...")

    fig = plt.figure(figsize=(20, 16))

    # Create custom grid layout
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.3)

    # Top row: Individual forecasts
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for i, (genre, color) in enumerate(zip(genres, colors)):
        if genre not in forecasts:
            continue

        ax = fig.add_subplot(gs[0, i])
        forecast_df = forecasts[genre]

        # Plot forecast with years 2025-2029
        ax.plot(forecast_df['year'], forecast_df['forecast_popularity'],
                marker='o', linewidth=2, color=color, markersize=6)

        # Add holdout points if available
        if genre in holdouts:
            holdout_df = holdouts[genre]
            ax.scatter(holdout_df['year'], holdout_df['actual'],
                      color='red', s=60, label='Actual', zorder=5)
            ax.scatter(holdout_df['year'], holdout_df['forecast'],
                      color='orange', s=60, label='Predicted', zorder=5)

        ax.set_title(f'{genre.title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Popularity', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add RMSE
        row = summary_df[summary_df['genre'] == genre]
        if not row.empty:
            rmse = row['avg_rmse_walkforward'].iloc[0]
            ax.text(0.05, 0.95, f'RMSE: {rmse:.2f}',
                   transform=ax.transAxes, fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Middle row: Performance comparison
    ax_comp = fig.add_subplot(gs[1, :2])

    # ARIMA vs XGBoost comparison
    xgb_rmse = [8.23, 4.55, 6.56, 4.37]  # From our analysis
    x = np.arange(len(summary_df))
    width = 0.35

    bars1 = ax_comp.bar(x - width/2, summary_df['avg_rmse_walkforward'], width,
                       label='ARIMA', color='#2E86AB', alpha=0.8)
    bars2 = ax_comp.bar(x + width/2, xgb_rmse, width,
                       label='XGBoost', color='#A23B72', alpha=0.8)

    ax_comp.set_title('Performance Comparison: ARIMA vs XGBoost', fontsize=14, fontweight='bold')
    ax_comp.set_ylabel('RMSE Score', fontsize=12)
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels([g.title() for g in summary_df['genre']])
    ax_comp.legend()
    ax_comp.grid(axis='y', alpha=0.3)

    # Middle right: Model parameters
    ax_params = fig.add_subplot(gs[1, 2:])

    # Extract (p,d,q) from final_pdq strings
    pdq_data = []
    for _, row in summary_df.iterrows():
        pdq_str = row['final_pdq']
        # Extract numbers from string like "(2, 1, 3)"
        import re
        matches = re.findall(r'\\d+', pdq_str)
        if len(matches) == 3:
            pdq_data.append([int(x) for x in matches])
        else:
            pdq_data.append([0, 0, 0])

    pdq_array = np.array(pdq_data)

    x = np.arange(len(summary_df))
    width = 0.25

    ax_params.bar(x - width, pdq_array[:, 0], width, label='p (AR)', color='#FF6B6B', alpha=0.8)
    ax_params.bar(x, pdq_array[:, 1], width, label='d (I)', color='#4ECDC4', alpha=0.8)
    ax_params.bar(x + width, pdq_array[:, 2], width, label='q (MA)', color='#45B7D1', alpha=0.8)

    ax_params.set_title('ARIMA Model Parameters', fontsize=14, fontweight='bold')
    ax_params.set_ylabel('Parameter Value', fontsize=12)
    ax_params.set_xticks(x)
    ax_params.set_xticklabels([g.title() for g in summary_df['genre']])
    ax_params.legend()
    ax_params.grid(axis='y', alpha=0.3)

    # Bottom: Key metrics table
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')

    # Create summary table data
    table_data = []
    for _, row in summary_df.iterrows():
        table_data.append([
            row['genre'].title(),
            f"{row['avg_rmse_walkforward']:.3f}",
            f"{row['holdout_rmse_2023_2024']:.2f}",
            str(int(row['folds'])),
            row['final_pdq'],
            "✓ Yes" if row['meets_thresholds_walkforward'] else "✗ No"
        ])

    table = ax_table.table(cellText=table_data,
                          colLabels=['Genre', 'Walk-Forward RMSE', 'Holdout RMSE', 'Folds', 'Model (p,d,q)', 'Meets Thresholds'],
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(6):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#E8E8E8')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#F8F8F8')

    plt.suptitle('ARIMA Time Series Forecasting: Comprehensive Analysis\\n' +
                 'Music Genre Popularity Prediction (2025-2029)',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig(output_dir / 'arima_thesis_dashboard.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Created arima_thesis_dashboard.png")

def main():
    """Generate all ARIMA thesis visualizations."""
    print("=" * 60)
    print("ARIMA THESIS VISUALIZATION GENERATOR")
    print("=" * 60)

    create_arima_performance_comparison()
    create_arima_forecast_summary()
    create_arima_model_summary()
    create_thesis_arima_dashboard()

    print(f"\n[SUCCESS] All ARIMA visualizations created in: {output_dir}")
    print("\nGenerated files:")
    for file in output_dir.glob("*.png"):
        print(f"  [PLOT] {file.name}")

    print(f"\n[READY] ARIMA visualizations ready for thesis inclusion!")

if __name__ == "__main__":
    main()
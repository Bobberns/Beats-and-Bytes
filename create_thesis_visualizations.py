#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Thesis Visualization Generator for XGBoost Feature Importance Analysis
====================================================================

This script creates publication-ready visualizations for the thesis paper
showing audio feature importance across different music genres.
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
output_dir = Path("Diagrams/thesis_visualizations")
output_dir.mkdir(exist_ok=True)

# Load feature importance data for all genres
genres = ['pop', 'hip-hop', 'latin', 'electronic']
feature_data = {}

print("Loading feature importance data...")
for genre in genres:
    file_path = f"Models/outputs/xgb_fixed/{genre}_feature_importance.csv"
    try:
        df = pd.read_csv(file_path)
        feature_data[genre] = df
        print(f"[OK] Loaded {genre}: {len(df)} features")
    except FileNotFoundError:
        print(f"[ERROR] Could not find {file_path}")

# Load summary data
try:
    summary_df = pd.read_csv("Models/outputs/xgb_fixed/xgb_fixed_summary.csv")
    print(f"[OK] Loaded summary data: {len(summary_df)} genres")
except FileNotFoundError:
    print("[ERROR] Could not find summary data")
    summary_df = None

def create_individual_genre_plots():
    """Create individual feature importance plots for each genre."""
    print("\nCreating individual genre plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for i, (genre, color) in enumerate(zip(genres, colors)):
        if genre not in feature_data:
            continue

        ax = axes[i]
        df = feature_data[genre].sort_values('importance', ascending=True)

        # Create horizontal bar plot
        bars = ax.barh(df['feature'], df['importance'], color=color, alpha=0.8)

        # Add value labels
        for bar, importance in zip(bars, df['importance']):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')

        # Formatting
        ax.set_title(f'{genre.title()} Genre', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, max(df['importance']) * 1.15)

        # Clean up feature names for display
        feature_labels = [f.replace('_', ' ').title() for f in df['feature']]
        ax.set_yticklabels(feature_labels, fontsize=10)

    plt.suptitle('XGBoost Feature Importance by Music Genre\n(Fixed Implementation)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'individual_genre_feature_importance.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Created individual_genre_feature_importance.png")

def create_comparative_heatmap():
    """Create a comparative heatmap showing feature importance across genres."""
    print("\nCreating comparative heatmap...")

    # Prepare data for heatmap
    all_features = set()
    for df in feature_data.values():
        all_features.update(df['feature'])

    # Create matrix
    importance_matrix = pd.DataFrame(index=sorted(all_features), columns=genres)

    for genre in genres:
        if genre in feature_data:
            df = feature_data[genre].set_index('feature')
            importance_matrix[genre] = df['importance']

    importance_matrix = importance_matrix.fillna(0).astype(float)

    # Create heatmap
    plt.figure(figsize=(12, 10))

    # Custom colormap
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    ax = sns.heatmap(importance_matrix,
                     annot=True,
                     fmt='.3f',
                     cmap=cmap,
                     center=0,
                     square=False,
                     cbar_kws={'label': 'Feature Importance Score'})

    # Formatting
    plt.title('Audio Feature Importance Across Music Genres\n(XGBoost Analysis)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Music Genre', fontsize=14, fontweight='bold')
    plt.ylabel('Audio Features', fontsize=14, fontweight='bold')

    # Clean up labels
    feature_labels = [f.replace('_', ' ').title() for f in importance_matrix.index]
    genre_labels = [g.title() for g in importance_matrix.columns]

    ax.set_yticklabels(feature_labels, rotation=0)
    ax.set_xticklabels(genre_labels, rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_heatmap.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Created feature_importance_heatmap.png")

def create_top_features_comparison():
    """Create a comparison of top 3 features per genre."""
    print("\nCreating top features comparison...")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data
    x_pos = np.arange(len(genres))
    width = 0.25
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    # Get top 3 features per genre
    top_features_data = {}
    for genre in genres:
        if genre in feature_data:
            top_3 = feature_data[genre].nlargest(3, 'importance')
            top_features_data[genre] = top_3

    # Create grouped bar chart
    for i in range(3):  # Top 3 features
        values = []
        labels = []

        for genre in genres:
            if genre in top_features_data and i < len(top_features_data[genre]):
                row = top_features_data[genre].iloc[i]
                values.append(row['importance'])
                if i == 0:  # Only add label for first set
                    labels.append(row['feature'].replace('_', ' ').title())
            else:
                values.append(0)

        bars = ax.bar(x_pos + i * width, values, width,
                     label=f'{"1st" if i==0 else "2nd" if i==1 else "3rd"} Most Important',
                     color=colors[i], alpha=0.8)

        # Add value labels
        for j, (bar, val) in enumerate(zip(bars, values)):
            if val > 0:
                feature_name = top_features_data[genres[j]].iloc[i]['feature'].replace('_', ' ')
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{feature_name}\n({val:.3f})',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Formatting
    ax.set_xlabel('Music Genre', fontsize=14, fontweight='bold')
    ax.set_ylabel('Feature Importance Score', fontsize=14, fontweight='bold')
    ax.set_title('Top 3 Audio Features by Genre\n(Most Predictive of Track Popularity)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([g.title() for g in genres])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'top_features_comparison.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Created top_features_comparison.png")

def create_model_performance_summary():
    """Create model performance summary visualization."""
    if summary_df is None:
        print("[ERROR] Skipping performance summary (no data)")
        return

    print("\nCreating model performance summary...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # RMSE comparison
    bars1 = ax1.bar(summary_df['genre'], summary_df['test_rmse'],
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    ax1.set_title('Root Mean Square Error (RMSE)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('RMSE Score', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars1, summary_df['test_rmse']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # MAE comparison
    bars2 = ax2.bar(summary_df['genre'], summary_df['test_mae'],
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    ax2.set_title('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('MAE Score', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars2, summary_df['test_mae']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # R² comparison
    bars3 = ax3.bar(summary_df['genre'], summary_df['test_r2'],
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    ax3.set_title('R² Score', fontsize=14, fontweight='bold')
    ax3.set_ylabel('R² Score', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Baseline (Mean)')

    # Add value labels
    for bar, val in zip(bars3, summary_df['test_r2']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # Sample sizes
    bars4 = ax4.bar(summary_df['genre'], summary_df['n_samples'],
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    ax4.set_title('Dataset Size per Genre', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Number of Tracks', fontsize=12)
    ax4.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars4, summary_df['n_samples']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}', ha='center', va='bottom', fontweight='bold')

    # Format all x-axis labels
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticklabels([g.title() for g in summary_df['genre']], rotation=0)

    plt.suptitle('XGBoost Model Performance Summary\n(Fixed Implementation)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance_summary.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Created model_performance_summary.png")

def create_thesis_dashboard():
    """Create a comprehensive dashboard for the thesis."""
    print("\nCreating thesis dashboard...")

    fig = plt.figure(figsize=(20, 16))

    # Create grid layout
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], width_ratios=[1, 1, 1])

    # Top row: Individual genre plots (smaller)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for i, (genre, color) in enumerate(zip(genres, colors)):
        if genre not in feature_data:
            continue

        ax = fig.add_subplot(gs[0, i if i < 3 else 2])  # Fit first 3 in top row
        if i == 3:  # Electronic goes in top right
            ax = fig.add_subplot(gs[0, 2])

        df = feature_data[genre].sort_values('importance', ascending=True)

        bars = ax.barh(df['feature'], df['importance'], color=color, alpha=0.8)

        # Add value labels (smaller)
        for bar, importance in zip(bars, df['importance']):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', ha='left', va='center', fontsize=8)

        ax.set_title(f'{genre.title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, max(df['importance']) * 1.15)

        # Clean feature names
        feature_labels = [f.replace('_', ' ').title() for f in df['feature']]
        ax.set_yticklabels(feature_labels, fontsize=8)

    # Middle: Comparative heatmap
    ax_heatmap = fig.add_subplot(gs[1, :])

    # Prepare heatmap data
    all_features = set()
    for df in feature_data.values():
        all_features.update(df['feature'])

    importance_matrix = pd.DataFrame(index=sorted(all_features), columns=genres)
    for genre in genres:
        if genre in feature_data:
            df = feature_data[genre].set_index('feature')
            importance_matrix[genre] = df['importance']
    importance_matrix = importance_matrix.fillna(0).astype(float)

    sns.heatmap(importance_matrix, annot=True, fmt='.3f',
                cmap='YlOrRd', ax=ax_heatmap, cbar_kws={'label': 'Importance'})
    ax_heatmap.set_title('Feature Importance Heatmap Across Genres',
                        fontsize=14, fontweight='bold')
    ax_heatmap.set_xlabel('Genre', fontsize=12)
    ax_heatmap.set_ylabel('Audio Features', fontsize=12)

    # Clean labels
    feature_labels = [f.replace('_', ' ').title() for f in importance_matrix.index]
    genre_labels = [g.title() for g in importance_matrix.columns]
    ax_heatmap.set_yticklabels(feature_labels, rotation=0)
    ax_heatmap.set_xticklabels(genre_labels, rotation=0)

    # Bottom: Performance metrics
    if summary_df is not None:
        ax_perf = fig.add_subplot(gs[2, :])

        x = np.arange(len(genres))
        width = 0.25

        bars1 = ax_perf.bar(x - width, summary_df['test_rmse'], width,
                           label='RMSE', color='#FF6B6B', alpha=0.8)
        bars2 = ax_perf.bar(x, summary_df['test_mae'], width,
                           label='MAE', color='#4ECDC4', alpha=0.8)
        bars3 = ax_perf.bar(x + width, np.abs(summary_df['test_r2']), width,
                           label='|R²|', color='#45B7D1', alpha=0.8)

        ax_perf.set_xlabel('Genre', fontsize=12)
        ax_perf.set_ylabel('Score', fontsize=12)
        ax_perf.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        ax_perf.set_xticks(x)
        ax_perf.set_xticklabels([g.title() for g in genres])
        ax_perf.legend()
        ax_perf.grid(axis='y', alpha=0.3)

    plt.suptitle('Music Genre Popularity Prediction: Audio Feature Analysis\n' +
                 'XGBoost Model Results (Fixed Implementation)',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'thesis_dashboard.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Created thesis_dashboard.png")

def main():
    """Generate all thesis visualizations."""
    print("=" * 60)
    print("THESIS VISUALIZATION GENERATOR")
    print("=" * 60)

    if not feature_data:
        print("[ERROR] No feature importance data found. Please run the fixed XGBoost pipeline first.")
        return

    # Create all visualizations
    create_individual_genre_plots()
    create_comparative_heatmap()
    create_top_features_comparison()
    create_model_performance_summary()
    create_thesis_dashboard()

    print(f"\n[SUCCESS] All visualizations created in: {output_dir}")
    print("\nGenerated files:")
    for file in output_dir.glob("*.png"):
        print(f"  [PLOT] {file.name}")

    print(f"\n[READY] Files ready for thesis inclusion!")

if __name__ == "__main__":
    main()
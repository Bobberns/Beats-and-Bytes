"""
Generate XGBoost temporal audio feature charts
Shows how audio features evolved over time (2008-2024) for each genre
Combined with feature importance from XGBoost models
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9
plt.rcParams['figure.figsize'] = (14, 8)

# Define paths
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "Datasets"
XGB_DIR = BASE_DIR / "Models" / "outputs" / "xgb_fixed"
OUTPUT_DIR = BASE_DIR / "Diagrams" / "xgb_temporal_charts"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Genre configuration
GENRES = ['pop', 'hip-hop', 'latin', 'electronic']
GENRE_COLORS = {
    'pop': '#FF6B6B',
    'hip-hop': '#4ECDC4',
    'latin': '#FFD93D',
    'electronic': '#95E1D3'
}
GENRE_DISPLAY_NAMES = {
    'pop': 'Pop',
    'hip-hop': 'Hip-Hop',
    'latin': 'Latin',
    'electronic': 'Electronic'
}

# Audio features
AUDIO_FEATURES = [
    'energy', 'tempo', 'danceability', 'loudness', 'liveness',
    'valence', 'speechiness', 'instrumentalness', 'acousticness'
]

FEATURE_DISPLAY_NAMES = {
    'energy': 'Energy',
    'tempo': 'Tempo (BPM)',
    'danceability': 'Danceability',
    'loudness': 'Loudness (dB)',
    'liveness': 'Liveness',
    'valence': 'Valence',
    'speechiness': 'Speechiness',
    'instrumentalness': 'Instrumentalness',
    'acousticness': 'Acousticness'
}

print("="*70)
print("LOADING DATA")
print("="*70)

# Load historical audio features by genre
print("\nLoading temporal audio feature data...")
historical_data = pd.read_csv(DATASETS_DIR / "05_genres_median_audio_features.csv")
historical_data = historical_data.rename(columns={'track_album_release_date': 'Year'})

# Rename feature columns to match our naming
feature_mapping = {
    'median_energy': 'energy',
    'median_tempo': 'tempo',
    'median_danceability': 'danceability',
    'median_loudness': 'loudness',
    'median_liveness': 'liveness',
    'median_valence': 'valence',
    'median_speechiness': 'speechiness',
    'median_instrumentalness': 'instrumentalness',
    'median_acousticness': 'acousticness'
}
historical_data = historical_data.rename(columns=feature_mapping)

# Load XGBoost feature importance for each genre
print("\nLoading XGBoost feature importance...")
feature_importance = {}
for genre in GENRES:
    importance_file = XGB_DIR / f"{genre}_feature_importance.csv"
    df = pd.read_csv(importance_file)
    # Convert to dictionary
    feature_importance[genre] = dict(zip(df['feature'], df['importance']))
    print(f"  ✓ {GENRE_DISPLAY_NAMES[genre]}: {len(df)} features")

print("\n" + "="*70)
print("GENERATING TEMPORAL FEATURE CHARTS")
print("="*70)

# Chart 1: Individual genre - all features over time with importance ranking
print("\n1. Creating comprehensive feature evolution charts per genre...")
for genre in GENRES:
    genre_data = historical_data[historical_data['playlist_genre'] == genre].sort_values('Year')

    # Get top 5 features by importance
    importance = feature_importance[genre]
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    for idx, feature in enumerate(AUDIO_FEATURES):
        ax = axes[idx]

        # Plot feature over time
        ax.plot(genre_data['Year'], genre_data[feature],
                marker='o', linewidth=2.5, markersize=7,
                color=GENRE_COLORS[genre], alpha=0.8)

        # Get importance value
        imp_value = importance[feature]
        imp_pct = imp_value * 100

        # Check if it's in top 5
        is_top5 = feature in [f[0] for f in top_features]
        rank = [i for i, (f, _) in enumerate(top_features, 1) if f == feature][0] if is_top5 else None

        # Add importance annotation
        if is_top5:
            ax.text(0.98, 0.98, f'Rank #{rank}\nImportance: {imp_pct:.1f}%',
                    transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7),
                    fontsize=9, fontweight='bold')
        else:
            ax.text(0.98, 0.98, f'Importance: {imp_pct:.1f}%',
                    transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5),
                    fontsize=8)

        # Styling
        ax.set_xlabel('Year', fontsize=10, fontweight='bold')
        ax.set_ylabel(FEATURE_DISPLAY_NAMES[feature], fontsize=10, fontweight='bold')
        ax.set_title(FEATURE_DISPLAY_NAMES[feature], fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle(f'{GENRE_DISPLAY_NAMES[genre]}: Audio Feature Evolution Over Time (2008-2024)\nwith XGBoost Feature Importance Rankings',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_file = OUTPUT_DIR / f"{genre}_temporal_features_full.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_file.name}")
    plt.close()

# Chart 2: Top 3 features per genre - detailed view
print("\n2. Creating top 3 features detailed view per genre...")
for genre in GENRES:
    genre_data = historical_data[historical_data['playlist_genre'] == genre].sort_values('Year')

    # Get top 3 features
    importance = feature_importance[genre]
    top_3 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (feature, imp_value) in enumerate(top_3):
        ax = axes[idx]

        # Plot feature over time
        ax.plot(genre_data['Year'], genre_data[feature],
                marker='o', linewidth=3, markersize=9,
                color=GENRE_COLORS[genre], alpha=0.8)

        # Add trend line
        z = np.polyfit(genre_data['Year'], genre_data[feature], 1)
        p = np.poly1d(z)
        ax.plot(genre_data['Year'], p(genre_data['Year']),
                linestyle='--', linewidth=2, color='gray', alpha=0.5, label='Trend')

        # Calculate change
        first_val = genre_data[feature].iloc[0]
        last_val = genre_data[feature].iloc[-1]
        change = last_val - first_val
        change_pct = (change / first_val * 100) if first_val != 0 else 0

        # Styling
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel(FEATURE_DISPLAY_NAMES[feature], fontsize=12, fontweight='bold')
        ax.set_title(f'#{idx+1}: {FEATURE_DISPLAY_NAMES[feature]}\nImportance: {imp_value*100:.1f}% | Change: {change:+.3f} ({change_pct:+.1f}%)',
                    fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=10)

    plt.suptitle(f'{GENRE_DISPLAY_NAMES[genre]}: Top 3 Most Important Audio Features - Temporal Evolution',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_file = OUTPUT_DIR / f"{genre}_top3_features_temporal.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_file.name}")
    plt.close()

# Chart 3: Cross-genre comparison for each feature
print("\n3. Creating cross-genre feature comparison charts...")
for feature in AUDIO_FEATURES:
    fig, ax = plt.subplots(figsize=(14, 7))

    for genre in GENRES:
        genre_data = historical_data[historical_data['playlist_genre'] == genre].sort_values('Year')

        importance = feature_importance[genre]
        imp_value = importance[feature] * 100

        ax.plot(genre_data['Year'], genre_data[feature],
                marker='o', linewidth=2.5, markersize=7,
                color=GENRE_COLORS[genre],
                label=f'{GENRE_DISPLAY_NAMES[genre]} (Imp: {imp_value:.1f}%)',
                alpha=0.7)

    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel(FEATURE_DISPLAY_NAMES[feature], fontsize=13, fontweight='bold')
    ax.set_title(f'{FEATURE_DISPLAY_NAMES[feature]} Evolution Across All Genres (2008-2024)\nwith XGBoost Importance by Genre',
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    output_file = OUTPUT_DIR / f"cross_genre_{feature}_temporal.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_file.name}")
    plt.close()

# Chart 4: Feature importance heatmap with temporal trends
print("\n4. Creating feature importance heatmap with trend indicators...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Prepare importance matrix
importance_matrix = []
for genre in GENRES:
    importance = feature_importance[genre]
    importance_matrix.append([importance[f] * 100 for f in AUDIO_FEATURES])

importance_df = pd.DataFrame(importance_matrix,
                             index=[GENRE_DISPLAY_NAMES[g] for g in GENRES],
                             columns=[FEATURE_DISPLAY_NAMES[f] for f in AUDIO_FEATURES])

# Plot heatmap
sns.heatmap(importance_df, annot=True, fmt='.1f', cmap='YlOrRd',
            cbar_kws={'label': 'Importance (%)'}, ax=ax1,
            linewidths=0.5, linecolor='gray')
ax1.set_title('XGBoost Feature Importance by Genre (%)', fontsize=13, fontweight='bold', pad=15)
ax1.set_xlabel('Audio Features', fontsize=11, fontweight='bold')
ax1.set_ylabel('Genre', fontsize=11, fontweight='bold')

# Calculate trends (2008-2024 change) for each genre-feature
trend_matrix = []
for genre in GENRES:
    genre_data = historical_data[historical_data['playlist_genre'] == genre].sort_values('Year')
    trends = []
    for feature in AUDIO_FEATURES:
        if len(genre_data) > 1:
            first = genre_data[feature].iloc[0]
            last = genre_data[feature].iloc[-1]
            pct_change = ((last - first) / first * 100) if first != 0 else 0
            trends.append(pct_change)
        else:
            trends.append(0)
    trend_matrix.append(trends)

trend_df = pd.DataFrame(trend_matrix,
                        index=[GENRE_DISPLAY_NAMES[g] for g in GENRES],
                        columns=[FEATURE_DISPLAY_NAMES[f] for f in AUDIO_FEATURES])

# Plot trend heatmap
sns.heatmap(trend_df, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            cbar_kws={'label': 'Change (%)'}, ax=ax2,
            linewidths=0.5, linecolor='gray')
ax2.set_title('Feature Temporal Change (2008-2024) (%)', fontsize=13, fontweight='bold', pad=15)
ax2.set_xlabel('Audio Features', fontsize=11, fontweight='bold')
ax2.set_ylabel('Genre', fontsize=11, fontweight='bold')

plt.suptitle('XGBoost Feature Importance vs Temporal Trends', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

output_file = OUTPUT_DIR / "importance_vs_trends_heatmap.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_file.name}")
plt.close()

# Chart 5: Dashboard - Top 3 features for all genres
print("\n5. Creating comprehensive top features dashboard...")
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

for genre_idx, genre in enumerate(GENRES):
    genre_data = historical_data[historical_data['playlist_genre'] == genre].sort_values('Year')
    importance = feature_importance[genre]
    top_3 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]

    for feat_idx, (feature, imp_value) in enumerate(top_3):
        ax = fig.add_subplot(gs[genre_idx, feat_idx])

        # Plot
        ax.plot(genre_data['Year'], genre_data[feature],
                marker='o', linewidth=2, markersize=6,
                color=GENRE_COLORS[genre], alpha=0.8)

        # Styling
        ax.set_title(f'{GENRE_DISPLAY_NAMES[genre]} - {FEATURE_DISPLAY_NAMES[feature]}\n({imp_value*100:.1f}%)',
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Year', fontsize=9)
        ax.set_ylabel(FEATURE_DISPLAY_NAMES[feature], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

plt.suptitle('Top 3 Most Important Features by Genre: Temporal Evolution (2008-2024)',
             fontsize=18, fontweight='bold', y=0.995)

output_file = OUTPUT_DIR / "top_features_dashboard.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_file.name}")
plt.close()

print("\n" + "="*70)
print("TEMPORAL FEATURE CHART GENERATION COMPLETE")
print("="*70)
print(f"\nAll charts saved to: {OUTPUT_DIR}")
print("\nGenerated chart types:")
print("  1. Individual genre comprehensive views (4 files)")
print("     - All 9 features with importance rankings")
print("  2. Top 3 features detailed views per genre (4 files)")
print("     - Includes trend lines and change percentages")
print("  3. Cross-genre comparison per feature (9 files)")
print("     - Shows how each feature evolved across all genres")
print("  4. Feature importance vs trends heatmap (1 file)")
print("     - Side-by-side comparison of importance and temporal change")
print("  5. Comprehensive top features dashboard (1 file)")
print("     - All genres' top 3 features in one view")
print(f"\nTotal: 19 temporal feature charts generated!")

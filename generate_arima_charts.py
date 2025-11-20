"""
Generate ARIMA forecast charts for genre popularity analysis
Shows historical data (2008-2024) and forecasts (2025-2029)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 6)

# Define paths
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "Datasets"
MODELS_DIR = BASE_DIR / "Models" / "outputs" / "arima"
OUTPUT_DIR = BASE_DIR / "Diagrams" / "arima_charts"

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

# Load data
print("Loading data...")
historical_data = pd.read_csv(DATASETS_DIR / "06_aggregated_dataset.csv")

# Load forecasts
forecasts = {}
for genre in GENRES:
    forecast_file = MODELS_DIR / f"{genre}_forecast_1to5y.csv"
    forecasts[genre] = pd.read_csv(forecast_file)
    print(f"Loaded {genre} forecast: {len(forecasts[genre])} years")

# Prepare historical data by genre
print("\nPreparing historical data...")
genre_history = {}
for genre in GENRES:
    genre_data = historical_data[historical_data['Most_Prominent_Genre'] == genre].copy()
    genre_data = genre_data[['Year', 'median_track_popularity']].rename(
        columns={'median_track_popularity': 'popularity'}
    )
    genre_history[genre] = genre_data.sort_values('Year')
    print(f"{GENRE_DISPLAY_NAMES[genre]}: {len(genre_data)} historical observations")

print("\n" + "="*60)
print("GENERATING CHARTS")
print("="*60)

# Chart 1: Individual genre charts with forecasts
print("\n1. Creating individual genre forecast charts...")
for genre in GENRES:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Historical data
    hist = genre_history[genre]
    forecast = forecasts[genre]

    # Plot historical
    ax.plot(hist['Year'], hist['popularity'],
            marker='o', linewidth=2.5, markersize=8,
            color=GENRE_COLORS[genre], label='Historical', alpha=0.8)

    # Plot forecast
    ax.plot(forecast['year'], forecast['forecast_popularity'],
            marker='s', linewidth=2.5, markersize=8, linestyle='--',
            color=GENRE_COLORS[genre], label='Forecast (2025-2029)', alpha=0.7)

    # Add vertical line at forecast boundary
    ax.axvline(x=2024.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    ax.text(2024.5, ax.get_ylim()[1] * 0.98, ' Forecast Start',
            verticalalignment='top', fontsize=9, color='gray')

    # Styling
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Popularity Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{GENRE_DISPLAY_NAMES[genre]} Genre Popularity: Historical & Forecast',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Set x-axis to show all years
    all_years = list(hist['Year']) + list(forecast['year'])
    ax.set_xticks(range(min(all_years), max(all_years) + 1, 2))
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save
    output_file = OUTPUT_DIR / f"{genre}_popularity_forecast.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_file.name}")
    plt.close()

# Chart 2: Comprehensive 4-panel dashboard
print("\n2. Creating comprehensive 4-genre dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, genre in enumerate(GENRES):
    ax = axes[idx]

    hist = genre_history[genre]
    forecast = forecasts[genre]

    # Plot historical
    ax.plot(hist['Year'], hist['popularity'],
            marker='o', linewidth=2, markersize=7,
            color=GENRE_COLORS[genre], label='Historical', alpha=0.8)

    # Plot forecast
    ax.plot(forecast['year'], forecast['forecast_popularity'],
            marker='s', linewidth=2, markersize=7, linestyle='--',
            color=GENRE_COLORS[genre], label='Forecast', alpha=0.7)

    # Add vertical line
    ax.axvline(x=2024.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

    # Styling
    ax.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax.set_ylabel('Popularity Score', fontsize=11, fontweight='bold')
    ax.set_title(f'{GENRE_DISPLAY_NAMES[genre]}',
                fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Set x-axis
    all_years = list(hist['Year']) + list(forecast['year'])
    ax.set_xticks(range(min(all_years), max(all_years) + 1, 3))
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Genre Popularity Trends: Historical Data & ARIMA Forecasts (2008-2029)',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

output_file = OUTPUT_DIR / "all_genres_dashboard.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_file.name}")
plt.close()

# Chart 3: Overlay comparison chart
print("\n3. Creating overlay comparison chart...")
fig, ax = plt.subplots(figsize=(14, 8))

for genre in GENRES:
    hist = genre_history[genre]
    forecast = forecasts[genre]

    display_name = GENRE_DISPLAY_NAMES[genre]

    # Plot historical
    ax.plot(hist['Year'], hist['popularity'],
            marker='o', linewidth=2, markersize=6,
            color=GENRE_COLORS[genre], label=f'{display_name} (Historical)',
            alpha=0.7)

    # Plot forecast
    ax.plot(forecast['year'], forecast['forecast_popularity'],
            marker='s', linewidth=2, markersize=6, linestyle='--',
            color=GENRE_COLORS[genre], label=f'{display_name} (Forecast)',
            alpha=0.7)

# Add vertical line
ax.axvline(x=2024.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
ax.text(2024.5, ax.get_ylim()[1] * 0.98, ' Forecast Period',
        verticalalignment='top', fontsize=11, color='gray', fontweight='bold')

# Styling
ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Popularity Score', fontsize=13, fontweight='bold')
ax.set_title('All Genres Comparison: Historical Trends & ARIMA Forecasts',
            fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

# Set x-axis
ax.set_xticks(range(2008, 2030, 2))
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()

output_file = OUTPUT_DIR / "all_genres_overlay.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_file.name}")
plt.close()

# Chart 4: Forecast summary comparison (2025-2029)
print("\n4. Creating forecast summary comparison...")
fig, ax = plt.subplots(figsize=(12, 7))

# Prepare data for grouped bar chart
forecast_years = list(range(2025, 2030))
x = range(len(forecast_years))
width = 0.2

for idx, genre in enumerate(GENRES):
    forecast = forecasts[genre]
    offset = (idx - 1.5) * width

    ax.bar([pos + offset for pos in x],
           forecast['forecast_popularity'],
           width,
           label=GENRE_DISPLAY_NAMES[genre],
           color=GENRE_COLORS[genre],
           alpha=0.8)

# Styling
ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Forecasted Popularity Score', fontsize=13, fontweight='bold')
ax.set_title('ARIMA Forecast Comparison Across Genres (2025-2029)',
            fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(forecast_years)
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()

output_file = OUTPUT_DIR / "forecast_comparison_bars.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_file.name}")
plt.close()

# Chart 5: Trend summary table
print("\n5. Creating forecast trend summary...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Prepare summary data
summary_data = []
for genre in GENRES:
    hist = genre_history[genre]
    forecast = forecasts[genre]

    hist_mean = hist['popularity'].mean()
    forecast_mean = forecast['forecast_popularity'].mean()
    change = forecast_mean - hist_mean
    change_pct = (change / hist_mean) * 100

    # Get last historical and first forecast
    last_hist = hist.iloc[-1]['popularity']
    first_forecast = forecast.iloc[0]['forecast_popularity']
    immediate_change = first_forecast - last_hist

    summary_data.append([
        GENRE_DISPLAY_NAMES[genre],
        f"{hist_mean:.2f}",
        f"{forecast_mean:.2f}",
        f"{change:+.2f}",
        f"{change_pct:+.2f}%",
        f"{last_hist:.2f}",
        f"{first_forecast:.2f}",
        f"{immediate_change:+.2f}"
    ])

headers = ['Genre', 'Historical\nMean', 'Forecast\nMean', 'Change', 'Change %',
           '2024\nActual', '2025\nForecast', '1-Year\nChange']

table = ax.table(cellText=summary_data, colLabels=headers,
                cellLoc='center', loc='center',
                colColours=['#E8E8E8']*8)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color code the genre column
for i in range(len(GENRES)):
    table[(i+1, 0)].set_facecolor(GENRE_COLORS[GENRES[i]])
    table[(i+1, 0)].set_text_props(weight='bold', color='white')

# Bold headers
for i in range(len(headers)):
    table[(0, i)].set_text_props(weight='bold')

plt.title('Genre Popularity Forecast Summary: Historical vs. Predicted Trends',
         fontsize=14, fontweight='bold', pad=20)

output_file = OUTPUT_DIR / "forecast_summary_table.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_file.name}")
plt.close()

print("\n" + "="*60)
print("CHART GENERATION COMPLETE")
print("="*60)
print(f"\nAll charts saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  - 4 individual genre forecast charts")
print("  - 1 comprehensive 4-panel dashboard")
print("  - 1 overlay comparison chart")
print("  - 1 forecast comparison bar chart")
print("  - 1 forecast summary table")
print(f"\nTotal: 8 charts generated successfully!")

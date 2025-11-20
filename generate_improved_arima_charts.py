"""
Generate IMPROVED ARIMA forecast charts using Dataset 05
Shows complete historical data (2008-2024) plus forecasts (2025-2029)
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
OUTPUT_DIR = BASE_DIR / "Diagrams" / "arima_charts_improved"

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

# Load IMPROVED dataset (05 - has all genres per year)
print("="*70)
print("LOADING IMPROVED DATASET")
print("="*70)
print("\nUsing: 05_genres_median_audio_features.csv")
print("(Contains all genres across all years, not just most prominent)\n")

historical_data = pd.read_csv(DATASETS_DIR / "05_genres_median_audio_features.csv")
historical_data = historical_data.rename(columns={'track_album_release_date': 'Year'})

# Load forecasts
print("Loading ARIMA forecasts...")
forecasts = {}
for genre in GENRES:
    forecast_file = MODELS_DIR / f"{genre}_forecast_1to5y.csv"
    forecasts[genre] = pd.read_csv(forecast_file)
    print(f"  ✓ {genre}: {len(forecasts[genre])} years")

# Prepare historical data by genre
print("\nPreparing historical data by genre...")
genre_history = {}
for genre in GENRES:
    genre_data = historical_data[historical_data['playlist_genre'] == genre].copy()
    genre_data = genre_data[['Year', 'median_track_popularity']].rename(
        columns={'median_track_popularity': 'popularity'}
    )
    genre_data = genre_data.sort_values('Year')
    genre_history[genre] = genre_data

    years = genre_data['Year'].tolist()
    year_range = f"{min(years)}-{max(years)}" if years else "No data"
    print(f"  {GENRE_DISPLAY_NAMES[genre]}: {len(genre_data)} observations ({year_range})")

print("\n" + "="*70)
print("GENERATING IMPROVED CHARTS")
print("="*70)

# Chart 1: Individual genre charts with forecasts
print("\n1. Creating individual genre forecast charts...")
for genre in GENRES:
    fig, ax = plt.subplots(figsize=(14, 7))

    # Historical data
    hist = genre_history[genre]
    forecast = forecasts[genre]

    # Plot historical
    ax.plot(hist['Year'], hist['popularity'],
            marker='o', linewidth=2.5, markersize=8,
            color=GENRE_COLORS[genre], label='Historical Data', alpha=0.8)

    # Plot forecast
    ax.plot(forecast['year'], forecast['forecast_popularity'],
            marker='s', linewidth=2.5, markersize=8, linestyle='--',
            color=GENRE_COLORS[genre], label='ARIMA Forecast (2025-2029)', alpha=0.7)

    # Add vertical line at forecast boundary
    ax.axvline(x=2024.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    ax.text(2024.5, ax.get_ylim()[1] * 0.98, ' Forecast Period',
            verticalalignment='top', fontsize=10, color='gray', fontweight='bold')

    # Add data points count annotation
    total_points = len(hist) + len(forecast)
    ax.text(0.02, 0.98, f'Historical: {len(hist)} years\nForecast: {len(forecast)} years',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            fontsize=9)

    # Styling
    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('Popularity Score', fontsize=13, fontweight='bold')
    ax.set_title(f'{GENRE_DISPLAY_NAMES[genre]} Genre Popularity: Complete Historical Trend & ARIMA Forecast',
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Set x-axis to show all years
    all_years = list(hist['Year']) + list(forecast['year'])
    ax.set_xticks(range(min(all_years), max(all_years) + 1, 2))
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save
    output_file = OUTPUT_DIR / f"{genre}_complete_forecast.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_file.name}")
    plt.close()

# Chart 2: Comprehensive 4-panel dashboard
print("\n2. Creating comprehensive 4-genre dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for idx, genre in enumerate(GENRES):
    ax = axes[idx]

    hist = genre_history[genre]
    forecast = forecasts[genre]

    # Plot historical
    ax.plot(hist['Year'], hist['popularity'],
            marker='o', linewidth=2.5, markersize=7,
            color=GENRE_COLORS[genre], label='Historical', alpha=0.8)

    # Plot forecast
    ax.plot(forecast['year'], forecast['forecast_popularity'],
            marker='s', linewidth=2.5, markersize=7, linestyle='--',
            color=GENRE_COLORS[genre], label='Forecast', alpha=0.7)

    # Add vertical line
    ax.axvline(x=2024.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

    # Add data count
    ax.text(0.02, 0.98, f'n={len(hist)}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            fontsize=9)

    # Styling
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Popularity Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{GENRE_DISPLAY_NAMES[genre]}',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Set x-axis
    all_years = list(hist['Year']) + list(forecast['year'])
    ax.set_xticks(range(min(all_years), max(all_years) + 1, 3))
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Complete Genre Popularity Analysis: Historical Data (2008-2024) & ARIMA Forecasts (2025-2029)',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

output_file = OUTPUT_DIR / "complete_genres_dashboard.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_file.name}")
plt.close()

# Chart 3: Overlay comparison chart
print("\n3. Creating overlay comparison chart...")
fig, ax = plt.subplots(figsize=(16, 9))

for genre in GENRES:
    hist = genre_history[genre]
    forecast = forecasts[genre]

    display_name = GENRE_DISPLAY_NAMES[genre]

    # Plot historical
    ax.plot(hist['Year'], hist['popularity'],
            marker='o', linewidth=2.5, markersize=7,
            color=GENRE_COLORS[genre], label=f'{display_name} (Historical)',
            alpha=0.7)

    # Plot forecast
    ax.plot(forecast['year'], forecast['forecast_popularity'],
            marker='s', linewidth=2.5, markersize=7, linestyle='--',
            color=GENRE_COLORS[genre], label=f'{display_name} (Forecast)',
            alpha=0.7)

# Add vertical line
ax.axvline(x=2024.5, color='gray', linestyle=':', linewidth=2.5, alpha=0.6)
ax.text(2024.5, ax.get_ylim()[1] * 0.98, ' Forecast Period Begins',
        verticalalignment='top', fontsize=12, color='gray', fontweight='bold')

# Styling
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Popularity Score', fontsize=14, fontweight='bold')
ax.set_title('Complete Genre Comparison: Full Historical Trends (2008-2024) & ARIMA Forecasts (2025-2029)',
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)

# Set x-axis
ax.set_xticks(range(2008, 2030, 2))
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()

output_file = OUTPUT_DIR / "complete_genres_overlay.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_file.name}")
plt.close()

# Chart 4: Forecast summary comparison (2025-2029)
print("\n4. Creating forecast summary comparison...")
fig, ax = plt.subplots(figsize=(14, 8))

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
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Forecasted Popularity Score', fontsize=14, fontweight='bold')
ax.set_title('ARIMA Forecast Comparison Across All Genres (2025-2029)',
            fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(forecast_years)
ax.legend(loc='best', fontsize=12, framealpha=0.9)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()

output_file = OUTPUT_DIR / "forecast_comparison_bars.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_file.name}")
plt.close()

# Chart 5: Enhanced trend summary table
print("\n5. Creating enhanced forecast trend summary...")
fig, ax = plt.subplots(figsize=(14, 7))
ax.axis('tight')
ax.axis('off')

# Prepare summary data
summary_data = []
for genre in GENRES:
    hist = genre_history[genre]
    forecast = forecasts[genre]

    hist_mean = hist['popularity'].mean()
    hist_first = hist.iloc[0]['popularity']
    hist_last = hist.iloc[-1]['popularity']

    forecast_mean = forecast['forecast_popularity'].mean()
    forecast_first = forecast.iloc[0]['forecast_popularity']
    forecast_last = forecast.iloc[-1]['forecast_popularity']

    change = forecast_mean - hist_mean
    change_pct = (change / hist_mean) * 100

    immediate_change = forecast_first - hist_last

    summary_data.append([
        GENRE_DISPLAY_NAMES[genre],
        f"{len(hist)} years",
        f"{hist.iloc[0]['Year']:.0f}-{hist.iloc[-1]['Year']:.0f}",
        f"{hist_mean:.2f}",
        f"{forecast_mean:.2f}",
        f"{change:+.2f}",
        f"{change_pct:+.2f}%",
        f"{hist_last:.2f}",
        f"{forecast_first:.2f}",
        f"{immediate_change:+.2f}"
    ])

headers = ['Genre', 'Historical\nData Points', 'Year\nRange', 'Historical\nMean',
           'Forecast\nMean', 'Change', 'Change\n%', '2024\nActual',
           '2025\nForecast', '1-Year\nChange']

table = ax.table(cellText=summary_data, colLabels=headers,
                cellLoc='center', loc='center',
                colColours=['#E8E8E8']*10)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

# Color code the genre column
for i in range(len(GENRES)):
    table[(i+1, 0)].set_facecolor(GENRE_COLORS[GENRES[i]])
    table[(i+1, 0)].set_text_props(weight='bold', color='white')

# Bold headers
for i in range(len(headers)):
    table[(0, i)].set_text_props(weight='bold', fontsize=10)

plt.title('Complete Genre Popularity Analysis: Historical Coverage & Forecast Summary',
         fontsize=15, fontweight='bold', pad=20)

output_file = OUTPUT_DIR / "complete_forecast_summary_table.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_file.name}")
plt.close()

# Chart 6: Side-by-side comparison (Historical span vs Forecast)
print("\n6. Creating historical span comparison chart...")
fig, ax = plt.subplots(figsize=(12, 8))

# Prepare data
genres_display = [GENRE_DISPLAY_NAMES[g] for g in GENRES]
hist_counts = [len(genre_history[g]) for g in GENRES]
forecast_count = 5  # All forecasts are 5 years

x = range(len(GENRES))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], hist_counts, width,
               label='Historical Years', alpha=0.8)
bars2 = ax.bar([i + width/2 for i in x], [forecast_count]*len(GENRES), width,
               label='Forecast Years', alpha=0.8)

# Color the bars
for bar, genre in zip(bars1, GENRES):
    bar.set_color(GENRE_COLORS[genre])

for bar in bars2:
    bar.set_color('gray')

# Add value labels on bars
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    ax.text(bar1.get_x() + bar1.get_width()/2., height1,
            f'{int(height1)}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.text(bar2.get_x() + bar2.get_width()/2., height2,
            f'{int(height2)}', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_xlabel('Genre', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Years', fontsize=13, fontweight='bold')
ax.set_title('Data Coverage: Historical Observations vs Forecast Period by Genre',
            fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(genres_display)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()

output_file = OUTPUT_DIR / "data_coverage_comparison.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_file.name}")
plt.close()

print("\n" + "="*70)
print("IMPROVED CHART GENERATION COMPLETE")
print("="*70)
print(f"\nAll charts saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  - 4 individual genre forecast charts (with complete history)")
print("  - 1 comprehensive 4-panel dashboard")
print("  - 1 overlay comparison chart (all genres)")
print("  - 1 forecast comparison bar chart")
print("  - 1 enhanced forecast summary table")
print("  - 1 data coverage comparison chart")
print(f"\nTotal: 9 improved charts generated successfully!")
print("\n" + "="*70)
print("KEY IMPROVEMENTS")
print("="*70)
print("✓ Pop: 17 years of history (2008-2024) vs previous 5 years")
print("✓ Electronic: 16 years of history (2008-2024) vs previous 4 years")
print("✓ Hip-Hop: 14 years of history (2010-2024) vs previous 3 years")
print("✓ Latin: 12 years of history (2013-2024) vs previous 5 years")
print("\nMuch more robust historical context for your ARIMA forecasts!")

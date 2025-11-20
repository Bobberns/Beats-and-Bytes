"""
Analyze genre coverage across both datasets to understand data availability
"""

import pandas as pd

# Load both datasets
genres_median = pd.read_csv("Datasets/05_genres_median_audio_features.csv")
aggregated = pd.read_csv("Datasets/06_aggregated_dataset.csv")

print("="*70)
print("DATASET COMPARISON ANALYSIS")
print("="*70)

print("\n1. DATASET: 05_genres_median_audio_features.csv")
print("   (All genres with data per year)")
print("-"*70)

# Rename column for consistency
genres_median = genres_median.rename(columns={'track_album_release_date': 'Year'})

for genre in ['pop', 'hip-hop', 'latin', 'electronic']:
    genre_data = genres_median[genres_median['playlist_genre'] == genre]
    years = sorted(genre_data['Year'].unique())
    count = len(years)
    year_range = f"{min(years)}-{max(years)}" if years else "No data"

    print(f"\n{genre.upper()}:")
    print(f"  Years: {years}")
    print(f"  Count: {count} observations")
    print(f"  Range: {year_range}")

print("\n" + "="*70)
print("2. DATASET: 06_aggregated_dataset.csv")
print("   (Only most prominent genre per year)")
print("-"*70)

for genre in ['pop', 'hip-hop', 'latin', 'electronic']:
    genre_data = aggregated[aggregated['Most_Prominent_Genre'] == genre]
    years = sorted(genre_data['Year'].unique())
    count = len(years)
    year_range = f"{min(years)}-{max(years)}" if years else "No data"

    print(f"\n{genre.upper()}:")
    print(f"  Years: {years}")
    print(f"  Count: {count} observations")
    print(f"  Range: {year_range}")

print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)

print(f"\n{'Genre':<15} {'Dataset 05':<20} {'Dataset 06':<20} {'Gain'}")
print("-"*70)

for genre in ['pop', 'hip-hop', 'latin', 'electronic']:
    count_05 = len(genres_median[genres_median['playlist_genre'] == genre])
    count_06 = len(aggregated[aggregated['Most_Prominent_Genre'] == genre])
    gain = count_05 - count_06

    print(f"{genre.capitalize():<15} {count_05:<20} {count_06:<20} +{gain}")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("""
Dataset 05 (genres_median_audio_features.csv) provides MORE observations
per genre and should be used for ARIMA modeling to improve forecast accuracy.

Current charts use Dataset 06, which only has 3-5 observations per genre.
Dataset 05 would provide 10-17 observations per genre - MUCH better for
time series modeling!
""")

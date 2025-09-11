import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data from the XGBoost model results
pop_features = {
    'median_energy': 0.676,
    'median_danceability': 0.266,
    'median_speechiness': 0.051
}

latin_features = {
    'median_valence': 0.300,
    'median_danceability': 0.205,
    'median_loudness': 0.177
}

# Create a combined dataset for visualization
all_features = set(pop_features.keys()) | set(latin_features.keys())
pop_values = [pop_features.get(feature, 0) for feature in all_features]
latin_values = [latin_features.get(feature, 0) for feature in all_features]

# Create the visualization
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(all_features))
width = 0.35

bars1 = ax.bar(x - width/2, pop_values, width, label='Pop', color='blue', alpha=0.7)
bars2 = ax.bar(x + width/2, latin_values, width, label='Latin', color='orange', alpha=0.7)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    if height > 0:
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

for bar in bars2:
    height = bar.get_height()
    if height > 0:
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

ax.set_xlabel('Audio Features')
ax.set_ylabel('Feature Importance')
ax.set_title('XGBoost Feature Importance Comparison: Pop vs Latin Genres')
ax.set_xticks(x)
ax.set_xticklabels(all_features, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig('Diagrams/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Feature importance comparison chart saved to 'Diagrams/feature_importance_comparison.png'")
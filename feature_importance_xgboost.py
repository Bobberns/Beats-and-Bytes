import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('Datasets/05_genres_median_audio_features.csv')

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Prepare the data for modeling
# We'll predict median_track_popularity based on audio features
# Convert track_album_release_date to just the year
df['year'] = df['track_album_release_date']

# Select features for modeling
feature_columns = [
    'median_energy', 'median_tempo', 'median_danceability', 'median_loudness',
    'median_liveness', 'median_valence', 'median_speechiness', 
    'median_instrumentalness', 'median_acousticness'
]

# Target variable
target_column = 'median_track_popularity'

# Prepare X and y
X = df[feature_columns]
y = df[target_column]

print("\nFeature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Create and train the XGBoost model
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    random_state=42,
    learning_rate=0.1
)

print("\nTraining the XGBoost model...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Extract feature importance
feature_importance = model.feature_importances_
feature_names = feature_columns

# Create a DataFrame for easier handling
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values(by='importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)

# Visualize feature importance
plt.figure(figsize=(12, 8))
sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
plt.title('XGBoost Feature Importance for Genre Popularity Prediction', fontsize=16)
plt.xlabel('Feature Importance Score', fontsize=14)
plt.ylabel('Audio Features', fontsize=14)
plt.tight_layout()

# Save the plot
plt.savefig('Diagrams/feature_importance_xgboost.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a more detailed visualization with values
plt.figure(figsize=(14, 10))
bars = plt.barh(importance_df['feature'], importance_df['importance'], color='skyblue')
plt.xlabel('Feature Importance Score', fontsize=14)
plt.ylabel('Audio Features', fontsize=14)
plt.title('XGBoost Feature Importance for Genre Popularity Prediction', fontsize=16)
plt.gca().invert_yaxis()  # Highest importance at the top

# Add value labels on bars
for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{importance:.3f}', 
             ha='left', va='center', fontsize=12)

plt.tight_layout()
plt.savefig('Diagrams/feature_importance_xgboost_detailed.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFeature importance visualization saved to 'Diagrams/feature_importance_xgboost.png' and 'Diagrams/feature_importance_xgboost_detailed.png'")
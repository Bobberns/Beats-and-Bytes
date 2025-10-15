#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fixed XGBoost Training Pipeline for Music Genre Popularity Prediction
==================================================================

This script fixes the original XGBoost implementation by:
- Using individual track data instead of aggregated yearly data (652 tracks vs 17 rows)
- Implementing proper time-aware validation for sufficient sample sizes
- Adding data validation and minimum sample size checks
- Using appropriate train/test splits for machine learning
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, train_test_split
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Configuration
YEAR_MIN, YEAR_MAX = 2008, 2024
RANDOM_STATE = 42
MIN_SAMPLES_PER_SPLIT = 20  # Minimum samples required for train/test splits

# Thesis thresholds for model evaluation
THRESH_RMSE, THRESH_MAE, THRESH_R2 = 8.0, 6.0, 0.70

FOCUS_GENRES = ["pop", "hip-hop", "latin", "electronic"]

# Audio features used for prediction (original track-level features)
FEATURES = [
    "energy",
    "tempo",
    "danceability",
    "loudness",
    "liveness",
    "valence",
    "speechiness",
    "instrumentalness",
    "acousticness"
]

TARGET = "track_popularity"

# Optimized hyperparameters for track-level data
MANUAL_PARAMS = {
    "pop": {
        "max_depth": 6, "learning_rate": 0.1, "n_estimators": 100,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0
    },
    "hip-hop": {
        "max_depth": 6, "learning_rate": 0.1, "n_estimators": 100,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0
    },
    "latin": {
        "max_depth": 6, "learning_rate": 0.1, "n_estimators": 100,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0
    },
    "electronic": {
        "max_depth": 6, "learning_rate": 0.1, "n_estimators": 100,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0
    }
}

# Grid for hyperparameter tuning (reduced for efficiency)
GRID = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [50, 100, 200],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda": [0.0, 1.0],
    "reg_alpha": [0.0, 0.1]
}

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance using standard metrics.

    Parameters:
    y_true (array): Actual values
    y_pred (array): Predicted values

    Returns:
    dict: Dictionary containing evaluation metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def train_xgboost_model(X_train, y_train, params):
    """
    Train an XGBoost model with specified parameters.

    Parameters:
    X_train (DataFrame): Training features
    y_train (Series): Training target
    params (dict): Model parameters

    Returns:
    XGBRegressor: Trained model
    """
    model = XGBRegressor(
        random_state=RANDOM_STATE,
        **params
    )
    model.fit(X_train, y_train)
    return model

def time_aware_split(df, test_size=0.2, validation_size=0.1):
    """
    Create time-aware train/validation/test splits.

    Parameters:
    df (DataFrame): Input data with 'track_album_release_date' column
    test_size (float): Proportion for test set
    validation_size (float): Proportion for validation set

    Returns:
    tuple: (train_df, val_df, test_df)
    """
    # Sort by year to ensure temporal order
    df_sorted = df.sort_values('track_album_release_date').reset_index(drop=True)

    n_samples = len(df_sorted)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * validation_size)
    n_train = n_samples - n_test - n_val

    train_df = df_sorted.iloc[:n_train]
    val_df = df_sorted.iloc[n_train:n_train + n_val]
    test_df = df_sorted.iloc[n_train + n_val:]

    return train_df, val_df, test_df

def hyperparameter_tuning(X_train, y_train, X_val, y_val, grid, max_evals=50):
    """
    Perform hyperparameter tuning using validation set.

    Parameters:
    X_train, y_train: Training data
    X_val, y_val: Validation data
    grid (dict): Parameter grid for tuning
    max_evals (int): Maximum evaluations to prevent overly long searches

    Returns:
    dict: Best parameters
    """
    best_params = None
    best_rmse = float('inf')

    param_combinations = list(ParameterGrid(grid))
    # Limit evaluations for efficiency
    param_combinations = param_combinations[:max_evals]

    print(f"    Evaluating {len(param_combinations)} parameter combinations...")

    for i, params in enumerate(param_combinations):
        try:
            model = train_xgboost_model(X_train, y_train, params)
            y_pred = model.predict(X_val)
            metrics = evaluate_model(y_val, y_pred)

            if metrics['rmse'] < best_rmse:
                best_rmse = metrics['rmse']
                best_params = params.copy()

            if (i + 1) % 10 == 0:
                print(f"    Completed {i + 1}/{len(param_combinations)} evaluations")

        except Exception as e:
            print(f"    Error with params {params}: {e}")
            continue

    return best_params

def process_genre(genre_df, genre_name, manual_params, grid, output_dir):
    """
    Process a single genre with proper validation.

    Parameters:
    genre_df (DataFrame): Data for a specific genre
    genre_name (str): Name of the genre
    manual_params (dict): Manual baseline parameters
    grid (dict): Parameter grid for tuning
    output_dir (str): Output directory

    Returns:
    dict: Results dictionary
    """
    print(f"Processing {genre_name}...")
    print(f"  Total samples: {len(genre_df)}")

    # Check if we have enough data
    if len(genre_df) < MIN_SAMPLES_PER_SPLIT * 3:  # Need enough for train/val/test
        print(f"  Insufficient data ({len(genre_df)} samples). Minimum required: {MIN_SAMPLES_PER_SPLIT * 3}")
        return None

    # Time-aware split
    train_df, val_df, test_df = time_aware_split(genre_df)

    print(f"  Train: {len(train_df)} samples ({train_df['track_album_release_date'].min():.0f}-{train_df['track_album_release_date'].max():.0f})")
    print(f"  Validation: {len(val_df)} samples ({val_df['track_album_release_date'].min():.0f}-{val_df['track_album_release_date'].max():.0f})")
    print(f"  Test: {len(test_df)} samples ({test_df['track_album_release_date'].min():.0f}-{test_df['track_album_release_date'].max():.0f})")

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_val = val_df[FEATURES]
    y_val = val_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    # Train with manual parameters
    print("  Training with manual parameters...")
    manual_model = train_xgboost_model(X_train, y_train, manual_params)
    y_manual_pred = manual_model.predict(X_val)
    manual_metrics = evaluate_model(y_val, y_manual_pred)

    print(f"  Manual params validation RMSE: {manual_metrics['rmse']:.4f}")

    # Hyperparameter tuning
    print("  Performing hyperparameter tuning...")
    best_params = hyperparameter_tuning(X_train, y_train, X_val, y_val, grid)

    if best_params:
        tuned_model = train_xgboost_model(X_train, y_train, best_params)
        y_tuned_pred = tuned_model.predict(X_val)
        tuned_metrics = evaluate_model(y_val, y_tuned_pred)

        print(f"  Tuned params validation RMSE: {tuned_metrics['rmse']:.4f}")

        # Select best model
        if tuned_metrics['rmse'] < manual_metrics['rmse']:
            final_params = best_params
            final_model = tuned_model
            print("  Selected tuned parameters")
        else:
            final_params = manual_params
            final_model = manual_model
            print("  Selected manual parameters")
    else:
        final_params = manual_params
        final_model = manual_model
        print("  Using manual parameters (tuning failed)")

    # Final evaluation on test set
    y_test_pred = final_model.predict(X_test)
    test_metrics = evaluate_model(y_test, y_test_pred)

    print(f"  Final test RMSE: {test_metrics['rmse']:.4f}")
    print(f"  Final test MAE: {test_metrics['mae']:.4f}")
    print(f"  Final test R²: {test_metrics['r2']:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': FEATURES,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Save results
    feature_importance.to_csv(
        os.path.join(output_dir, f'{genre_name}_feature_importance.csv'),
        index=False
    )

    # Save test predictions
    test_results = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_test_pred,
        'year': test_df['track_album_release_date'].values
    })
    test_results.to_csv(
        os.path.join(output_dir, f'{genre_name}_test_predictions.csv'),
        index=False
    )

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    bars = plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Feature Importance')
    plt.title(f'{genre_name.title()} - XGBoost Feature Importance')
    plt.gca().invert_yaxis()

    # Add value labels
    for bar, importance in zip(bars, feature_importance['importance']):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}', ha='left', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{genre_name}_feature_importance.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'genre': genre_name,
        'n_samples': len(genre_df),
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_test': len(test_df),
        'final_params': final_params,
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_r2': test_metrics['r2'],
        'meets_thresholds': (
            test_metrics['rmse'] <= THRESH_RMSE and
            test_metrics['mae'] <= THRESH_MAE and
            test_metrics['r2'] >= THRESH_R2
        ),
        'top_features': feature_importance.head(3)['feature'].tolist()
    }

def main():
    """
    Main function to execute the fixed XGBoost pipeline.
    """
    # Create output directory
    output_dir = os.path.join(os.getcwd(), 'outputs', 'xgb_fixed')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print("="*60)

    # Load individual track data
    try:
        df = pd.read_csv(os.path.join('..', 'Datasets', '03_four_genres_spotify_data.csv'))
        print(f"Loaded {len(df)} individual tracks")
    except FileNotFoundError:
        print("Error: Track dataset not found. Please ensure '03_four_genres_spotify_data.csv' exists.")
        return

    # Data validation and preparation
    print("\nData validation:")
    print(f"  Original samples: {len(df)}")
    print(f"  Year range: {df['track_album_release_date'].min():.0f} - {df['track_album_release_date'].max():.0f}")
    print(f"  Genres: {df['playlist_genre'].value_counts().to_dict()}")

    # Filter for focus genres and time range
    df = df[df['playlist_genre'].isin(FOCUS_GENRES)]
    df = df[(df['track_album_release_date'] >= YEAR_MIN) & (df['track_album_release_date'] <= YEAR_MAX)]
    df = df.dropna(subset=FEATURES + [TARGET])

    print(f"  After filtering: {len(df)} samples")
    print(f"  Genre distribution: {df['playlist_genre'].value_counts().to_dict()}")

    # Process each genre
    results = []

    for genre in FOCUS_GENRES:
        print(f"\n{'='*60}")
        genre_df = df[df['playlist_genre'] == genre].copy()

        if len(genre_df) == 0:
            print(f"No data for {genre}")
            continue

        result = process_genre(
            genre_df, genre, MANUAL_PARAMS[genre], GRID, output_dir
        )

        if result:
            results.append(result)

    # Create summary report
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY RESULTS")
        print("="*60)

        summary_df = pd.DataFrame(results)
        summary_df.to_csv(os.path.join(output_dir, 'xgb_fixed_summary.csv'), index=False)

        for _, row in summary_df.iterrows():
            print(f"\n{row['genre'].upper()}:")
            print(f"  Samples: {row['n_samples']} (train: {row['n_train']}, val: {row['n_val']}, test: {row['n_test']})")
            print(f"  Test RMSE: {row['test_rmse']:.4f}")
            print(f"  Test MAE: {row['test_mae']:.4f}")
            print(f"  Test R²: {row['test_r2']:.4f}")
            print(f"  Meets thresholds: {row['meets_thresholds']}")
            print(f"  Top features: {', '.join(row['top_features'])}")

        # Overall statistics
        avg_rmse = summary_df['test_rmse'].mean()
        avg_mae = summary_df['test_mae'].mean()
        avg_r2 = summary_df['test_r2'].mean()

        print(f"\nOVERALL AVERAGES:")
        print(f"  Average RMSE: {avg_rmse:.4f}")
        print(f"  Average MAE: {avg_mae:.4f}")
        print(f"  Average R²: {avg_r2:.4f}")
        print(f"  Genres meeting thresholds: {summary_df['meets_thresholds'].sum()}/{len(summary_df)}")

        print(f"\nAll results saved to: {output_dir}")

    else:
        print("No results generated. Check data and model configurations.")

if __name__ == "__main__":
    main()
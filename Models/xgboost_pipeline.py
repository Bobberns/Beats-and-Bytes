#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XGBoost Training Pipeline for Music Genre Popularity Forecasting
==============================================================

This script implements the XGBoost training pipeline for forecasting music genre 
popularity based on audio features, following the thesis methodology of:
- Using aggregated yearly genre data (2008-2024)
- Time-aware cross-validation with walk-forward approach
- Feature importance analysis for identifying influential audio characteristics
- Hyperparameter tuning for optimal model performance
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Configuration
YEAR_MIN, YEAR_MAX = 2008, 2024
TRAIN_YEARS, TEST_YEARS = 3, 2
RANDOM_STATE = 42

# Thesis thresholds for model evaluation
THRESH_RMSE, THRESH_MAE, THRESH_R2 = 8.0, 5.0, 0.70

FOCUS_GENRES = ["pop", "hip-hop", "latin", "electronic"]

# Audio features used for prediction
FEATURES = [
    "median_energy",
    "median_tempo", 
    "median_danceability",
    "median_loudness",
    "median_liveness",
    "median_valence",
    "median_speechiness",
    "median_instrumentalness",
    "median_acousticness"
]

TARGET = "median_track_popularity"

# Manual baseline hyperparameters per genre
MANUAL_PARAMS = {
    "pop": {
        "max_depth": 4, "learning_rate": 0.08, "n_estimators": 300,
        "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0, "reg_alpha": 0.0
    },
    "hip-hop": {
        "max_depth": 4, "learning_rate": 0.08, "n_estimators": 300,
        "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0, "reg_alpha": 0.0
    },
    "latin": {
        "max_depth": 4, "learning_rate": 0.08, "n_estimators": 300,
        "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0, "reg_alpha": 0.0
    },
    "electronic": {
        "max_depth": 4, "learning_rate": 0.08, "n_estimators": 300,
        "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0, "reg_alpha": 0.0
    }
}

# Grid for hyperparameter tuning (108 combinations)
GRID = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [50, 100, 200],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

def generate_time_windows(years, train_years=3, test_years=2):
    """
    Generate all contiguous time windows for walk-forward validation.
    
    Parameters:
    years (list): List of available years
    train_years (int): Number of years for training
    test_years (int): Number of years for testing
    
    Returns:
    list: List of tuples containing (train_years, test_years)
    """
    windows = []
    total = train_years + test_years
    
    for i in range(len(years) - total + 1):
        train_set = years[i:i+train_years]
        test_set = years[i+train_years:i+total]
        
        # Check if years are contiguous
        if (all(np.diff(train_set) == 1) and 
            all(np.diff(test_set) == 1) and 
            train_set[-1] + 1 == test_set[0]):
            windows.append((train_set, test_set))
            
    return windows

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

def hyperparameter_tuning(genre_df, train_years, test_years, grid):
    """
    Perform hyperparameter tuning using grid search.
    
    Parameters:
    genre_df (DataFrame): Data for a specific genre
    train_years (list): Training years
    test_years (list): Test years
    grid (dict): Parameter grid for tuning
    
    Returns:
    dict: Best parameters
    """
    # Split data
    train_data = genre_df[genre_df['Year'].isin(train_years)]
    test_data = genre_df[genre_df['Year'].isin(test_years)]
    
    if len(train_data) == 0 or len(test_data) == 0:
        return None
    
    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]
    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]
    
    best_params = None
    best_rmse = float('inf')
    
    # Grid search
    for params in ParameterGrid(grid):
        try:
            model = train_xgboost_model(X_train, y_train, params)
            y_pred = model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred)
            
            if metrics['rmse'] < best_rmse:
                best_rmse = metrics['rmse']
                best_params = params.copy()
        except Exception:
            continue
            
    return best_params

def walk_forward_validation(genre_df, genre_name, manual_params, grid):
    """
    Perform walk-forward validation for a specific genre.
    
    Parameters:
    genre_df (DataFrame): Data for a specific genre
    genre_name (str): Name of the genre
    manual_params (dict): Manual baseline parameters
    grid (dict): Parameter grid for tuning
    
    Returns:
    tuple: (metrics_df, candidates_df, feature_importance_list)
    """
    years = sorted(genre_df['Year'].unique())
    windows = generate_time_windows(years, TRAIN_YEARS, TEST_YEARS)
    
    metrics_records = []
    candidates_records = []
    feature_importance_list = []
    
    print(f"Performing walk-forward validation for {genre_name}...")
    print(f"Found {len(windows)} time windows for validation")
    
    for fold_idx, (train_years, test_years) in enumerate(windows, 1):
        print(f"  Processing fold {fold_idx}/{len(windows)}: Train {train_years} -> Test {test_years}")
        
        # Split data
        train_data = genre_df[genre_df['Year'].isin(train_years)]
        test_data = genre_df[genre_df['Year'].isin(test_years)]
        
        if len(train_data) == 0 or len(test_data) == 0:
            continue
            
        X_train = train_data[FEATURES]
        y_train = train_data[TARGET]
        X_test = test_data[FEATURES]
        y_test = test_data[TARGET]
        
        # Train with manual parameters
        manual_model = train_xgboost_model(X_train, y_train, manual_params)
        y_manual_pred = manual_model.predict(X_test)
        manual_metrics = evaluate_model(y_test, y_manual_pred)
        
        # Record manual parameters results
        candidates_records.append({
            'genre': genre_name,
            'fold': fold_idx,
            'train_start': train_years[0],
            'train_end': train_years[-1],
            'test_start': test_years[0],
            'test_end': test_years[-1],
            'rmse': manual_metrics['rmse'],
            'mae': manual_metrics['mae'],
            'r2': manual_metrics['r2'],
            'source': 'manual'
        } | manual_params)
        
        # Hyperparameter tuning
        best_params = hyperparameter_tuning(genre_df, train_years, test_years, grid)
        
        if best_params:
            # Train with best parameters
            tuned_model = train_xgboost_model(X_train, y_train, best_params)
            y_tuned_pred = tuned_model.predict(X_test)
            tuned_metrics = evaluate_model(y_test, y_tuned_pred)
            
            # Record tuned parameters results
            candidates_records.append({
                'genre': genre_name,
                'fold': fold_idx,
                'train_start': train_years[0],
                'train_end': train_years[-1],
                'test_start': test_years[0],
                'test_end': test_years[-1],
                'rmse': tuned_metrics['rmse'],
                'mae': tuned_metrics['mae'],
                'r2': tuned_metrics['r2'],
                'source': 'grid'
            } | best_params)
            
            # Select best model based on RMSE
            if tuned_metrics['rmse'] < manual_metrics['rmse']:
                final_metrics = tuned_metrics
                final_params = best_params
                final_predictions = y_tuned_pred
            else:
                final_metrics = manual_metrics
                final_params = manual_params
                final_predictions = y_manual_pred
        else:
            final_metrics = manual_metrics
            final_params = manual_params
            final_predictions = y_manual_pred
        
        # Record final fold metrics
        metrics_records.append({
            'genre': genre_name,
            'fold': fold_idx,
            'train_start': train_years[0],
            'train_end': train_years[-1],
            'test_start': test_years[0],
            'test_end': test_years[-1],
            'rmse': final_metrics['rmse'],
            'mae': final_metrics['mae'],
            'r2': final_metrics['r2']
        } | final_params)
        
        # Record feature importance
        final_model = train_xgboost_model(X_train, y_train, final_params)
        feature_importance = pd.DataFrame({
            'feature': FEATURES,
            'importance': final_model.feature_importances_,
            'fold': fold_idx,
            'genre': genre_name
        })
        feature_importance_list.append(feature_importance)
    
    # Convert to DataFrames
    metrics_df = pd.DataFrame(metrics_records)
    candidates_df = pd.DataFrame(candidates_records)
    
    if feature_importance_list:
        feature_importance_df = pd.concat(feature_importance_list, ignore_index=True)
    else:
        feature_importance_df = pd.DataFrame(columns=['feature', 'importance', 'fold', 'genre'])
    
    return metrics_df, candidates_df, feature_importance_df

def holdout_evaluation(genre_df, genre_name, final_params):
    """
    Evaluate model on holdout period (2023-2024).
    
    Parameters:
    genre_df (DataFrame): Data for a specific genre
    genre_name (str): Name of the genre
    final_params (dict): Final model parameters
    
    Returns:
    DataFrame: Holdout evaluation results
    """
    # Train on all data up to 2022
    train_data = genre_df[genre_df['Year'] <= 2022]
    holdout_data = genre_df[genre_df['Year'].isin([2023, 2024])]
    
    if len(train_data) == 0 or len(holdout_data) == 0:
        return None
    
    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]
    X_holdout = holdout_data[FEATURES]
    y_holdout = holdout_data[TARGET]
    
    # Train final model
    model = train_xgboost_model(X_train, y_train, final_params)
    y_pred = model.predict(X_holdout)
    
    # Evaluate
    metrics = evaluate_model(y_holdout, y_pred)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Year': holdout_data['Year'].values,
        'Actual': y_holdout.values,
        'Predicted': y_pred,
        'Residual': y_holdout.values - y_pred
    })
    
    return results, metrics

def create_summary_report(all_metrics, all_candidates, all_feature_importance):
    """
    Create a summary report of XGBoost model performance.
    
    Parameters:
    all_metrics (list): List of metrics DataFrames for all genres
    all_candidates (list): List of candidates DataFrames for all genres
    all_feature_importance (list): List of feature importance DataFrames for all genres
    
    Returns:
    DataFrame: Summary report
    """
    if not all_metrics:
        return pd.DataFrame()
    
    # Concatenate all data
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    candidates_df = pd.concat(all_candidates, ignore_index=True)
    feature_importance_df = pd.concat(all_feature_importance, ignore_index=True) if all_feature_importance else pd.DataFrame()
    
    # Calculate summary statistics by genre
    summary_stats = metrics_df.groupby('genre').agg({
        'rmse': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'r2': ['mean', 'std'],
        'fold': 'count'
    }).round(4)
    
    # Flatten column names
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
    summary_stats = summary_stats.reset_index()
    
    # Rename columns for clarity
    summary_stats.rename(columns={
        'rmse_mean': 'avg_rmse_walkforward',
        'rmse_std': 'std_rmse_walkforward',
        'mae_mean': 'avg_mae_walkforward',
        'mae_std': 'std_mae_walkforward',
        'r2_mean': 'avg_r2_walkforward',
        'r2_std': 'std_r2_walkforward',
        'fold_count': 'folds'
    }, inplace=True)
    
    # Determine final parameters (lowest average RMSE)
    final_params_dict = {}
    for genre in summary_stats['genre']:
        genre_metrics = metrics_df[metrics_df['genre'] == genre]
        avg_rmse_per_param_set = genre_metrics.groupby([
            'max_depth', 'learning_rate', 'n_estimators',
            'subsample', 'colsample_bytree', 'reg_lambda', 'reg_alpha'
        ])['rmse'].mean()
        
        best_params_idx = avg_rmse_per_param_set.idxmin()
        final_params_dict[genre] = {
            'max_depth': best_params_idx[0],
            'learning_rate': best_params_idx[1],
            'n_estimators': best_params_idx[2],
            'subsample': best_params_idx[3],
            'colsample_bytree': best_params_idx[4],
            'reg_lambda': best_params_idx[5],
            'reg_alpha': best_params_idx[6]
        }
    
    # Add final parameters to summary
    summary_stats['final_params'] = summary_stats['genre'].map(lambda x: str(final_params_dict.get(x, {})))
    
    # Add threshold compliance information
    summary_stats['meets_thresholds_walkforward'] = (
        (summary_stats['avg_rmse_walkforward'] <= THRESH_RMSE) & 
        (summary_stats['avg_mae_walkforward'] <= THRESH_MAE) & 
        (summary_stats['avg_r2_walkforward'] >= THRESH_R2)
    )
    
    return summary_stats, candidates_df, feature_importance_df

def plot_feature_importance(feature_importance_df, genre_name, output_dir):
    """
    Plot feature importance for a specific genre.
    
    Parameters:
    feature_importance_df (DataFrame): Feature importance data
    genre_name (str): Name of the genre
    output_dir (str): Output directory for plots
    """
    if feature_importance_df.empty:
        return
    
    # Calculate mean importance across folds
    mean_importance = feature_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=True)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(mean_importance)), mean_importance.values, color='#008dff')
    plt.yticks(range(len(mean_importance)), mean_importance.index)
    plt.xlabel('Mean Feature Importance')
    plt.title(f'{genre_name.title()} - Feature Importance (XGBoost)')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, mean_importance.values)):
        plt.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{genre_name}_xgb_feature_importance.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to execute the XGBoost training pipeline.
    """
    # Create output directory
    output_dir = os.path.join(os.getcwd(), 'outputs', 'xgb')
    os.makedirs(output_dir, exist_ok=True)
    diagnostics_dir = os.path.join(output_dir, 'diagnostics')
    os.makedirs(diagnostics_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Diagnostics directory: {diagnostics_dir}")
    
    print("Loading and preparing data...")
    
    # Load aggregated dataset
    try:
        df = pd.read_csv(os.path.join('..', 'Datasets', '06_aggregated_dataset.csv'))
    except FileNotFoundError:
        print("Error: Aggregated dataset not found. Please run data preprocessing first.")
        return
    
    # Filter for focus genres and time range
    df = df[df['Most_Prominent_Genre'].isin(FOCUS_GENRES)]
    df = df[(df['Year'] >= YEAR_MIN) & (df['Year'] <= YEAR_MAX)]
    
    # Rename column for consistency
    df = df.rename(columns={'Most_Prominent_Genre': 'genre'})
    
    print(f"Loaded {len(df)} records for {len(df['genre'].unique())} genres")
    
    # Store results for all genres
    all_metrics = []
    all_candidates = []
    all_feature_importance = []
    all_holdout_results = {}
    all_holdout_metrics = {}
    
    # Process each genre
    for genre in FOCUS_GENRES:
        print(f"\nProcessing {genre}...")
        
        genre_df = df[df['genre'] == genre].copy()
        genre_years = sorted(genre_df['Year'].unique())
        print(f"  Available years: {genre_years} ({len(genre_years)} years)")
        
        # Check if we have enough consecutive years for at least one fold
        min_years_needed = TRAIN_YEARS + TEST_YEARS
        has_consecutive_years = False
        
        # Check for any sequence of consecutive years of required length
        for i in range(len(genre_years) - min_years_needed + 1):
            window = genre_years[i:i + min_years_needed]
            if all(window[j] + 1 == window[j + 1] for j in range(len(window) - 1)):
                has_consecutive_years = True
                break
                
        if not has_consecutive_years:
            print(f"  Skipping {genre}: Insufficient consecutive data for walk-forward validation")
            continue
        
        # Perform walk-forward validation
        metrics_df, candidates_df, feature_importance_df = walk_forward_validation(
            genre_df, genre, MANUAL_PARAMS[genre], GRID
        )
        
        print(f"  Walk-forward validation completed for {genre}")
        print(f"  Metrics DataFrame shape: {metrics_df.shape}")
        print(f"  Candidates DataFrame shape: {candidates_df.shape}")
        print(f"  Feature importance DataFrame shape: {feature_importance_df.shape}")
        
        if not metrics_df.empty:
            all_metrics.append(metrics_df)
            all_candidates.append(candidates_df)
            if not feature_importance_df.empty:
                all_feature_importance.append(feature_importance_df)
            
            # Plot feature importance
            plot_feature_importance(feature_importance_df, genre, diagnostics_dir)
            
            # Calculate final parameters for this genre
            avg_rmse_per_param_set = metrics_df.groupby([
                'max_depth', 'learning_rate', 'n_estimators',
                'subsample', 'colsample_bytree', 'reg_lambda', 'reg_alpha'
            ])['rmse'].mean()
            
            if not avg_rmse_per_param_set.empty:
                best_params_idx = avg_rmse_per_param_set.idxmin()
                final_params = {
                    'max_depth': best_params_idx[0],
                    'learning_rate': best_params_idx[1],
                    'n_estimators': best_params_idx[2],
                    'subsample': best_params_idx[3],
                    'colsample_bytree': best_params_idx[4],
                    'reg_lambda': best_params_idx[5],
                    'reg_alpha': best_params_idx[6]
                }
                
                print(f"  Final parameters for {genre}: {final_params}")
                
                # Holdout evaluation
                holdout_eval_result = holdout_evaluation(genre_df, genre, final_params)
                if holdout_eval_result is not None:
                    holdout_results, holdout_metrics = holdout_eval_result
                    all_holdout_results[genre] = holdout_results
                    all_holdout_metrics[genre] = holdout_metrics
                    
                    # Save holdout results
                    holdout_results.to_csv(
                        os.path.join(output_dir, f'{genre}_holdout_2023_2024_vs_xgb.csv'), 
                        index=False
                    )
                    
                    print(f"  Holdout RMSE: {holdout_metrics['rmse']:.4f}")
                    print(f"  Holdout MAE: {holdout_metrics['mae']:.4f}")
                    print(f"  Holdout R²: {holdout_metrics['r2']:.4f}")
        
        # Save genre-specific results only if they're not empty
        if not metrics_df.empty:
            print(f"  Saving metrics for {genre} to {os.path.join(output_dir, f'{genre}_xgb_walkforward_metrics.csv')}")
            try:
                metrics_df.to_csv(os.path.join(output_dir, f'{genre}_xgb_walkforward_metrics.csv'), index=False)
                print(f"  Successfully saved metrics for {genre}")
            except Exception as e:
                print(f"  Error saving metrics for {genre}: {e}")
        else:
            print(f"  No metrics to save for {genre}")
            
        if not candidates_df.empty:
            print(f"  Saving candidates for {genre} to {os.path.join(output_dir, f'{genre}_xgb_wf_candidates.csv')}")
            try:
                candidates_df.to_csv(os.path.join(output_dir, f'{genre}_xgb_wf_candidates.csv'), index=False)
                print(f"  Successfully saved candidates for {genre}")
            except Exception as e:
                print(f"  Error saving candidates for {genre}: {e}")
        else:
            print(f"  No candidates to save for {genre}")
            
        if not feature_importance_df.empty:
            print(f"  Saving feature importance for {genre} to {os.path.join(output_dir, f'{genre}_xgb_feature_importance_by_fold.csv')}")
            try:
                feature_importance_df.to_csv(os.path.join(output_dir, f'{genre}_xgb_feature_importance_by_fold.csv'), index=False)
                print(f"  Successfully saved feature importance for {genre}")
            except Exception as e:
                print(f"  Error saving feature importance for {genre}: {e}")
        else:
            print(f"  No feature importance to save for {genre}")
    
    # Create summary report
    print("\nCreating summary report...")
    summary_stats, candidates_df_all, feature_importance_df_all = create_summary_report(
        all_metrics, all_candidates, all_feature_importance
    )
    
    if not summary_stats.empty:
        # Save summary files
        summary_stats.to_csv(os.path.join(output_dir, 'xgb_summary_by_genre.csv'), index=False)
        candidates_df_all.to_csv(os.path.join(output_dir, 'xgb_all_candidates.csv'), index=False)
        if not feature_importance_df_all.empty:
            feature_importance_df_all.to_csv(os.path.join(output_dir, 'xgb_all_feature_importance.csv'), index=False)
        
        # Save grid used
        with open(os.path.join(output_dir, 'grid_xgb_used.txt'), 'w') as f:
            f.write("Manual params per genre:\n")
            for genre, params in MANUAL_PARAMS.items():
                f.write(f"{genre}: {params}\n")
            f.write("\nGrid:\n")
            for params in ParameterGrid(GRID):
                f.write(f"{params}\n")
        
        print("\nXGBoost Training Pipeline Completed!")
        print("=" * 50)
        print(summary_stats[['genre', 'folds', 'avg_rmse_walkforward', 'avg_mae_walkforward', 'avg_r2_walkforward']])
        print("=" * 50)
        
        # Print feature importance summary
        if not feature_importance_df_all.empty:
            print("\nTop Features by Genre:")
            for genre in FOCUS_GENRES:
                genre_features = feature_importance_df_all[feature_importance_df_all['genre'] == genre]
                if not genre_features.empty:
                    mean_importance = genre_features.groupby('feature')['importance'].mean().sort_values(ascending=False)
                    top_features = mean_importance.head(3)
                    print(f"  {genre.title()}: {', '.join([f'{f} ({v:.3f})' for f, v in top_features.items()])}")
    
    else:
        print("No results to summarize. Check data or model configurations.")

if __name__ == "__main__":
    main()
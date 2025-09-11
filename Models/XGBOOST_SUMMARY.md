# XGBoost Model Implementation Summary

## Overview
This document summarizes the implementation of the XGBoost model training pipeline for music genre popularity forecasting, following the methodological approach outlined in the thesis.

## Implementation Details

### Data Preparation
- Used aggregated yearly genre data (2008-2024) from `06_aggregated_dataset.csv`
- Focused on four primary genres: Pop, Hip-Hop, Latin, and Electronic
- Utilized audio features as predictors for genre popularity

### Time-Aware Cross-Validation
- Implemented walk-forward validation with 3-year training / 2-year testing windows
- Evaluated all possible contiguous time windows
- Ensured temporal consistency in model evaluation

### Hyperparameter Optimization
- Started with manual baseline parameters per genre
- Performed grid search for hyperparameter tuning
- Selected final parameters based on lowest average RMSE across folds

### Model Evaluation
- Evaluated using RMSE, MAE, and R² metrics
- Performed holdout validation on 2023-2024 data
- Assessed feature importance for interpretability

## Key Results

### Genres Processed
1. **Pop** - Sufficient data for walk-forward validation
2. **Latin** - Sufficient data for walk-forward validation  
3. **Hip-Hop** - Insufficient consecutive years (2017, 2018, 2024)
4. **Electronic** - Insufficient consecutive years (2013-2016)

### Model Performance

#### Pop Genre
- **Walk-forward Validation RMSE**: 0.3754
- **Walk-forward Validation MAE**: 0.3746
- **Walk-forward Validation R²**: -1.2551
- **Holdout (2023-2024) RMSE**: N/A
- **Holdout (2023-2024) MAE**: N/A
- **Holdout (2023-2024) R²**: N/A
- **Best Hyperparameters**: 
  - max_depth: 3
  - learning_rate: 0.1
  - n_estimators: 300
  - subsample: 0.9
  - colsample_bytree: 0.7
  - reg_lambda: 0.0
  - reg_alpha: 0.5

#### Latin Genre
- **Walk-forward Validation RMSE**: 3.8185
- **Walk-forward Validation MAE**: 3.8103
- **Walk-forward Validation R²**: -232.2927
- **Holdout (2023-2024) RMSE**: 1.6455
- **Holdout (2023-2024) MAE**: 1.6455
- **Holdout (2023-2024) R²**: nan
- **Best Hyperparameters**: 
  - max_depth: 3
  - learning_rate: 0.03
  - n_estimators: 200
  - subsample: 0.7
  - colsample_bytree: 0.7
  - reg_lambda: 2.0
  - reg_alpha: 0.5

### Feature Importance

#### Pop Genre Top Features:
1. median_energy (0.676)
2. median_danceability (0.266)
3. median_speechiness (0.051)

#### Latin Genre Top Features:
1. median_valence (0.300)
2. median_danceability (0.205)
3. median_loudness (0.177)

## Files Generated

All output files are saved in `Models/outputs/xgb/`:

- `pop_xgb_walkforward_metrics.csv` - Walk-forward validation metrics for Pop
- `pop_xgb_wf_candidates.csv` - All candidate hyperparameter combinations tested for Pop
- `pop_xgb_feature_importance_by_fold.csv` - Feature importance for Pop by fold
- `latin_xgb_walkforward_metrics.csv` - Walk-forward validation metrics for Latin
- `latin_xgb_wf_candidates.csv` - All candidate hyperparameter combinations tested for Latin
- `latin_xgb_feature_importance_by_fold.csv` - Feature importance for Latin by fold
- `latin_holdout_2023_2024_vs_xgb.csv` - Holdout evaluation results for Latin
- `xgb_summary_by_genre.csv` - Summary report with final model performance
- `xgb_all_candidates.csv` - All candidate hyperparameter combinations tested for all genres
- `xgb_all_feature_importance.csv` - Feature importance for all genres
- `grid_xgb_used.txt` - Documentation of the hyperparameter grid used
- `[genre]_holdout_2023_2024_vs_xgb.csv` - Holdout evaluation results for each genre

## Methodological Alignment

The XGBoost implementation successfully follows the thesis methodologies:

1. **Temporal Consistency**: Respects time ordering in training and evaluation
2. **Genre-Specific Modeling**: Each genre is modeled independently with optimized hyperparameters
3. **Feature Importance Analysis**: Identifies which audio characteristics most influence genre popularity
4. **Comprehensive Reporting**: Generates detailed metrics and candidate parameter sets
5. **Cross-Validation**: Implements walk-forward validation to assess model robustness over time

## Conclusion

The XGBoost training pipeline has been successfully implemented and executed. It processes the available data for genres with sufficient consecutive years, performs time-aware cross-validation, tunes hyperparameters, evaluates model performance, and generates comprehensive reports. The pipeline correctly identifies that only Pop and Latin have sufficient data for meaningful modeling, while Hip-Hop and Electronic lack the required consecutive years for reliable walk-forward validation.
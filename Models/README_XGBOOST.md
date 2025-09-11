# XGBoost Training Pipeline

This directory contains the implementation of the XGBoost model training pipeline for music genre popularity forecasting.

## Files

- `xgboost_training_pipeline_final.ipynb` - Main Jupyter notebook implementation
- `xgboost_pipeline.py` - Python script version of the pipeline
- `README_XGBOOST.md` - Documentation for the XGBoost pipeline
- `XGBOOST_SUMMARY.md` - Summary of XGBoost implementation and results

## Methodology

The XGBoost training pipeline follows these key methodological principles aligned with the thesis:

1. **Data Preparation**
   - Uses aggregated yearly genre data (2008-2024)
   - Focuses on four primary genres: Pop, Hip-Hop, Latin, and Electronic
   - Utilizes audio features as predictors for genre popularity

2. **Time-Aware Cross-Validation**
   - Implements walk-forward validation with 3-year training / 2-year testing windows
   - Evaluates all possible contiguous time windows
   - Ensures temporal consistency in model evaluation

3. **Hyperparameter Optimization**
   - Starts with manual baseline parameters per genre
   - Performs grid search for hyperparameter tuning
   - Selects final parameters based on lowest average RMSE across folds

4. **Model Evaluation**
   - Evaluates using RMSE, MAE, and R² metrics
   - Performs holdout validation on 2023-2024 data
   - Assesses feature importance for interpretability

## Key Features

- **Genre-Specific Modeling**: Each genre is modeled independently with optimized hyperparameters
- **Feature Importance Analysis**: Identifies which audio characteristics most influence genre popularity
- **Temporal Consistency**: Respects time ordering in training and evaluation
- **Comprehensive Reporting**: Generates detailed metrics and candidate parameter sets

## Outputs

The pipeline generates the following outputs in `../Models/outputs/xgb/`:

- Walk-forward validation metrics for each genre
- All candidate hyperparameter combinations tested
- Holdout evaluation results (2023-2024)
- Feature importance analysis
- Summary reports with final model performance

## Usage

To run the XGBoost training pipeline:

1. Ensure the aggregated dataset (`06_aggregated_dataset.csv`) exists in the Datasets directory
2. Execute the Jupyter notebook or Python script:
   ```bash
   python xgboost_pipeline.py
   ```
3. View results in the output directory
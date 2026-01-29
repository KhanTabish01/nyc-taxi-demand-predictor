"""
XGBoost Model Training Pipeline

Loads engineered features, trains XGBoost model, evaluates on validation/test sets,
and saves model artifacts.
"""

import json
import pickle

from loguru import logger
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import typer
import xgboost as xgb

from urban_mobility_forecaster.config import (
    FEATURE_LIST_FILE,
    MODEL_FILE,
    MODELS_DIR,
    RESULTS_FILE,
    VERBOSE_EVAL,
    XGBOOST_PARAMS,
)
from urban_mobility_forecaster.features import load_features, prepare_features

app = typer.Typer()


def train_model(X_train, y_train, X_val, y_val, params):
    """Train XGBoost model."""
    logger.info("Initializing XGBoost model...")
    model = xgb.XGBRegressor(**params)
    
    logger.info("Training XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=VERBOSE_EVAL
    )
    
    logger.success("Training complete!")
    return model


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):
    """Evaluate model on all sets."""
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    results = {
        'train': {
            'mae': float(mean_absolute_error(y_train, y_train_pred)),
            'rmse': float(mean_squared_error(y_train, y_train_pred) ** 0.5),
            'r2': float(r2_score(y_train, y_train_pred)),
        },
        'val': {
            'mae': float(mean_absolute_error(y_val, y_val_pred)),
            'rmse': float(mean_squared_error(y_val, y_val_pred) ** 0.5),
            'r2': float(r2_score(y_val, y_val_pred)),
        },
        'test': {
            'mae': float(mean_absolute_error(y_test, y_test_pred)),
            'rmse': float(mean_squared_error(y_test, y_test_pred) ** 0.5),
            'r2': float(r2_score(y_test, y_test_pred)),
        }
    }
    
    # Log results
    logger.info("\n" + "="*70)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("="*70)
    logger.info(f"TRAIN:  MAE={results['train']['mae']:.4f}, RMSE={results['train']['rmse']:.4f}, R²={results['train']['r2']:.4f}")
    logger.info(f"VAL:    MAE={results['val']['mae']:.4f}, RMSE={results['val']['rmse']:.4f}, R²={results['val']['r2']:.4f}")
    logger.info(f"TEST:   MAE={results['test']['mae']:.4f}, RMSE={results['test']['rmse']:.4f}, R²={results['test']['r2']:.4f}")
    logger.info("="*70)
    
    # Baseline comparison
    baseline_mae = 11.21
    improvement = ((baseline_mae - results['test']['mae']) / baseline_mae) * 100
    logger.info(f"Baseline (24h-lag) Test MAE: {baseline_mae}")
    logger.info(f"XGBoost Test MAE: {results['test']['mae']:.4f}")
    logger.info(f"Improvement: {improvement:+.2f}%")
    
    # Feature importance
    logger.info("\nTop 10 Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']:20s}: {row['importance']:.6f}")
    
    return results


def save_model(model, feature_cols, results, model_path, feature_path, results_path):
    """Save model and metadata."""
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Save model
    logger.info(f"Saving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature list
    logger.info(f"Saving feature list to {feature_path}...")
    with open(feature_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    
    # Save results with metadata
    results_with_meta = {
        'model_type': 'XGBoost',
        'n_features': len(feature_cols),
        'features': feature_cols,
        'hyperparameters': XGBOOST_PARAMS,
        'metrics': results,
    }
    
    logger.info(f"Saving results to {results_path}...")
    with open(results_path, 'w') as f:
        json.dump(results_with_meta, f, indent=2)
    
    logger.success("All artifacts saved!")


@app.command()
def main():
    """Train XGBoost model on engineered features."""
    logger.info("="*70)
    logger.info("XGBoost Model Training Pipeline")
    logger.info("="*70)
    
    try:
        # Load data
        train_df, val_df, test_df = load_features()
        
        # Prepare features
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_features(train_df, val_df, test_df)
        
        # Get feature columns from train_df
        from urban_mobility_forecaster.config import FEATURE_COLUMNS
        feature_cols = FEATURE_COLUMNS
        
        # Train model
        model = train_model(X_train, y_train, X_val, y_val, XGBOOST_PARAMS)
        
        # Evaluate
        results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, feature_cols)
        
        # Save artifacts
        save_model(model, feature_cols, results, MODEL_FILE, FEATURE_LIST_FILE, RESULTS_FILE)
        
        logger.success("✓ Training pipeline complete!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    app()

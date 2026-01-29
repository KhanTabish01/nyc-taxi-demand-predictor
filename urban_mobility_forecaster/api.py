"""
FastAPI Service for NYC Taxi Demand Prediction

Serves the trained XGBoost model via REST API with /predict endpoint.
"""

import json
import pickle

from fastapi import FastAPI, HTTPException
from loguru import logger
import numpy as np
from pydantic import BaseModel, Field

from urban_mobility_forecaster.config import (
    FEATURE_COLUMNS,
    MODEL_FILE,
    RESULTS_FILE,
)

# Initialize FastAPI app
app = FastAPI(
    title="NYC Taxi Demand Predictor",
    description="Predict taxi pickups for NYC zones using XGBoost",
    version="1.0.0",
)

# Global model cache
_model = None
_results = None


def load_model():
    """Load trained XGBoost model from disk."""
    global _model, _results
    
    if _model is None:
        logger.info(f"Loading model from {MODEL_FILE}...")
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
        
        with open(MODEL_FILE, 'rb') as f:
            _model = pickle.load(f)
        logger.info("Model loaded successfully!")
    
    if _results is None:
        logger.info(f"Loading results from {RESULTS_FILE}...")
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE, 'r') as f:
                _results = json.load(f)
            logger.info("Results loaded successfully!")
    
    return _model, _results


# Request/Response schemas
class PredictionRequest(BaseModel):
    """Schema for prediction request with 17 engineered features."""
    
    # Temporal features
    hour_sin: float = Field(..., description="Hour of day (sin component)")
    hour_cos: float = Field(..., description="Hour of day (cos component)")
    dow_sin: float = Field(..., description="Day of week (sin component)")
    dow_cos: float = Field(..., description="Day of week (cos component)")
    month_sin: float = Field(..., description="Month (sin component)")
    month_cos: float = Field(..., description="Month (cos component)")
    
    # Lag features
    lag_1h: float = Field(..., description="Pickups 1 hour ago")
    lag_24h: float = Field(..., description="Pickups 24 hours ago")
    lag_168h: float = Field(..., description="Pickups 168 hours ago (1 week)")
    diff_24h: float = Field(..., description="Change from 24h ago")
    
    # Rolling statistics
    rolling_7d_mean: float = Field(..., description="7-day rolling mean")
    rolling_7d_std: float = Field(..., description="7-day rolling std")
    rolling_14d_mean: float = Field(..., description="14-day rolling mean")
    rolling_7d_cv: float = Field(..., description="7-day coefficient of variation")
    
    # Zone features
    zone_mean_demand: float = Field(..., description="Historical zone mean demand")
    zone_rank: float = Field(..., description="Zone popularity rank")
    zone_is_top50: int = Field(..., description="1 if zone in top 50, else 0")


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    
    predicted_pickups: float = Field(..., description="Predicted number of pickups")
    confidence: str = Field(..., description="Prediction confidence level")
    model_version: str = Field(..., description="Model version/timestamp")


class HealthResponse(BaseModel):
    """Schema for health check response."""
    
    status: str
    model_loaded: bool
    n_features: int
    test_mae: float = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
        logger.info("✓ API ready for predictions!")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        model, results = load_model()
        test_mae = results.get('metrics', {}).get('test', {}).get('mae') if results else None
        
        return HealthResponse(
            status="healthy",
            model_loaded=model is not None,
            n_features=len(FEATURE_COLUMNS),
            test_mae=test_mae,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Model not available")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict taxi pickups for a given zone-hour with engineered features.
    
    Returns predicted number of pickups and confidence level.
    """
    try:
        model, results = load_model()
        
        # Extract features in correct order
        features = np.array([[
            request.hour_sin,
            request.hour_cos,
            request.dow_sin,
            request.dow_cos,
            request.month_sin,
            request.month_cos,
            request.lag_1h,
            request.lag_24h,
            request.lag_168h,
            request.diff_24h,
            request.rolling_7d_mean,
            request.rolling_7d_std,
            request.rolling_14d_mean,
            request.rolling_7d_cv,
            request.zone_mean_demand,
            request.zone_rank,
            request.zone_is_top50,
        ]])
        
        # Make prediction
        prediction = float(model.predict(features)[0])
        
        # Determine confidence based on prediction magnitude
        # Higher predictions (busier zones/times) are more reliable
        if prediction < 5:
            confidence = "low"
        elif prediction < 20:
            confidence = "medium"
        else:
            confidence = "high"
        
        logger.info(f"Prediction made: {prediction:.2f} pickups (confidence: {confidence})")
        
        return PredictionResponse(
            predicted_pickups=max(0, prediction),  # No negative predictions
            confidence=confidence,
            model_version="xgboost-v1.0-test-mae-0.84",
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/info")
async def model_info():
    """Get model information and performance metrics."""
    try:
        model, results = load_model()
        
        if results is None:
            raise HTTPException(status_code=500, detail="Results metadata not available")
        
        return {
            "model_type": results.get("model_type"),
            "n_features": results.get("n_features"),
            "features": results.get("features"),
            "hyperparameters": results.get("hyperparameters"),
            "metrics": results.get("metrics"),
        }
    
    except Exception as e:
        logger.error(f"Info request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting API server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )

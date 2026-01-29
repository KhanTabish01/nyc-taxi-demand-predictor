from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# ============================================================================
# XGBoost Model Training Configuration
# ============================================================================

# Feature columns (17 features from feature engineering)
FEATURE_COLUMNS = [
    # Temporal features (cyclical encoding)
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
    'month_sin', 'month_cos',
    # Lag features (temporal dependencies)
    'lag_1h', 'lag_24h', 'lag_168h',
    'diff_24h',
    # Rolling statistics (trend & volatility)
    'rolling_7d_mean', 'rolling_7d_std',
    'rolling_14d_mean', 'rolling_7d_cv',
    # Zone features (spatial context)
    'zone_mean_demand', 'zone_rank', 'zone_is_top50'
]

TARGET_COLUMN = 'pickups'
METADATA_COLUMNS = ['pickup_hour', 'zone_id']

# Data file paths
TRAIN_FEATURES_FILE = PROCESSED_DATA_DIR / 'train_features.csv'
VAL_FEATURES_FILE = PROCESSED_DATA_DIR / 'val_features.csv'
TEST_FEATURES_FILE = PROCESSED_DATA_DIR / 'test_features.csv'

# Model file paths
MODEL_FILE = MODELS_DIR / 'xgboost_model.pkl'
FEATURE_LIST_FILE = MODELS_DIR / 'feature_list.txt'
RESULTS_FILE = MODELS_DIR / 'results.json'

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
}

# Training configuration
VERBOSE_EVAL = 50

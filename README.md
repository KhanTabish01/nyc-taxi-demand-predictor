# NYC Taxi Demand Forecaster

> Predicts hourly taxi demand by NYC zone to optimize fleet allocation and reduce driver idle time.

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Problem Statement

**Business Context:**  
Taxi fleet operators lose revenue when drivers idle in low-demand areas while high-demand zones go underserved. Accurate hourly demand forecasts enable efficient fleet positioning, reducing wait times for passengers and maximizing driver earnings.

**Technical Goal:**  
Predict the number of taxi pickups per NYC zone for the next hour using historical trip data, weather patterns, and temporal features.

---

## 📊 Results

- **Dataset:** 41M NYC Yellow Taxi trips (2024)
- **Coverage:** 12 months, 265 zones, ~3.4M trips/month average
- **Pipeline Status:** ✅ Data ingestion complete
- **Model:** In development (XGBoost regression)
- **Target Metric:** MAE (Mean Absolute Error) on hourly zone-level pickups

---

## 🏗️ Architecture

**Data Pipeline:**
```
NYC TLC (Parquet) → download.py → data/raw/
                  → ingest.py → SQLite → data/processed/taxi.db
                  → train.py → model.pkl
                  → api.py → FastAPI Inference Service
```

**Tech Stack:**
- **Data:** Pandas, Parquet, SQLite
- **ML:** Scikit-Learn / XGBoost
- **API:** FastAPI (Day 3)
- **Deployment:** Docker (Day 3)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- 10GB free disk space (for data)

### Setup
```bash
# Clone repo
git clone https://github.com/KhanTabish01/mobility-forecaster.git
cd mobility-forecaster

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Pipeline
```bash
# Download 2024 data (12 months, ~5GB)
python urban_mobility_forecaster/download.py

# Ingest into SQLite (~41M rows)
python urban_mobility_forecaster/ingest.py

# Train model (coming soon)
# python urban_mobility_forecaster/modeling/train.py

# Start API (coming soon)
# uvicorn urban_mobility_forecaster.api:app --reload
```

---

## 📂 Project Structure

```
├── urban_mobility_forecaster/   # Source code
│   ├── download.py              # Data acquisition from NYC TLC
│   ├── ingest.py                # ETL pipeline to SQLite
│   ├── modeling/
│   │   ├── train.py             # Model training pipeline
│   │   └── predict.py           # Inference logic
│   └── api.py                   # FastAPI service
├── data/                        # Data storage (gitignored)
│   ├── raw/                     # Original Parquet files (~5GB)
│   └── processed/               # SQLite database (~2GB)
├── notebooks/                   # EDA and experiments
├── docs/                        # Design decisions and documentation
└── requirements.txt
```

---

## 🧪 Design Decisions

### Why SQLite?
SQLite offers zero-config portability and sufficient performance for <50M rows with proper indexing. Trade-off: Limited write concurrency makes it unsuitable for high-throughput production, but perfect for MVP development and local experimentation. Migration path to PostgreSQL is straightforward when scale demands it.

### Why Pre-aggregate to Hourly?
The prediction target is aggregate demand (pickups per zone per hour), not individual trip attributes. Pre-aggregating reduces data volume by 1000x while preserving temporal patterns. Raw trip-level features (distance, fare) are averaged during aggregation to retain signal.

### Time-Series Splitting
Standard random train/test splits cause data leakage in time-series forecasting. We use temporal split: train on Jan-Oct 2024, validate on Nov 2024, test on Dec 2024. Features must only use historical data (lag features, rolling windows) to prevent look-ahead bias.

See [DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md) for full rationale.

---

## 🔮 Future Improvements

- [ ] Add real-time streaming ingestion (Kafka)
- [ ] Switch to PostgreSQL for production scale
- [ ] Implement A/B testing framework for model updates
- [ ] Add monitoring/alerting for model drift

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙋 Contact

Built as a production-style ML system demonstrating end-to-end pipeline development.  
[LinkedIn](https://www.linkedin.com/in/tabish-khan-864457219/) | [Email](mailto:tkhan1051@gmail.com)


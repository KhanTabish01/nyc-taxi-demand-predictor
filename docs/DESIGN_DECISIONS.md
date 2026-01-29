## Purpose
Document the why behind key choices so future you/interviewers can see the trade-offs and constraints. Keep entries short and dated.

## How to add an entry
1) State the decision and date.
2) List 2–3 options considered.
3) Note constraints (data size, time, infra limits).
4) Record the rationale and what would change the decision later.

## Decision 1: Feature Encoding for Hour-of-Day (2026-01-26)
**Decision:** Encode hour as cyclical sin/cos rather than one-hot 24 categories.

**Options Considered:**
1. One-hot encoding (24 binary columns): Simple, fast, no relationships
2. Sine/Cosine (2 columns): Captures circularity (hour 23 → 0 is continuous)
3. Ordinal (0-23): Wrong—suggests order but hour 0 ≠ hour 23

**Constraints:** Time-series requires circularity; hour-of-day explains 10.3x variance.

**Rationale:** EDA showed hour is strongest feature (peak 18:00, trough 5:00). Cyclical encoding respects that 11pm ≈ 1am (both night). Formula: `sin(2π × hour/24)`, `cos(2π × hour/24)`.

**Change triggers:** If one-hot beats sin/cos in validation (unlikely), switch it.

---

## Decision 3: Baseline Model Cascade Before XGBoost (2026-01-29)
**Decision:** Build simple rule-based baselines (B1–B4) before training XGBoost to establish performance floor and prevent over-engineering.

**Options Considered:**
1. Skip baselines, go straight to XGBoost: Faster but no interpretability, risk over-fit
2. Build baseline cascade: Understand signal hierarchy, set realistic benchmarks
3. Use ensemble of baselines + XGBoost: More complex but potentially robust

**Constraints:** Time pressure (3-day MVP sprint); need to validate features matter before ML.

**Rationale:** Baseline cascade revealed:
- **B1 (Global Mean) → B2 (Hour-of-Day Mean):** 76.8% error reduction → hour-of-day is dominant signal
- **B2 → B3 (24h-Lag):** Marginal 0.6% improvement on val, but 18.6% improvement on test → day-to-day persistence adds value in unseen data
- **B3 → B4 (Rolling Avg):** Degradation (over-smoothing) → rolling windows hurt prediction

**Benchmark Set:** XGBoost must beat test MAE **11.21** (B3 best test performance) to justify added complexity. This prevents gold-plating.

**Change triggers:** If XGBoost fails to beat 11.21 MAE, return to hybrid model (B2/B3 as base features for meta-learner).

---

## Current assumptions (fill these in)
- Data volume assumption: ~41M raw trips/year → hourly-zone aggregates (~300K records/month)
- Target latency for inference: <100ms per prediction (single zone-hour)
- Update cadence for new data: Daily (ingest previous day's trips, retrain weekly)
- Baseline cascade results: B3 (24h-lag) achieves 11.21 MAE on test → XGBoost must beat this


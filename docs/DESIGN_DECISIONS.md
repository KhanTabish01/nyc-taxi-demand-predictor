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

## Decision 2: Temporal Train/Val/Test Split (2026-01-26)
**Decision:** Split by date (Jan-Oct train, Nov val, Dec test) NOT random shuffle.

**Options Considered:**
1. Random 80/10/10 split: Standard ML, but violates time-series assumption
2. Temporal split: Respects causality, tests true forecasting
3. K-fold time-series CV: More complex, but could be added later

**Constraints:** Must prevent leakage (future data cannot touch training).

**Rationale:** Time-series forecasting requires temporal order. Random split leaks future information into training set → overly optimistic metrics. Temporal split tests real scenario: "predict next month given history".

**Change triggers:** If we have more data (>2 years), implement expanding window CV.

---

## Decision 3: Sparse Zone Handling (2026-01-26, PENDING)
**Decision:** TBD—drop zones with <50 trips/month OR train separate models?

**Options Considered:**
1. Drop sparse zones: Simple, focuses on high-demand areas
2. Separate low-demand model: More complex, but improves minority coverage
3. Impute zeros with group means: Middle ground, some bias

**Constraints:** Sparsity is 52% (many zone-hours inactive). Need to handle during feature engineering.

**Rationale:** Pareto principle—top 50 zones = 95% demand. Low-demand zones are noisy. Decision pending: Is business requirement to serve all zones or optimize for high-demand?

**Change triggers:** If product requires 100% zone coverage, build option 2 or 3.

## Current assumptions (fill these in)
- Data volume assumption: ~41M raw trips/year → hourly-zone aggregates (~300K records/month)
- Target latency for inference: <100ms per prediction (single zone-hour)
- Update cadence for new data: Daily (ingest previous day's trips, retrain weekly)


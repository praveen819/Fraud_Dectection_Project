# Real-Time Transaction Anomaly & Fraud Detection Service

A production-shaped anomaly/fraud detection system: trains an Isolation Forest
on transaction features, serves real-time scoring through a FastAPI endpoint,
logs every prediction, and includes a drift-monitoring script that flags when
the model should be retrained.

Built to demonstrate the full lifecycle — not just a notebook model, but a
deployable, monitorable service.

## Architecture

```
scripts/generate_data.py  -> data/transactions.csv        (synthetic, realistic transaction data)
scripts/train.py          -> models/isolation_forest.joblib, scaler.joblib, baseline_stats.json
app/main.py                (FastAPI)  -> /predict, /health, /monitor/summary
scripts/monitor.py         -> polls /monitor/summary, flags drift, recommends retraining
Dockerfile                 -> containerizes the whole service
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate data (swap for a real dataset, e.g. Kaggle Credit Card Fraud, when ready)
python scripts/generate_data.py

# 2. Train the model
python scripts/train.py

# 3. Serve it
uvicorn app.main:app --reload --port 8000

# 4. Score a transaction
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d \
  '{"txn_id":"TXN1","amount":980,"hour_of_day":3,"txn_count_last_hour":8,"distance_from_home_km":450,"is_new_merchant":1}'

#if above command doesn't work use below command:
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body '{"txn_id":"TXN1","amount":980,"hour_of_day":3,"txn_count_last_hour":8,"distance_from_home_km":450,"is_new_merchant":1}'

# 5. Check for drift (run periodically in production, e.g. via cron)
python scripts/monitor.py --url http://localhost:8000
```

## Docker

```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

## Design notes

- **Isolation Forest** over a supervised classifier because in real fraud
  systems, labeled fraud is rare/delayed — anomaly detection doesn't need
  labels to flag unusual behavior, only to validate afterward.
- **`contamination`** is set from the expected fraud rate; in production
  you'd tune this against a validation set and cost-of-false-positive vs
  cost-of-missed-fraud tradeoffs.
- **Baseline stats saved at train time** (`baseline_stats.json`) are what
  `monitor.py` compares live traffic against — this is the minimal version
  of what tools like Evidently or WhyLabs do at scale.
- **Prediction log (`data/prediction_log.csv`)** doubles as an audit trail
  and the raw material for retraining on real outcomes later.

## Known limitation

The synthetic data in `generate_data.py` is intentionally well-separated
(the model hits ~100% on it), which makes it a clean demo but not evidence
of real-world performance. Swap in a real dataset (Kaggle's Credit Card
Fraud Detection set is a good drop-in — same schema, just remap columns)
before citing an accuracy number anywhere.

## Resume bullet (once you've run it against real or Kaggle data)

> Built and deployed a real-time fraud/anomaly detection service (Isolation
> Forest, FastAPI, Docker) with automated drift monitoring that flags when
> retraining is needed; processes transaction-level features and returns
> scored predictions in <10ms.

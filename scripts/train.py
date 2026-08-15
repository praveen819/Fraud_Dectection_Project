"""
Trains an Isolation Forest anomaly/fraud detector and saves:
  - the fitted model
  - the fitted feature scaler
  - a baseline score distribution (used later by monitor.py to detect drift)

Run: python train.py
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = "C:\\Users\\Admin\\Downloads\\fraud-detection-project\\data\\transactions.csv"
MODEL_DIR = "C:\\Users\\Admin\\Downloads\\fraud-detection-project\\models"
FEATURES = ["amount", "hour_of_day", "txn_count_last_hour",
            "distance_from_home_km", "is_new_merchant"]


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # contamination = expected fraud rate; IsolationForest uses this to set its
    # internal decision threshold for "how much of the data is anomalous"
    contamination = y.mean()
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # IsolationForest: -1 = anomaly, 1 = normal. Flip to match label convention (1 = fraud).
    raw_pred = model.predict(X_scaled)
    y_pred = np.where(raw_pred == -1, 1, 0)
    scores = -model.score_samples(X_scaled)  # higher score = more anomalous

    print(classification_report(y, y_pred, target_names=["normal", "fraud"]))
    print(f"ROC-AUC (score vs true label): {roc_auc_score(y, scores):.3f}")

    joblib.dump(model, f"{MODEL_DIR}/isolation_forest.joblib")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")

    baseline = {
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "p95_score": float(np.percentile(scores, 95)),
        "features": FEATURES,
        "contamination": float(contamination),
    }
    with open(f"{MODEL_DIR}/baseline_stats.json", "w") as f:
        json.dump(baseline, f, indent=2)

    print("\nSaved model, scaler, and baseline stats to models/")


if __name__ == "__main__":
    main()

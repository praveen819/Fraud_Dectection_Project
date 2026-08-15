"""
Polls the running API's /monitor/summary and flags when the live anomaly
score distribution has drifted meaningfully from the training-time baseline.
In a real deployment this would run on a schedule (cron / Airflow) and page
someone or trigger scripts/train.py automatically when drift is flagged.

Run: python monitor.py --url http://localhost:8000
"""
import argparse
import sys

import requests

DRIFT_THRESHOLD_STD = 1.5  # flag if recent mean score is >1.5 baseline-std away


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    summary = requests.get(f"{args.url}/monitor/summary", timeout=5).json()
    if summary.get("count", 0) < 20:
        print(f"Only {summary.get('count', 0)} predictions logged so far — not enough to judge drift yet.")
        return

    baseline_mean = summary["baseline_mean_score"]
    recent_mean = summary["recent_mean_score"]
    recent_rate = summary["recent_anomaly_rate"]

    drift = abs(recent_mean - baseline_mean)
    print(f"Baseline mean score: {baseline_mean:.4f}")
    print(f"Recent mean score:   {recent_mean:.4f}")
    print(f"Recent anomaly rate: {recent_rate:.2%}")

    if drift > DRIFT_THRESHOLD_STD:
        print("\n⚠️  DRIFT DETECTED — recent scores diverge from training baseline.")
        print("    Recommended action: retrain (python scripts/train.py) on recent data.")
        sys.exit(1)
    else:
        print("\n✅ No significant drift detected.")


if __name__ == "__main__":
    main()

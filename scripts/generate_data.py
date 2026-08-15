"""
Generates a synthetic but realistic transaction dataset for fraud/anomaly detection.

Why synthetic instead of downloading a Kaggle set: this environment can't reach
Kaggle, and shipping a generator means anyone (recruiter, interviewer) can
`python generate_data.py` and reproduce your results with zero setup friction —
that's a good thing to point out in an interview.

Swap this out for the real Kaggle "Credit Card Fraud Detection" dataset later
if you want real-world numbers; the rest of the pipeline doesn't change.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_NORMAL = 9800
N_FRAUD = 200  # ~2% fraud rate, realistic class imbalance


def make_normal_transactions(n):
    return pd.DataFrame({
        "amount": RNG.gamma(shape=2.0, scale=40, size=n),           # small, typical purchases
        "hour_of_day": RNG.normal(14, 4, n).clip(0, 23),             # mostly daytime
        "txn_count_last_hour": RNG.poisson(1.2, n),                  # normal usage pace
        "distance_from_home_km": RNG.gamma(shape=1.5, scale=5, size=n),
        "is_new_merchant": RNG.binomial(1, 0.05, n),                 # rarely a brand-new merchant
        "label": 0,
    })


def make_fraud_transactions(n):
    return pd.DataFrame({
        "amount": RNG.gamma(shape=3.0, scale=250, size=n),           # unusually large amounts
        "hour_of_day": RNG.normal(3, 3, n).clip(0, 23),               # odd hours
        "txn_count_last_hour": RNG.poisson(6, n),                    # rapid-fire transactions
        "distance_from_home_km": RNG.gamma(shape=2.0, scale=200, size=n),  # far from home
        "is_new_merchant": RNG.binomial(1, 0.7, n),                  # usually a new/unknown merchant
        "label": 1,
    })


def main():
    normal = make_normal_transactions(N_NORMAL)
    fraud = make_fraud_transactions(N_FRAUD)
    df = pd.concat([normal, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df["txn_id"] = [f"TXN{100000+i}" for i in range(len(df))]
    df = df[["txn_id", "amount", "hour_of_day", "txn_count_last_hour",
             "distance_from_home_km", "is_new_merchant", "label"]]
    df.to_csv("C:\\Users\\Admin\\Downloads\\fraud-detection-project\\data\\transactions.csv", index=False)
    print(f"Wrote {len(df)} rows ({df['label'].sum()} fraud / {len(df) - df['label'].sum()} normal)")


if __name__ == "__main__":
    main()

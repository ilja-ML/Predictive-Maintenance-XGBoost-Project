"""
Time-series feature engineering for predictive maintenance.

Creates degradation signals from raw sensor data:
- Pre-failure windows (target labeling)
- Rolling statistics per device
"""

import pandas as pd

METRIC_COLS: list[str] = [
    "metric1", "metric2", "metric3", "metric4", "metric5",
    "metric6", "metric7", "metric8", "metric9",
]
ROLLING_WINDOWS: list[int] = [3, 7]
PRE_FAILURE_DAYS: int = 7

# Columns excluded from model features to prevent data leakage
DROP_COLS: list[str] = ["date", "device", "failure", "target"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all feature engineering steps. Returns enriched DataFrame."""
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["device", "date"]).reset_index(drop=True)

    # Pre-failure window: label the 7 days leading up to each failure as target=1,
    # allowing the model to learn degradation patterns instead of point-in-time failures.
    df["target"] = _create_pre_failure_target(df)

    df = _add_rolling_features(df)

    # Rolling windows produce NaN at the start of each device's time series
    rolling_cols = [c for c in df.columns if "_roll" in c]
    df[rolling_cols] = (
        df.groupby("device")[rolling_cols]
        .apply(lambda g: g.bfill().ffill())
        .reset_index(level=0, drop=True)
    )

    n_before = len(df)
    df = df.dropna(subset=rolling_cols)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"   ⚠ {n_dropped} Zeilen mit verbleibenden NaN entfernt.")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Returns model-safe feature columns, excluding leakage-prone fields."""
    return [c for c in df.columns if c not in DROP_COLS]


def _create_pre_failure_target(df: pd.DataFrame) -> pd.Series:
    """
    Reverse rolling max over 'failure' column per device.
    Propagates failure=1 backward by PRE_FAILURE_DAYS to create
    a degradation window the model can learn from.
    """
    target = (
        df.groupby("device")["failure"]
        .transform(
            lambda s: s[::-1]
            .rolling(window=PRE_FAILURE_DAYS + 1, min_periods=1)
            .max()[::-1]
        )
    )
    return target.astype(int)


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes rolling means per device for each metric and window size."""
    grouped = df.groupby("device")

    for window in ROLLING_WINDOWS:
        rolled = grouped[METRIC_COLS].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        rolled.columns = [f"{col}_roll{window}_mean" for col in METRIC_COLS]
        df = pd.concat([df, rolled], axis=1)

    return df


if __name__ == "__main__":
    test_df = pd.read_csv("data.csv")
    result = engineer_features(test_df)
    print(f"Shape: {test_df.shape} → {result.shape}")
    print(f"Target-Rate: {result['target'].mean():.2%} (Failure-Rate: {result['failure'].mean():.2%})")
    print(f"Features: {get_feature_columns(result)}")

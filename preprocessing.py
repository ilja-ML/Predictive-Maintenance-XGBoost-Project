"""
Data preprocessing with leakage-safe splitting and feature selection.

Key design decisions:
- GroupShuffleSplit isolates complete devices into train/test to prevent
  temporal leakage across device boundaries.
- Feature selection via SelectFromModel reduces noise from low-importance metrics.
- SMOTE available but disabled by default when using XGBoost's scale_pos_weight.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE

from feature_engineering import get_feature_columns

TARGET_COL: str = "target"
MAX_FEATURES: int = 15


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = True,
    use_feature_selection: bool = True,
    max_features: int = MAX_FEATURES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """
    Splits by device group, scales, optionally resamples, and selects features.

    Returns: X_train, X_test, y_train, y_test, scaler, feature_cols
    """
    feature_cols: list[str] = get_feature_columns(df)
    y: pd.Series = df[TARGET_COL]
    groups: pd.Series = df["device"]

    # GroupShuffleSplit prevents data leakage by ensuring no device
    # appears in both train and test sets.
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, y=y, groups=groups))

    X_train: pd.DataFrame = df.iloc[train_idx][feature_cols].copy()
    X_test: pd.DataFrame = df.iloc[test_idx][feature_cols].copy()
    y_train: np.ndarray = y.iloc[train_idx].to_numpy()
    y_test: np.ndarray = y.iloc[test_idx].to_numpy()

    train_devices = groups.iloc[train_idx].nunique()
    test_devices = groups.iloc[test_idx].nunique()
    print(f"   ✔ GroupShuffleSplit: {train_devices} Geräte Train, "
          f"{test_devices} Geräte Test (keine Überlappung)")

    # Scaler fitted exclusively on training data
    scaler = StandardScaler()
    X_train_scaled: np.ndarray = scaler.fit_transform(X_train)
    X_test_scaled: np.ndarray = scaler.transform(X_test)

    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

    # Feature selection reduces dimensionality and removes noisy metrics
    # that would increase false positives without improving recall.
    if use_feature_selection:
        selector_rf = RandomForestClassifier(
            n_estimators=50, max_depth=10,
            class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1,
        )
        selector = SelectFromModel(
            estimator=selector_rf,
            max_features=max_features,
            threshold=-np.inf,
        )
        selector.fit(X_train_scaled, y_train)
        X_train_scaled = selector.transform(X_train_scaled)
        X_test_scaled = selector.transform(X_test_scaled)

        selected_mask = selector.get_support()
        feature_cols = [f for f, keep in zip(feature_cols, selected_mask) if keep]

        print(f"   ✔ Feature Selection: {len(feature_cols)} von "
              f"{sum(1 for _ in selected_mask)} Features behalten")
        print(f"     → {feature_cols}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols

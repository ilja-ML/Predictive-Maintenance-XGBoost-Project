"""
XGBoost model training with native class imbalance handling via scale_pos_weight.
"""

import numpy as np
from xgboost import XGBClassifier


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float = 1.0,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    random_state: int = 42,
) -> XGBClassifier:
    """
    Trains an XGBClassifier.

    scale_pos_weight adjusts the loss function to penalize misclassification
    of the minority class (failures) proportionally to the class ratio,
    eliminating the need for synthetic oversampling (SMOTE).
    """
    clf = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf

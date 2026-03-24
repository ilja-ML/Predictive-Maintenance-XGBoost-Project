"""
Model evaluation with cost-sensitive threshold optimization.

Metrics chosen for highly imbalanced data: PR-AUC, F2-Score, explicit FN/FP counts.
Standard accuracy is misleading at 0.09% failure rate and deliberately omitted.
"""

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    fbeta_score,
)
from economic_analysis import calculate_financial_impact

sns.set_theme(style="whitegrid", font_scale=1.15)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    report: str = classification_report(
        y_true, y_pred,
        target_names=["Kein Ausfall (0)", "Ausfall (1)"],
    )
    print("\n" + "=" * 60)
    print("  KLASSIFIKATIONSBERICHT")
    print("=" * 60)
    print(report)
    return report


def print_advanced_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
    """PR-AUC and F2-Score are better suited than accuracy for extreme class imbalance."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    # F2 weights recall 4x higher than precision (beta²=4)
    f2 = fbeta_score(y_true, y_pred, beta=2)

    print("\n" + "=" * 60)
    print("  ERWEITERTE METRIKEN")
    print("=" * 60)
    print(f"  PR-AUC:  {pr_auc:.4f}")
    print(f"  F2-Score: {f2:.4f}")
    print("-" * 60)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"  False Negatives: {fn}")
    print(f"  False Positives: {fp}")
    print("=" * 60)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Grid search over classification thresholds (0.01–0.99).
    Optimizes for minimum total cost using the asymmetric cost matrix
    (FN=5000€, FP=200€) instead of maximizing accuracy.
    """
    best_threshold = 0.5
    lowest_cost = float('inf')
    best_cm = None
    best_y_pred = None

    for t in np.linspace(0.01, 0.99, 99):
        y_pred_t = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred_t, labels=[0, 1])
        cost = calculate_financial_impact(cm)["cost_ml"]

        if cost < lowest_cost:
            lowest_cost = cost
            best_threshold = t
            best_cm = cm
            best_y_pred = y_pred_t

    return best_threshold, best_y_pred, best_cm


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save: bool = True) -> np.ndarray:
    cm: np.ndarray = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Kein Ausfall", "Ausfall"],
        yticklabels=["Kein Ausfall", "Ausfall"],
        linewidths=0.8, ax=ax,
    )
    ax.set_xlabel("Vorhersage", fontsize=13)
    ax.set_ylabel("Tatsächlich", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save:
        path = OUTPUT_DIR / "confusion_matrix.png"
        fig.savefig(path, dpi=150)
        print(f"✅  Confusion Matrix gespeichert: {path}")

    plt.close(fig)
    return cm


def plot_feature_importance(clf: Any, feature_names: Sequence[str], save: bool = True) -> None:
    importances: np.ndarray = clf.feature_importances_
    sorted_idx: np.ndarray = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color=sns.color_palette("viridis", n_colors=len(feature_names)),
    )
    ax.set_xlabel("Feature Importance", fontsize=13)
    ax.set_title("Feature Importance – XGBoost", fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save:
        path = OUTPUT_DIR / "feature_importance.png"
        fig.savefig(path, dpi=150)
        print(f"✅  Feature Importance gespeichert: {path}")

    plt.close(fig)

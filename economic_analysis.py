"""
Economic impact analysis comparing ML-based maintenance vs. run-to-failure strategy.
"""

from typing import Any

import numpy as np

# Asymmetric cost matrix (€): undetected failures (FN) cause 25x more
# damage than unnecessary inspections (FP).
COST_TRUE_POSITIVE: float = 500.0
COST_FALSE_NEGATIVE: float = 5_000.0
COST_FALSE_POSITIVE: float = 200.0
COST_TRUE_NEGATIVE: float = 0.0


def calculate_financial_impact(cm: np.ndarray) -> dict[str, Any]:
    """
    Computes ML model cost vs. run-to-failure baseline from a 2x2 confusion matrix.
    Run-to-failure assumes all actual failures (TP + FN) result in unplanned downtime.
    """
    tn, fp, fn, tp = cm.ravel()

    cost_tp: float = tp * COST_TRUE_POSITIVE
    cost_fn: float = fn * COST_FALSE_NEGATIVE
    cost_fp: float = fp * COST_FALSE_POSITIVE
    cost_tn: float = tn * COST_TRUE_NEGATIVE

    cost_ml: float = cost_tp + cost_fn + cost_fp + cost_tn

    total_failures: int = int(tp + fn)
    cost_run_to_fail: float = total_failures * COST_FALSE_NEGATIVE

    savings: float = cost_run_to_fail - cost_ml
    savings_percent: float = (
        (savings / cost_run_to_fail * 100) if cost_run_to_fail > 0 else 0.0
    )

    return {
        "cost_ml": cost_ml,
        "cost_run_to_fail": cost_run_to_fail,
        "savings": savings,
        "savings_percent": savings_percent,
        "detail": {
            "True Positives (geplante Wartung)": {"count": int(tp), "cost": cost_tp},
            "False Negatives (Stillstand)":      {"count": int(fn), "cost": cost_fn},
            "False Positives (unnötige Wartung)": {"count": int(fp), "cost": cost_fp},
            "True Negatives (kein Ausfall)":      {"count": int(tn), "cost": cost_tn},
        },
    }


def print_financial_report(result: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("  WIRTSCHAFTLICHE BEWERTUNG – PREDICTIVE MAINTENANCE")
    print("=" * 60)

    for category, values in result["detail"].items():
        print(f"  {category}:")
        print(f"    Anzahl: {values['count']:>6}")
        print(f"    Kosten: {values['cost']:>10,.2f} €")

    print("-" * 60)
    print(f"  💰  Gesamtkosten MIT ML-Modell:      {result['cost_ml']:>12,.2f} €")
    print(f"  🔧  Gesamtkosten OHNE Modell (R2F):  {result['cost_run_to_fail']:>12,.2f} €")
    print("-" * 60)
    print(f"  ✅  Ersparnis durch Predictive Maint.: {result['savings']:>10,.2f} €")
    print(f"      ({result['savings_percent']:.1f} % der Run-to-Failure-Kosten)")
    print("=" * 60 + "\n")

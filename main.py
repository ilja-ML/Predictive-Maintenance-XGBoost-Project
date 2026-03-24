"""
Predictive Maintenance – Pipeline Orchestration

Workflow: CSV → Feature Engineering → Preprocessing → XGBoost → Evaluation → Cost Analysis
"""

import sys
import joblib
import numpy as np
import pandas as pd
from feature_engineering import engineer_features
from preprocessing import prepare_data
from model import train_xgboost
from evaluation import (
    print_classification_report,
    print_advanced_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    find_optimal_threshold,
)
from economic_analysis import calculate_financial_impact, print_financial_report


def main() -> None:

    # 1. Data Ingestion
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    print(f"🔄  Schritt 1/6 – Daten werden aus '{csv_path}' eingelesen …")
    df = pd.read_csv(csv_path)
    print(f"   ✔ {len(df)} Datensätze eingelesen  "
          f"(Ausfallrate: {df['failure'].mean():.2%})")

    # 2. Feature Engineering
    print("\n🔄  Schritt 2/6 – Feature Engineering …")
    df = engineer_features(df)
    print(f"   ✔ {len(df)} Datensätze nach Feature Engineering")
    print(f"   ✔ Target-Rate (Pre-Failure Window): {df['target'].mean():.2%}  "
          f"(vs. originale Failure-Rate: {df['failure'].mean():.2%})")
    n_features = len([c for c in df.columns if c not in ['date', 'device', 'failure', 'target']])
    print(f"   ✔ {n_features} Features generiert")

    # 3. Preprocessing (GroupShuffleSplit, Scaling, Feature Selection)
    print("\n🔄  Schritt 3/6 – Datenvorbereitung …")
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data(df, use_smote=False)
    print(f"   ✔ Trainingsdaten: {X_train.shape[0]}  |  Testdaten: {X_test.shape[0]}")
    if len(feature_cols) > 5:
        print(f"   ✔ Feature-Spalten ({len(feature_cols)}): {feature_cols[:5]} …")
    else:
        print(f"   ✔ Feature-Spalten ({len(feature_cols)}): {feature_cols}")

    # 4. Model Training
    # scale_pos_weight handles the extreme class imbalance natively in XGBoost's loss function,
    # making SMOTE unnecessary and avoiding synthetic sample artifacts.
    print("\n🔄  Schritt 4/6 – XGBoost wird trainiert …")
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    spw = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"   ✔ scale_pos_weight = {n_neg} / {n_pos} = {spw:.1f}")

    clf = train_xgboost(X_train, y_train, scale_pos_weight=spw)
    print(f"   ✔ Modell trainiert  ({clf.n_estimators} Boosting-Runden, "
          f"max_depth={clf.max_depth}, lr={clf.learning_rate})")

    # 5. Evaluation & Cost-Based Threshold Optimization
    # Standard threshold of 0.5 is suboptimal for asymmetric cost structures.
    # Grid search over thresholds to minimize total cost (FN: 5000€ vs FP: 200€).
    print("\n🔄  Schritt 5/6 – Modell-Evaluation & Threshold Tuning …")
    y_prob = clf.predict_proba(X_test)[:, 1]

    best_t, y_pred_opt, best_cm = find_optimal_threshold(y_test, y_prob)
    print(f"   ✔ Optimaler Schwellenwert: {best_t:.2f}")

    print_classification_report(y_test, y_pred_opt)
    print_advanced_metrics(y_test, y_pred_opt, y_prob)
    plot_confusion_matrix(y_test, y_pred_opt)
    plot_feature_importance(clf, feature_names=feature_cols)

    # 6. Economic Evaluation
    print("\n🔄  Schritt 6/6 – Wirtschaftliche Bewertung …")
    result = calculate_financial_impact(best_cm)
    print_financial_report(result)

    # Persist model artifacts for the FastAPI inference endpoint (app.py)
    joblib.dump(clf, "output/model.joblib")
    joblib.dump(scaler, "output/scaler.joblib")
    print("   ✔ Modell und Scaler exportiert → output/*.joblib")

    print("🏁  Analyse abgeschlossen. Plots: output/\n")


if __name__ == "__main__":
    main()

"""
FastAPI endpoint for real-time predictive maintenance inference.

Loads a pre-trained XGBoost model and scaler, accepts sensor readings,
and returns a cost-based maintenance recommendation.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Predictive Maintenance API")

# Pre-trained artifacts exported by main.py
model = joblib.load("output/model.joblib")
scaler = joblib.load("output/scaler.joblib")

# Cost-optimized threshold from evaluation grid search (see main.py step 5)
OPTIMAL_THRESHOLD: float = 0.18
COST_UNPLANNED_DOWNTIME: float = 5_000.0
COST_PLANNED_INSPECTION: float = 200.0


class SensorData(BaseModel):
    """Incoming sensor feature vector (must match training feature order)."""
    metrics: list[float]


@app.post("/predict")
def predict_maintenance(data: SensorData) -> dict[str, str]:
    """Return failure probability, risk cost, and recommended action."""
    input_scaled: np.ndarray = scaler.transform([data.metrics])

    prob: float = float(model.predict_proba(input_scaled)[0][1])
    prediction: int = int(prob > OPTIMAL_THRESHOLD)

    # Expected monetary risk = P(failure) × cost of unplanned downtime
    risk_cost: float = prob * COST_UNPLANNED_DOWNTIME
    action: str = "Sofortige Inspektion empfohlen" if prediction == 1 else "Normalbetrieb"
    maintenance_cost: float = COST_PLANNED_INSPECTION if prediction == 1 else 0.0

    return {
        "failure_probability": f"{prob:.2%}",
        "estimated_risk_cost": f"{risk_cost:.2f} €",
        "recommended_action": action,
        "maintenance_cost": f"{maintenance_cost:.2f} €",
    }
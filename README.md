# Predictive Maintenance – Methodology

## Executive Summary

Machine failure prediction on an industrial sensor dataset with extreme class imbalance (0.09% failure rate). The final XGBoost model, combined with time-series feature engineering and cost-optimized thresholding, reduces total maintenance costs by approximately 11% compared to a reactive run-to-failure strategy.

## Business Problem

Rare but costly machine failures create an asymmetric cost structure:

| Scenario | Cost | Description |
|---|---|---|
| False Negative (missed failure) | 5,000 € | Unplanned downtime, emergency repair |
| False Positive (unnecessary inspection) | 200 € | Preventive check on healthy machine |
| **Cost ratio** | **25:1** | Missing a failure is 25x more expensive than a false alarm |

At 0.09% failure rate, a naive classifier predicting "no failure" for every observation achieves 99.91% accuracy while providing zero business value. Standard accuracy is therefore deliberately excluded from evaluation.

## Feature Engineering

### Pre-Failure Window (Target Relabeling)

Raw sensor readings on the exact failure day are often indistinguishable from normal operation. Instead of predicting the point-in-time failure event, a 7-day pre-failure window is created: for each failure, the preceding 7 days are also labeled as `target=1`. This models the degradation phase leading up to failure, giving the classifier a learnable signal.

**Implementation:** Reverse rolling max over the `failure` column, grouped by device.

### Rolling Statistics

Sensor metrics alone capture instantaneous readings but miss trends. Rolling means over 3-day and 7-day windows (per device) expose short-term fluctuations and longer-term drift patterns. This expands the feature space from 9 raw metrics to 27 features (9 original + 9 × 2 rolling windows).

## Validation Strategy

### Why Not Random Split?

A random train/test split on time-series data from multiple devices introduces **temporal leakage**: the model could memorize device-specific patterns from training samples and "predict" test samples from the same device's timeline.

### GroupShuffleSplit on Device Level

Complete devices are assigned exclusively to either the training or test set using `GroupShuffleSplit(groups=device)`. This ensures the model is evaluated on entirely unseen devices, simulating real-world deployment where predictions must generalize to new machines.

### Feature Selection

`SelectFromModel` with a lightweight RandomForest selector reduces the 27 engineered features to the top 15 by importance. This removes noise from low-signal metrics (e.g., metric1, metric3) that would otherwise increase false positives without improving recall.

## Modeling Approach

### Why XGBoost?

- **Native imbalance handling:** `scale_pos_weight = count(class_0) / count(class_1)` adjusts the loss function directly, penalizing missed failures proportionally to the class ratio. This eliminates the need for synthetic oversampling (SMOTE) and its associated artifacts.
- **Gradient boosting** iteratively corrects errors from previous rounds, making it more effective at learning rare-event patterns than bagging-based methods (Random Forest) on tabular data.

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 200 | Sufficient boosting rounds for convergence |
| `learning_rate` | 0.1 | Standard shrinkage, balances speed and generalization |
| `max_depth` | 6 | Controls tree complexity, prevents overfitting |
| `scale_pos_weight` | dynamic | Computed from training data class ratio |

## Economic Evaluation

### Cost-Optimized Threshold

The default classification threshold of 0.5 assumes symmetric costs. Given the 25:1 cost ratio (FN vs. FP), the optimal threshold is found via grid search over [0.01, 0.99], selecting the value that minimizes **total monetary cost** rather than maximizing accuracy or F1-score.

### Benchmark: Run-to-Failure

The baseline comparison is a reactive strategy where no predictive model is used. All actual failures (TP + FN) result in unplanned downtime at 5,000 € each. The ML model's value is measured as the cost difference between this baseline and the model-assisted strategy.

## Project Structure

```
predictive_maintenance/
├── main.py                  # Pipeline orchestration
├── feature_engineering.py   # Pre-failure windows, rolling features
├── preprocessing.py         # GroupShuffleSplit, scaling, feature selection
├── model.py                 # XGBoost training
├── evaluation.py            # Metrics, threshold optimization, plots
├── economic_analysis.py     # Cost comparison vs. run-to-failure
├── data.csv                 # Input data (not included)
└── output/                  # Generated plots
```

## Requirements

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
```
## Data Source
The dataset used for this project is publicly available on Kaggle: [Predictive Maintenance Dataset](https://www.kaggle.com/datasets/hiimanshuagarwal/predictive-maintenance-dataset). 

*Note: The raw `data.csv` is not included in this repository due to size constraints and best practices. To run the code locally, please download the dataset from Kaggle, rename it to `data.csv` (if necessary), and place it in the root directory of this project.*

## Usage

```bash
pip install -r requirements.txt
python main.py              # uses data.csv in current directory
python main.py path/to/data.csv  # custom path
```

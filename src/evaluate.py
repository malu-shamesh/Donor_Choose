"""Evaluate a trained classifier and save ROC/confusion matrix reports."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve


def evaluate_and_report(model, X, y, output_dir: Path, split_name: str) -> dict:
    """Predict on X, save ROC curve data and confusion matrix, return metrics."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    fpr, tpr, thresholds = roc_curve(y, probabilities)
    roc_auc = auc(fpr, tpr)

    matrix = confusion_matrix(y, predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
        output_dir / f"roc_curve_{split_name}.csv", index=False
    )
    pd.DataFrame(matrix, index=["not_approved", "approved"], columns=["not_approved", "approved"]).to_csv(
        output_dir / f"confusion_matrix_{split_name}.csv"
    )

    return {"roc_auc": float(roc_auc), "fpr": fpr, "tpr": tpr}


def best_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> float:
    """The threshold maximizing tpr * (1 - fpr), balancing true positives against false positives."""
    return thresholds[np.argmax(tpr * (1 - fpr))]

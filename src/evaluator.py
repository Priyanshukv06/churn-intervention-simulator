from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute core binary classification metrics given true labels and
    predicted probabilities.
    """
    y_pred = (y_proba >= threshold).astype(int)

    roc = roc_auc_score(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision_score * recall_score / (precision_score + recall_score)
        if (precision_score + recall_score) > 0
        else 0.0
    )

    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "accuracy": float(accuracy),
        "precision": float(precision_score),
        "recall": float(recall_score),
        "f1": float(f1),
    }


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str | Path,
    threshold: float = 0.5,
) -> None:
    """Save a confusion matrix heatmap to PNG."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    save_path = Path(save_path)
    _ensure_dir(save_path.parent)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str | Path,
) -> None:
    """Save ROC curve to PNG."""
    fig, ax = plt.subplots(figsize=(4, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()

    save_path = Path(save_path)
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str | Path,
) -> None:
    """Save Precision-Recall curve to PNG."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision, label="PR curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(True)
    fig.tight_layout()

    save_path = Path(save_path)
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

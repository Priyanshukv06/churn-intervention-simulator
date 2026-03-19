from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from xgboost import XGBClassifier
import yaml

from src.data_loader import load_and_prepare_data
from src.evaluator import (
    compute_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """
    Heuristic: negatives / positives [web:103][web:97].
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    if pos == 0:
        raise ValueError("No positive samples in y_train; cannot compute scale_pos_weight.")
    return float(neg / pos)


def build_model(xgb_params: Dict, scale_pos_weight: float) -> XGBClassifier:
    """
    Build an XGBClassifier with given params and scale_pos_weight.
    """
    params = dict(xgb_params)  # copy
    params["scale_pos_weight"] = scale_pos_weight

    model = XGBClassifier(**params)
    return model


def train_model(
    model: XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> XGBClassifier:
    """
    Train XGBoost with evaluation set for early stopping if desired.
    """
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,
    )
    return model


def save_model(model: XGBClassifier, path: str | Path) -> None:
    """
    Save trained XGBClassifier to disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def run_training_pipeline(config_path: str | Path = "configs/config.yaml") -> None:
    """
    End-to-end training procedure:
    - Load config
    - Load & split data
    - Build & fit preprocessor
    - Transform features
    - Train XGBoost with scale_pos_weight
    - Save model and basic evaluation plots
    """
    from src.preprocessor import fit_and_save_preprocessor, transform_features

    config = load_config(config_path)
    data_cfg = config["data"]
    model_cfg = config["model"]
    eval_cfg = config["eval"]

    # 1) Load & split data
    X_train_df, X_test_df, y_train_sr, y_test_sr = load_and_prepare_data(config_path)
    y_train = y_train_sr.to_numpy()
    y_test = y_test_sr.to_numpy()

    # 2) Fit and save preprocessor
    preprocessor_path = model_cfg["preprocessor_save_path"]
    preprocessor = fit_and_save_preprocessor(X_train_df, preprocessor_path)

    # 3) Transform features
    X_train = transform_features(preprocessor, X_train_df)
    X_test = transform_features(preprocessor, X_test_df)

    # 4) Compute class imbalance weight
    spw = compute_scale_pos_weight(y_train)
    print(f"scale_pos_weight (neg/pos): {spw:.3f}")

    # 5) Build model (GPU if available via config: device='cuda') [web:92][web:99]
    xgb_params = model_cfg["xgb_params"]
    model = build_model(xgb_params, spw)

    # 6) Train model
    model = train_model(model, X_train, y_train, X_test, y_test)

    # 7) Save model
    model_path = model_cfg["model_save_path"]
    save_model(model, model_path)
    print(f"Model saved to {model_path}")
    print(f"Preprocessor saved to {preprocessor_path}")

    # 8) Evaluate
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_proba)
    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # 9) Save evaluation plots
    artifacts_dir = Path(eval_cfg["artifacts_path"])
    plot_roc_curve(y_test, y_proba, artifacts_dir / "roc_curve.png")
    plot_pr_curve(y_test, y_proba, artifacts_dir / "pr_curve.png")
    plot_confusion_matrix(y_test, y_proba, artifacts_dir / "confusion_matrix.png")
    print(f"Evaluation plots saved under {artifacts_dir}")


if __name__ == "__main__":
    run_training_pipeline()

"""
Matching model: Train a LightGBM classifier to distinguish true matches
from non-matches based on the feature vectors. Includes threshold tuning
for F0.5 optimization.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import logging
import pickle
import os

logger = logging.getLogger(__name__)


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F-beta score."""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def compute_entity_level_f05(
    predictions: dict,
    ground_truth: dict,
) -> float:
    """
    Compute macro-averaged F0.5 at entity level.

    Args:
        predictions: dict s1_id -> set of predicted match ids
        ground_truth: dict s1_id -> set of true match ids
    """
    scores = []
    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, set())

        # Singleton handling
        if not true_matches and not pred_matches:
            scores.append(1.0)
            continue
        if not true_matches and pred_matches:
            scores.append(0.0)
            continue

        if not pred_matches:
            scores.append(0.0)
            continue

        tp = len(pred_matches & true_matches)
        fp = len(pred_matches - true_matches)
        fn = len(true_matches - pred_matches)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05 = compute_f_beta(precision, recall, beta=0.5)
        scores.append(f05)

    return np.mean(scores) if scores else 0.0


def train_model(
    feature_df: pd.DataFrame,
    feature_cols: list,
    model_path: str = None,
    n_folds: int = 5,
) -> tuple:
    """
    Train a LightGBM model with cross-validation for threshold selection.

    Returns (model, best_threshold)
    """
    logger.info("Training LightGBM model...")

    X = feature_df[feature_cols].values
    y = feature_df["label"].values

    # Handle class imbalance
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": scale_pos_weight,
        "min_child_samples": 20,
        "max_depth": -1,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    # Cross-validation for threshold tuning
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"  Fold {fold + 1}/{n_folds}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=200),
            ],
        )
        oof_preds[val_idx] = model.predict(X_val)

    # Find optimal threshold on OOF predictions
    # Evaluate at entity level using the OOF predictions
    best_threshold = _find_best_threshold_entity_level(
        feature_df, oof_preds, feature_cols
    )
    logger.info(f"  Best threshold from CV: {best_threshold:.4f}")

    # Retrain on full data
    logger.info("  Retraining on full data...")
    full_train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        params,
        full_train_data,
        num_boost_round=800,  # Use a reasonable round count
    )

    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({"model": final_model, "threshold": best_threshold}, f)
        logger.info(f"  Model saved to {model_path}")

    # Feature importance
    importance = final_model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    logger.info("  Top 10 features by gain:")
    for fname, imp in feat_imp[:10]:
        logger.info(f"    {fname}: {imp:.1f}")

    return final_model, best_threshold


def _find_best_threshold_entity_level(
    feature_df: pd.DataFrame,
    oof_preds: np.ndarray,
    feature_cols: list,
) -> float:
    """
    Find the threshold that maximizes macro-averaged F0.5 on OOF predictions.
    Uses entity-level evaluation.
    """
    # Try thresholds from 0.1 to 0.9
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_f05 = 0.0
    best_thresh = 0.5

    for thresh in thresholds:
        # Build predictions dict
        predictions = {}
        for i, row in feature_df.iterrows():
            s1_id = row["s1_id"]
            s2s3_id = row["s2s3_id"]
            if oof_preds[feature_df.index.get_loc(i)] >= thresh:
                if s1_id not in predictions:
                    predictions[s1_id] = set()
                predictions[s1_id].add(s2s3_id)

        # Build ground truth dict from the feature df
        ground_truth = {}
        for _, row in feature_df.iterrows():
            s1_id = row["s1_id"]
            if s1_id not in ground_truth:
                ground_truth[s1_id] = set()
            if row["label"] == 1:
                ground_truth[s1_id].add(row["s2s3_id"])

        f05 = compute_entity_level_f05(predictions, ground_truth)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh

    logger.info(f"  Threshold search: best F0.5 = {best_f05:.4f} at threshold = {best_thresh:.4f}")
    return best_thresh


def predict_matches(
    model,
    feature_df: pd.DataFrame,
    feature_cols: list,
    threshold: float,
    s1_entity_ids: list,
) -> dict:
    """
    Use trained model to predict matches.

    Returns dict: s1_id -> set of matched s2s3 ids
    """
    logger.info(f"Predicting matches with threshold={threshold:.4f}...")

    if len(feature_df) == 0:
        return {s1_id: set() for s1_id in s1_entity_ids}

    X = feature_df[feature_cols].values
    probs = model.predict(X)

    predictions = {}
    for i, (_, row) in enumerate(feature_df.iterrows()):
        s1_id = row["s1_id"]
        s2s3_id = row["s2s3_id"]
        if probs[i] >= threshold:
            if s1_id not in predictions:
                predictions[s1_id] = set()
            predictions[s1_id].add(s2s3_id)

    # Ensure every S1 entity has an entry
    for s1_id in s1_entity_ids:
        if s1_id not in predictions:
            predictions[s1_id] = set()

    n_matched = sum(1 for v in predictions.values() if v)
    total_matches = sum(len(v) for v in predictions.values())
    logger.info(f"  {n_matched}/{len(predictions)} S1 entities have matches, {total_matches} total match links")

    return predictions


def load_model(model_path: str) -> tuple:
    """Load a saved model and threshold."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["threshold"]

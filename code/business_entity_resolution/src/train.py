"""Train the entity matcher with a randomized LightGBM hyperparameter search.

Run from ``code/business_entity_resolution``::

    python src/train.py --data-dir ../../dataset --model-path models/lgbm_model.pkl
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blocking import generate_candidates, evaluate_blocking_recall
from features import build_feature_matrix, get_feature_columns
from preprocessing import (
    load_source,
    load_ground_truth,
    parse_ground_truth,
    preprocess_dataframe,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_randomized_model(
    features: pd.DataFrame,
    model_path=None,
    n_iter: int = 20,
    n_splits: int = 5,
    seed: int = 42,
):
    """Search LightGBM settings and return a Booster and decision threshold."""
    if features.empty or features["label"].nunique() != 2:
        raise ValueError("Candidate features must contain both positive and negative labels")

    feature_cols = get_feature_columns(features)
    X = features[feature_cols]
    y = features["label"].astype(int)
    groups = features["s1_id"]
    n_groups = groups.nunique()
    if n_splits > n_groups:
        raise ValueError(f"n_splits={n_splits} exceeds the {n_groups} distinct source1 entities")

    estimator = LGBMClassifier(
        objective="binary", class_weight="balanced", n_jobs=1,
        random_state=seed, verbosity=-1, subsample_freq=1,
    )
    parameters = {
        "n_estimators": randint(200, 1201),
        "num_leaves": randint(15, 128),
        "max_depth": [-1, 5, 8, 12, 16],
        "learning_rate": loguniform(0.01, 0.15),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_samples": randint(10, 101),
        "reg_alpha": loguniform(1e-4, 10),
        "reg_lambda": loguniform(1e-4, 10),
    }
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=estimator, param_distributions=parameters, n_iter=n_iter,
        scoring=make_scorer(fbeta_score, beta=0.5, zero_division=0), cv=cv,
        refit=True, n_jobs=-1, random_state=seed, verbose=1, error_score="raise",
    )
    logger.info("Randomized search: %d settings, %d grouped folds, %d candidate pairs",
                n_iter, n_splits, len(features))
    search.fit(X, y, groups=groups)
    logger.info("Best mean CV F0.5: %.4f", search.best_score_)
    logger.info("Best parameters: %s", search.best_params_)

    booster = search.best_estimator_.booster_
    threshold = 0.5
    if model_path:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as model_file:
            pickle.dump({"model": booster, "threshold": threshold}, model_file)
        logger.info("Saved compatible model artifact to %s", model_path)
    return booster, threshold


def main():
    parser = argparse.ArgumentParser(
        description="Train the entity matcher using RandomizedSearchCV"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("../../dataset"))
    parser.add_argument("--model-path", type=Path, default=Path("models/lgbm_model.pkl"))
    parser.add_argument("--n-iter", type=int, default=20,
                        help="Number of parameter settings sampled")
    parser.add_argument("--cv", type=int, default=5,
                        help="Number of grouped cross-validation folds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-word", type=int, default=50)
    parser.add_argument("--top-k-char", type=int, default=30)
    parser.add_argument("--min-shared-tokens", type=int, default=2)
    args = parser.parse_args()

    if args.n_iter < 1 or args.cv < 2:
        parser.error("--n-iter must be positive and --cv must be at least 2")

    data_dir = args.data_dir
    logger.info("Loading and preprocessing training data from %s", data_dir)
    s1 = preprocess_dataframe(load_source(data_dir / "train" / "train_source1.tsv"))
    s2 = preprocess_dataframe(load_source(data_dir / "train" / "train_source2.tsv"))
    s3 = preprocess_dataframe(load_source(data_dir / "train" / "train_source3.tsv"))
    ground_truth = parse_ground_truth(
        load_ground_truth(data_dir / "train" / "train_ground_truth.tsv")
    )

    logger.info("Generating candidate pairs")
    candidates = generate_candidates(
        s1, s2, s3,
        top_k_word=args.top_k_word,
        top_k_char=args.top_k_char,
        min_shared_tokens=args.min_shared_tokens,
    )
    recall = evaluate_blocking_recall(candidates, ground_truth)
    logger.info("Blocking recall: %.4f", recall)

    features = build_feature_matrix(
        s1, pd.concat([s2, s3], ignore_index=True), candidates, ground_truth
    )
    if features.empty or features["label"].nunique() != 2:
        raise ValueError("Candidate features must contain both positive and negative labels")

    train_randomized_model(
        features, model_path=args.model_path, n_iter=args.n_iter,
        n_splits=args.cv, seed=args.seed,
    )


if __name__ == "__main__":
    main()

"""
Blocking / Candidate Generation for entity resolution.

Uses multiple blocking strategies combined:
1. TF-IDF cosine similarity on combined name+address text (within same country)
2. Character n-gram TF-IDF for fuzzy name matching
3. Exact/near-exact name token blocking as a fallback

The goal: high recall (capture all true matches) with reasonable reduction ratio.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def _build_tfidf_candidates(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    text_col: str = "combined_text",
    top_k: int = 50,
    ngram_range: tuple = (1, 2),
    analyzer: str = "word",
    min_score: float = 0.05,
) -> dict:
    """
    Use TF-IDF cosine similarity to find top-k candidates from S2/S3
    for each S1 entity.

    Returns dict: s1_entity_id -> list of (s2s3_entity_id, score)
    """
    logger.info(f"Building TF-IDF candidates (analyzer={analyzer}, ngrams={ngram_range}, top_k={top_k})...")

    all_texts = pd.concat([s1_df[text_col], s2s3_df[text_col]], ignore_index=True)

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=100000,
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    s1_vectors = tfidf_matrix[: len(s1_df)]
    s2s3_vectors = tfidf_matrix[len(s1_df):]

    s1_ids = s1_df["entity_id"].values
    s2s3_ids = s2s3_df["entity_id"].values

    candidates = {}

    # Process in batches to avoid memory issues
    batch_size = 500
    for start in range(0, len(s1_df), batch_size):
        end = min(start + batch_size, len(s1_df))
        batch_vectors = s1_vectors[start:end]

        # Cosine similarity (vectors are already L2-normalized by TF-IDF)
        sim_matrix = (batch_vectors @ s2s3_vectors.T).toarray()

        for i in range(end - start):
            scores = sim_matrix[i]
            # Get top-k indices
            if top_k < len(scores):
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
            else:
                top_indices = np.arange(len(scores))

            s1_id = s1_ids[start + i]
            cands = []
            for idx in top_indices:
                if scores[idx] >= min_score:
                    cands.append((s2s3_ids[idx], float(scores[idx])))
            candidates[s1_id] = cands

    logger.info(f"  TF-IDF ({analyzer}) generated candidates for {len(candidates)} S1 entities")
    return candidates


def _build_char_ngram_candidates(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    top_k: int = 30,
    min_score: float = 0.1,
) -> dict:
    """Character n-gram TF-IDF for catching typos and transliterations."""
    return _build_tfidf_candidates(
        s1_df, s2s3_df,
        text_col="name_norm",
        top_k=top_k,
        ngram_range=(2, 4),
        analyzer="char_wb",
        min_score=min_score,
    )


def _build_token_blocking_candidates(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    min_shared_tokens: int = 2,
    max_block_size: int = 500,
) -> dict:
    """
    Token blocking: if S1 and S2/S3 share at least min_shared_tokens
    name tokens, they are candidates.
    """
    logger.info("Building token blocking candidates...")

    # Build inverted index for S2/S3
    inverted_index = defaultdict(set)
    for _, row in s2s3_df.iterrows():
        tokens = set(row["name_norm"].split())
        for tok in tokens:
            if len(tok) >= 3:  # Skip very short tokens
                inverted_index[tok].add(row["entity_id"])

    # Prune blocks that are too large (common words)
    inverted_index = {
        tok: ids for tok, ids in inverted_index.items()
        if len(ids) <= max_block_size
    }

    candidates = {}
    for _, row in s1_df.iterrows():
        s1_id = row["entity_id"]
        tokens = set(row["name_norm"].split())
        candidate_counts = defaultdict(int)
        for tok in tokens:
            if tok in inverted_index:
                for s2s3_id in inverted_index[tok]:
                    candidate_counts[s2s3_id] += 1

        cands = [
            (cid, count)
            for cid, count in candidate_counts.items()
            if count >= min_shared_tokens
        ]
        candidates[s1_id] = cands

    logger.info(f"  Token blocking generated candidates for {len(candidates)} S1 entities")
    return candidates


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    top_k_word: int = 50,
    top_k_char: int = 30,
    min_shared_tokens: int = 2,
) -> dict:
    """
    Main blocking function. Combines multiple blocking strategies
    and filters by country.

    Returns dict: s1_entity_id -> set of candidate s2/s3 entity_ids
    """
    logger.info("Starting candidate generation...")

    # Combine S2 and S3
    s2s3_df = pd.concat([s2_df, s3_df], ignore_index=True)

    # Group by country for country-aware blocking
    s1_countries = s1_df["country_norm"].unique()
    s2s3_countries = s2s3_df["country_norm"].unique()
    all_countries = set(list(s1_countries) + list(s2s3_countries))

    all_candidates = defaultdict(set)

    for country in all_countries:
        s1_country = s1_df[s1_df["country_norm"] == country]
        s2s3_country = s2s3_df[s2s3_df["country_norm"] == country]

        if len(s1_country) == 0 or len(s2s3_country) == 0:
            logger.info(f"  Skipping country '{country}': S1={len(s1_country)}, S2S3={len(s2s3_country)}")
            continue

        logger.info(f"  Processing country '{country}': S1={len(s1_country)}, S2S3={len(s2s3_country)}")

        # Strategy 1: Word-level TF-IDF on combined text
        word_cands = _build_tfidf_candidates(
            s1_country, s2s3_country,
            text_col="combined_text",
            top_k=top_k_word,
            ngram_range=(1, 2),
            analyzer="word",
            min_score=0.05,
        )
        for s1_id, cands in word_cands.items():
            for cid, _ in cands:
                all_candidates[s1_id].add(cid)

        # Strategy 2: Char n-gram TF-IDF on name
        char_cands = _build_char_ngram_candidates(
            s1_country, s2s3_country,
            top_k=top_k_char,
            min_score=0.1,
        )
        for s1_id, cands in char_cands.items():
            for cid, _ in cands:
                all_candidates[s1_id].add(cid)

        # Strategy 3: Token blocking on name
        tok_cands = _build_token_blocking_candidates(
            s1_country, s2s3_country,
            min_shared_tokens=min_shared_tokens,
        )
        for s1_id, cands in tok_cands.items():
            for cid, _ in cands:
                all_candidates[s1_id].add(cid)

    # Ensure every S1 entity has an entry (even if empty)
    for s1_id in s1_df["entity_id"].values:
        if s1_id not in all_candidates:
            all_candidates[s1_id] = set()

    total_pairs = sum(len(v) for v in all_candidates.values())
    avg_cands = total_pairs / max(len(all_candidates), 1)
    logger.info(f"Candidate generation complete: {total_pairs} total pairs, avg {avg_cands:.1f} per S1 entity")

    return dict(all_candidates)


def evaluate_blocking_recall(candidates: dict, ground_truth: dict) -> float:
    """
    Compute blocking recall: fraction of true matches that appear
    in the candidate set.
    """
    total_true = 0
    found = 0
    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue
        cands = candidates.get(s1_id, set())
        for m in true_matches:
            total_true += 1
            if m in cands:
                found += 1
    recall = found / max(total_true, 1)
    logger.info(f"Blocking recall: {recall:.4f} ({found}/{total_true})")
    return recall

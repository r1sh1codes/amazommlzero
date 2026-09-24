# Business Entity Resolution

## Setup

```bash
pip install -r requirements.txt
```

## Directory Structure

Place the dataset at the expected path relative to this directory:

```
amazonml/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── output/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── main.py
│       │   ├── preprocessing.py
│       │   ├── blocking.py
│       │   ├── features.py
│       │   └── matcher.py
│       ├── requirements.txt
│       └── README.md
└── utils/
    └── validate_submission.py
```

## Usage

All commands run from `code/business_entity_resolution/`.

### Full pipeline (train + predict on test set)

```bash
python src/main.py --data-dir ../../dataset --output-dir ../../output
```

### Validation only (evaluate on held-out training split)

```bash
python src/main.py --data-dir ../../dataset --output-dir ../../output --validate-only --val-fraction 0.2
```

### Skip training (use saved model)

```bash
python src/main.py --data-dir ../../dataset --output-dir ../../output --skip-train
```

### Override threshold

```bash
python src/main.py --data-dir ../../dataset --output-dir ../../output --threshold 0.6
```

### Tune blocking parameters

```bash
python src/main.py --data-dir ../../dataset --output-dir ../../output --top-k-word 80 --top-k-char 50 --min-shared-tokens 1
```

## Outputs

- `output/matching_results.tsv` - final entity matches (upload to leaderboard)
- `output/candidate_pairs.tsv` - blocking candidate set

## Approach

1. **Preprocessing**: Unicode normalization, abbreviation expansion, punctuation removal, legal suffix stripping
2. **Blocking** (3 strategies combined, per-country):
   - Word-level TF-IDF cosine on name+address (top-K)
   - Character n-gram TF-IDF on name (catches typos/transliterations)
   - Token blocking on name (shared token overlap)
3. **Feature Engineering**: 30+ string similarity features per pair (Levenshtein, Jaro-Winkler, Jaccard, cosine, char-ngram, number overlap, etc.)
4. **Matching**: LightGBM classifier with auto-threshold tuning for F0.5

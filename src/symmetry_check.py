# ============================================================
# Week 3c — Symmetry Check for single_flip
# Question: does the model fail equally on BOTH flip directions
# (high tone -> low tone, and low tone -> high tone), or is it
# secretly better at detecting one direction than the other?
#
# Reuses the embeddings already cached by week3b_breakdown.py --
# no model rerun needed, this is a pure re-analysis step.
# ============================================================

import unicodedata
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import balanced_accuracy_score

# reuse the extraction/caching machinery from the previous script
from week3b_breakdown import get_or_extract_embeddings, MODELS

HIGH = "\u0301"  # acute (high tone)
LOW = "\u0300"   # grave (low tone)


def detect_flip_direction(correct_sentence: str, flipped_sentence: str):
    """Compares the original sentence to its single_flip corrupted version
    and works out which direction the tone was flipped: 'high_to_low',
    'low_to_high', or None if no single clean tone-mark difference is found
    (shouldn't normally happen, but we guard against it just in case)."""
    a = list(unicodedata.normalize("NFD", correct_sentence))
    b = list(unicodedata.normalize("NFD", flipped_sentence))

    if len(a) != len(b):
        return None  # unexpected -- lengths should match for a single-char swap

    diffs = [i for i in range(len(a)) if a[i] != b[i]]
    if len(diffs) != 1:
        return None  # expect exactly one differing character position

    pos = diffs[0]
    original_char, new_char = a[pos], b[pos]

    if original_char == HIGH and new_char == LOW:
        return "high_to_low"
    elif original_char == LOW and new_char == HIGH:
        return "low_to_high"
    else:
        return None


def add_flip_direction_column(df: pd.DataFrame) -> pd.DataFrame:
    """For every single_flip row, look up its correct sibling (same pair_id)
    and record which direction the tone was flipped."""
    df = df.copy()
    df["flip_direction"] = None

    correct_lookup = df[df["label"] == "correct"].set_index("pair_id")["sentence"].to_dict()

    flip_mask = df["corruption_type"] == "single_flip"
    for idx in df[flip_mask].index:
        pair_id = df.at[idx, "pair_id"]
        correct_sentence = correct_lookup.get(pair_id)
        flipped_sentence = df.at[idx, "sentence"]
        if correct_sentence is None:
            continue
        direction = detect_flip_direction(correct_sentence, flipped_sentence)
        df.at[idx, "flip_direction"] = direction

    n_found = df[flip_mask]["flip_direction"].notna().sum()
    n_total = flip_mask.sum()
    print(f"Detected flip direction for {n_found}/{n_total} single_flip rows")
    print(df[flip_mask]["flip_direction"].value_counts(dropna=False))
    return df


def evaluate_symmetry(layer_vectors, df, n_repeats=20):
    """For each layer, evaluate balanced accuracy separately on
    high_to_low flips vs low_to_high flips (each paired against the
    correct examples), to check whether the model's blindness to
    single_flip is symmetric or direction-dependent."""
    y = np.array([1 if l == "correct" else 0 for l in df["label"]])
    groups = df["pair_id"].values
    flip_direction = df["flip_direction"].values
    corruption_type = df["corruption_type"].values

    rows = []
    for layer_idx, X in enumerate(layer_vectors):
        for repeat in range(n_repeats):
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=repeat)
            train_idx, test_idx = next(splitter.split(X, y, groups=groups))

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            test_direction = flip_direction[test_idx]
            test_ctype = corruption_type[test_idx]

            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)

            for direction in ["high_to_low", "low_to_high"]:
                subset_mask = (test_direction == direction) | (y_test == 1)
                if subset_mask.sum() < 4 or len(np.unique(y_test[subset_mask])) < 2:
                    continue
                rows.append({
                    "layer": layer_idx, "repeat": repeat, "flip_direction": direction,
                    "balanced_accuracy": balanced_accuracy_score(y_test[subset_mask], preds[subset_mask]),
                })

            # also keep the pooled single_flip number (both directions together)
            # as a reference point, same definition as in week3b_breakdown.py
            subset_mask = (test_ctype == "single_flip") | (y_test == 1)
            if subset_mask.sum() >= 4 and len(np.unique(y_test[subset_mask])) == 2:
                rows.append({
                    "layer": layer_idx, "repeat": repeat, "flip_direction": "both_pooled",
                    "balanced_accuracy": balanced_accuracy_score(y_test[subset_mask], preds[subset_mask]),
                })

    raw = pd.DataFrame(rows)
    summary = raw.groupby(["layer", "flip_direction"], as_index=False).agg(
        balanced_accuracy=("balanced_accuracy", "mean")
    )
    return summary


def run_symmetry_check(pairs_csv="yoruba_tone_pairs.csv"):
    df = pd.read_csv(pairs_csv)
    df = add_flip_direction_column(df)
    sentences = df["sentence"].tolist()

    all_summaries = {}
    for model_key, model_name in MODELS.items():
        print(f"\n=== {model_key} ===")
        # this will load from embedding_cache/ automatically if week3b already ran
        layer_vectors = get_or_extract_embeddings(sentences, model_key, model_name)
        summary = evaluate_symmetry(layer_vectors, df)
        summary["model"] = model_key
        all_summaries[model_key] = summary

        pivot = summary.pivot(index="layer", columns="flip_direction", values="balanced_accuracy").round(3)
        print(pivot)

        # simple symmetry check: average absolute gap between the two directions across layers
        if "high_to_low" in pivot.columns and "low_to_high" in pivot.columns:
            gap = (pivot["high_to_low"] - pivot["low_to_high"]).abs().mean()
            print(f"Average |high_to_low - low_to_high| gap across layers: {gap:.3f}")
            if gap < 0.03:
                print("-> Looks SYMMETRIC (both directions equally undetectable)")
            else:
                print("-> Looks ASYMMETRIC (one direction is more detectable than the other)")

    combined = pd.concat(all_summaries.values(), ignore_index=True)
    combined.to_csv("tone_flip_symmetry_results.csv", index=False)
    print("\nSaved to tone_flip_symmetry_results.csv")
    return combined


if __name__ == "__main__":
    results = run_symmetry_check()

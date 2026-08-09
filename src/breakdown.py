# Breaking Down Probe Accuracy by Corruption Type
# Answers: is the probe detecting real fine-grained tone knowledge,
# or just noticing "lots of accents went missing" (a surface artifact)?


import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

MODELS = {
    "afro-xlmr-base": "Davlan/afro-xlmr-base",
    "xlm-roberta-base": "xlm-roberta-base",
}

CACHE_DIR = "embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def extract_all_layer_embeddings(sentences, model_name, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()

    all_layer_vectors = None

    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start:start + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                             max_length=64, return_tensors="pt").to(device)
            out = model(**enc)
            hidden_states = out.hidden_states
            mask = enc["attention_mask"].unsqueeze(-1)

            if all_layer_vectors is None:
                all_layer_vectors = [[] for _ in hidden_states]

            for layer_idx, layer_hs in enumerate(hidden_states):
                summed = (layer_hs * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                pooled = (summed / counts).cpu().numpy()
                all_layer_vectors[layer_idx].append(pooled)

            if start % (batch_size * 10) == 0:
                print(f"  {model_name}: {start}/{len(sentences)} sentences done")

    all_layer_vectors = [np.concatenate(b, axis=0) for b in all_layer_vectors]
    del model
    torch.cuda.empty_cache()
    return all_layer_vectors


def get_or_extract_embeddings(sentences, model_key, model_name):
    """Checks the cache first -- only calls the (slow) model if needed."""
    cache_path = os.path.join(CACHE_DIR, f"{model_key}_embeddings.npz")
    if os.path.exists(cache_path):
        print(f"  Loading cached embeddings for {model_key} from {cache_path}")
        data = np.load(cache_path)
        return [data[f"layer_{i}"] for i in range(len(data.files))]

    print(f"  No cache found for {model_key} -- running model (this is the slow step)")
    layer_vectors = extract_all_layer_embeddings(sentences, model_name)
    save_dict = {f"layer_{i}": v for i, v in enumerate(layer_vectors)}
    np.savez_compressed(cache_path, **save_dict)
    print(f"  Cached embeddings to {cache_path} for next time")
    return layer_vectors


def evaluate_by_corruption_type(layer_vectors, df, n_repeats=3):
    """For each layer, trains one real probe + one control (shuffled) probe,
    then reports balanced accuracy SEPARATELY for each corruption_type,
    instead of lumping all 'incorrect' examples together."""
    y = np.array([1 if l == "correct" else 0 for l in df["label"]])
    groups = df["pair_id"].values
    corruption_types = df["corruption_type"].values
    real_corruption_types = [c for c in df["corruption_type"].unique() if c != "none"]

    rows = []
    for layer_idx, X in enumerate(layer_vectors):
        for repeat in range(n_repeats):
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=repeat)
            train_idx, test_idx = next(splitter.split(X, y, groups=groups))

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            test_corruption = corruption_types[test_idx]

            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(X_train, y_train)

            y_train_shuffled = np.random.RandomState(repeat).permutation(y_train)
            clf_control = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf_control.fit(X_train, y_train_shuffled)

            preds_real = clf.predict(X_test)
            preds_control = clf_control.predict(X_test)

            # overall (all corruption types pooled) -- same as the original script
            rows.append({
                "layer": layer_idx, "repeat": repeat, "corruption_type": "ALL",
                "real_balanced_accuracy": balanced_accuracy_score(y_test, preds_real),
                "control_balanced_accuracy": balanced_accuracy_score(y_test, preds_control),
            })

            # per corruption type: correct examples + only THIS type of incorrect example
            for ctype in real_corruption_types:
                subset_mask = (test_corruption == ctype) | (y_test == 1)
                if subset_mask.sum() < 4 or len(np.unique(y_test[subset_mask])) < 2:
                    continue  # skip if too few examples or only one class present
                rows.append({
                    "layer": layer_idx, "repeat": repeat, "corruption_type": ctype,
                    "real_balanced_accuracy": balanced_accuracy_score(y_test[subset_mask], preds_real[subset_mask]),
                    "control_balanced_accuracy": balanced_accuracy_score(y_test[subset_mask], preds_control[subset_mask]),
                })

    raw = pd.DataFrame(rows)
    summary = raw.groupby(["layer", "corruption_type"], as_index=False).agg(
        real_balanced_accuracy=("real_balanced_accuracy", "mean"),
        control_balanced_accuracy=("control_balanced_accuracy", "mean"),
    )
    return summary


def plot_breakdown(summary, model_key):
    plt.figure(figsize=(8, 5))
    types = summary["corruption_type"].unique()
    colors = plt.cm.tab10.colors
    for i, ctype in enumerate(sorted(types)):
        sub = summary[summary["corruption_type"] == ctype].sort_values("layer")
        style = "-" if ctype == "ALL" else "-"
        width = 2.5 if ctype == "ALL" else 1.5
        plt.plot(sub["layer"], sub["real_balanced_accuracy"], marker="o",
                  label=ctype, color=colors[i], linewidth=width)
    plt.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance level")
    plt.xlabel("Layer")
    plt.ylabel("Balanced probe accuracy")
    plt.title(f"{model_key}: accuracy by corruption type, per layer")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fname = f"breakdown_{model_key}.png"
    plt.savefig(fname, dpi=150)
    print(f"Saved {fname}")


def run_breakdown_analysis(pairs_csv="yoruba_tone_pairs.csv"):
    df = pd.read_csv(pairs_csv)
    sentences = df["sentence"].tolist()

    all_summaries = {}
    for model_key, model_name in MODELS.items():
        print(f"\n=== {model_key} ===")
        layer_vectors = get_or_extract_embeddings(sentences, model_key, model_name)
        summary = evaluate_by_corruption_type(layer_vectors, df)
        summary["model"] = model_key
        all_summaries[model_key] = summary

        print(summary.pivot(index="layer", columns="corruption_type", values="real_balanced_accuracy").round(3))
        plot_breakdown(summary, model_key)

    combined = pd.concat(all_summaries.values(), ignore_index=True)
    combined.to_csv("tone_probing_breakdown_by_corruption_type.csv", index=False)
    print("\nSaved combined breakdown to tone_probing_breakdown_by_corruption_type.csv")
    return combined


if __name__ == "__main__":
    results = run_breakdown_analysis()

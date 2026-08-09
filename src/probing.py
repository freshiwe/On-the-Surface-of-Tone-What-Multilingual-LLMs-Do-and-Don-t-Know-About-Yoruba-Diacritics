
# Week 3 — Extracting Hidden States and Training Probes
# Run this in Google Colab WITH A GPU (Runtime -> Change runtime type -> GPU)


# --- CELL 1: install what's needed ---
# !pip install transformers torch scikit-learn pandas numpy matplotlib

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



# Step 1: turn each sentence into one vector PER LAYER of the model.
# We mean-pool over the real tokens (ignoring padding) to get a single
# fixed-size vector representing "what the model thinks about this
# sentence" at each layer.

def extract_all_layer_embeddings(sentences, model_name, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()

    all_layer_vectors = None  # will become a list, one array per layer

    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start:start + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                             max_length=64, return_tensors="pt").to(device)
            out = model(**enc)
            # out.hidden_states is a tuple: (embedding_layer, layer1, layer2, ..., layerN)
            hidden_states = out.hidden_states
            mask = enc["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)

            if all_layer_vectors is None:
                all_layer_vectors = [[] for _ in hidden_states]

            for layer_idx, layer_hs in enumerate(hidden_states):
                # mean-pool over real tokens only (mask out padding)
                summed = (layer_hs * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                pooled = (summed / counts).cpu().numpy()
                all_layer_vectors[layer_idx].append(pooled)

            if start % (batch_size * 10) == 0:
                print(f"  {model_name}: {start}/{len(sentences)} sentences done")

    # concatenate batches back together -> one array per layer, shape (n_sentences, hidden_dim)
    all_layer_vectors = [np.concatenate(layer_batches, axis=0) for layer_batches in all_layer_vectors]

    del model
    torch.cuda.empty_cache()
    return all_layer_vectors  # list of length num_layers+1



# Step 2: for each layer, train a simple probe (logistic regression)
# to predict correct/incorrect from that layer's vectors.
# We also run a CONTROL version with shuffled labels -- if the probe
# still does well on shuffled labels, something is wrong with the setup
# (e.g. the split is leaking information), so this is a sanity check,

def train_probes_per_layer(layer_vectors, labels, groups, n_repeats=3):
    We use BALANCED
    # accuracy instead, where chance level is always 0.5 regardless of
    # class imbalance, and we weight the classifier to not just exploit
    # the imbalance.
    results = []
    y = np.array([1 if l == "correct" else 0 for l in labels])

    for layer_idx, X in enumerate(layer_vectors):
        real_accs, control_accs = [], []

        for repeat in range(n_repeats):
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=repeat)
            train_idx, test_idx = next(splitter.split(X, y, groups=groups))

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # real probe
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(X_train, y_train)
            real_acc = balanced_accuracy_score(y_test, clf.predict(X_test))
            real_accs.append(real_acc)

            # control: same split, but shuffled training labels
            y_train_shuffled = np.random.RandomState(repeat).permutation(y_train)
            clf_control = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf_control.fit(X_train, y_train_shuffled)
            control_acc = balanced_accuracy_score(y_test, clf_control.predict(X_test))
            control_accs.append(control_acc)

        results.append({
            "layer": layer_idx,
            "real_balanced_accuracy": np.mean(real_accs),
            "real_balanced_accuracy_std": np.std(real_accs),
            "control_balanced_accuracy": np.mean(control_accs),
            "control_balanced_accuracy_std": np.std(control_accs),
        })

    return pd.DataFrame(results)



# Step 3: run everything, for every model, and plot

def run_full_probing_experiment(pairs_csv="yoruba_tone_pairs.csv"):
    df = pd.read_csv(pairs_csv)
    sentences = df["sentence"].tolist()
    labels = df["label"].tolist()
    groups = df["pair_id"].tolist()  # keep same pair_id together in train OR test, never split

    all_results = {}

    for model_key, model_name in MODELS.items():
        print(f"\n=== {model_key} ({model_name}) ===")
        layer_vectors = extract_all_layer_embeddings(sentences, model_name)
        print(f"  Extracted {len(layer_vectors)} layers, vector shape per layer: {layer_vectors[0].shape}")

        results_df = train_probes_per_layer(layer_vectors, labels, groups)
        results_df["model"] = model_key
        all_results[model_key] = results_df
        print(results_df[["layer", "real_balanced_accuracy", "control_balanced_accuracy"]].to_string(index=False))

    combined = pd.concat(all_results.values(), ignore_index=True)
    combined.to_csv("tone_probing_results.csv", index=False)
    print("\nSaved combined results to tone_probing_results.csv")

    # plot: one line per model for real accuracy, dashed line for control
    plt.figure(figsize=(8, 5))
    colors = plt.cm.tab10.colors
    for i, (model_key, results_df) in enumerate(all_results.items()):
        plt.plot(results_df["layer"], results_df["real_balanced_accuracy"],
                  marker="o", label=f"{model_key} (real)", color=colors[i])
        plt.plot(results_df["layer"], results_df["control_balanced_accuracy"],
                  marker="x", linestyle="--", label=f"{model_key} (control/shuffled)", color=colors[i], alpha=0.5)
    plt.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance level (balanced)")
    plt.xlabel("Layer")
    plt.ylabel("Balanced probe accuracy")
    plt.title("Tone-error detection: balanced accuracy by layer")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("tone_probing_by_layer.png", dpi=150)
    print("Saved plot to tone_probing_by_layer.png")

    return combined


if __name__ == "__main__":
    results = run_full_probing_experiment()

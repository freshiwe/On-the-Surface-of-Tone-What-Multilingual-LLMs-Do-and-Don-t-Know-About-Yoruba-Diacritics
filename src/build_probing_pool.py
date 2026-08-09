

# --- CELL 1: install what's needed ---
# !pip install pandas requests

import re
import unicodedata
import pandas as pd

# --- CELL 2: pull sentences from MasakhaNER (Yoruba split) ---


import requests

MASAKHANER_FILES = [
    "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/data/yor/train.txt",
    "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/data/yor/dev.txt",
    "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/data/yor/test.txt",
]

def parse_conll(text: str):
    """CoNLL format: one 'token TAG' per line, blank line = end of sentence."""
    sentences = []
    current_tokens = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            if current_tokens:
                sentences.append(" ".join(current_tokens))
                current_tokens = []
            continue
        # token and tag are separated by whitespace; token is everything before the last space
        parts = line.rsplit(" ", 1)
        token = parts[0]
        current_tokens.append(token)
    if current_tokens:
        sentences.append(" ".join(current_tokens))
    return sentences

def get_masakhaner_sentences():
    sentences = []
    for url in MASAKHANER_FILES:
        resp = requests.get(url)
        resp.raise_for_status()
        sentences.extend(parse_conll(resp.text))
    return sentences


# --- CELL 3: pull sentences from MENYO-20k (local clone) ---
# Non-religious: news, TED talks, movie/radio transcripts, tech articles.

from pathlib import Path

MENYO_LOCAL_DIR = "./menyo-20k_MT"  # <-- change this to your actual clone path if different

MENYO_FILENAMES = ["train.tsv", "dev.tsv", "test.tsv"]

def get_menyo_sentences(local_dir: str = MENYO_LOCAL_DIR):
    sentences = []
    root = Path(local_dir).resolve()
    print(f"  Looking under: {root}")
    if not root.exists():
        print(f"  [error] this folder doesn't exist — check MENYO_LOCAL_DIR")
        return sentences

    for filename in MENYO_FILENAMES:
        # search recursively for the file by name, rather than assuming it's
        # in a "data" subfolder — this handles zip downloads (folder names
        # like menyo-20k_MT-master), nested clones, etc.
        matches = list(root.rglob(filename))
        if not matches:
            print(f"  [skip] no file named {filename} found anywhere under {root}")
            continue
        path = matches[0]
        if len(matches) > 1:
            print(f"  [note] multiple {filename} files found, using: {path}")
        df = pd.read_csv(path, sep="\t", quoting=3, on_bad_lines="skip")
        col = [c for c in df.columns if "yoruba" in c.lower()][0]
        sentences.extend(df[col].dropna().astype(str).tolist())
    return sentences


# --- CELL 4: helper functions to check for diacritics / clean text ---

# Yoruba tone marks + special letters we expect in fully diacritized text
YORUBA_DIACRITIC_CHARS = set("àáèéìíòóùúẹẹ́ẹ̀ọọ́ọ̀ṣńṇ̀")

def has_diacritics(sentence: str) -> bool:
    """Return True if the sentence contains at least a few Yoruba diacritic marks.
    A sentence with zero diacritics is almost certainly NOT properly toned Yoruba
    (or is a proper noun / number-only line), so we filter those out."""
    normalized = unicodedata.normalize("NFC", sentence)
    count = sum(1 for ch in normalized if ch in YORUBA_DIACRITIC_CHARS)
    return count >= 2  # at least 2 marked characters — tune this threshold if needed

def word_count(sentence: str) -> int:
    return len(sentence.strip().split())

def clean_sentence(sentence: str) -> str:
    sentence = sentence.strip()
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = sentence.replace("\ufeff", "")  # strip stray BOM characters seen in the raw file
    return sentence


# --- CELL 5: build the combined, filtered pool ---

def build_pool(min_words=5, max_words=15):
    print("Downloading MasakhaNER Yoruba...")
    masakhaner_sents = get_masakhaner_sentences()
    print(f"  -> {len(masakhaner_sents)} raw sentences")

    print(f"Reading MENYO-20k from local clone ({MENYO_LOCAL_DIR})...")
    menyo_sents = get_menyo_sentences()
    print(f"  -> {len(menyo_sents)} raw sentences")

    all_sents = [(s, "masakhaner") for s in masakhaner_sents] + \
                [(s, "menyo20k") for s in menyo_sents]

    rows = []
    seen = set()
    for raw, source in all_sents:
        s = clean_sentence(raw)
        if not s or s in seen:
            continue
        wc = word_count(s)
        if wc < min_words or wc > max_words:
            continue
        if not has_diacritics(s):
            continue
        seen.add(s)
        rows.append({"sentence": s, "source": source, "word_count": wc})

    df = pd.DataFrame(rows)
    print(f"\nFinal pool size after filtering: {len(df)} sentences")
    print(df["source"].value_counts())
    return df


if __name__ == "__main__":
    pool = build_pool()
    pool.to_csv("yoruba_probing_pool.csv", index=False)
    print("\nSaved to yoruba_probing_pool.csv")
    print(pool.head(10))

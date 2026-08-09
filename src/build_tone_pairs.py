
# Yoruba Tone-Corruption Script — Week 2, Plan 1 (Step 2)
# Takes clean sentences from yoruba_probing_pool.csv and builds
# correct-vs-incorrect tone pairs for your probing dataset.


import unicodedata
import random
import pandas as pd

random.seed(42)  # so results are reproducible

# Yoruba tone marks (as Unicode COMBINING characters, after NFD decomposition):
#   U+0301 = combining acute accent  -> HIGH tone
#   U+0300 = combining grave accent  -> LOW tone
#   (no mark on a vowel/syllabic nasal = MID tone, the "default")
#
# IMPORTANT THING TO NOTE: the underdot in ẹ, ọ, ṣ (U+0323, combining dot below) is NOT a
# tone mark — it's part of the base letter and must be left alone, or you'll
# accidentally turn "ọ" into "o", which is a spelling change, not a tone change.

HIGH = "\u0301"   # acute
LOW = "\u0300"    # grave
UNDERDOT = "\u0323"  # must never be touched

TONE_MARKS = {HIGH, LOW}


def to_nfd(sentence: str):
    """Decompose into base characters + separate combining marks, as a list."""
    return list(unicodedata.normalize("NFD", sentence))


def to_nfc(chars) -> str:
    """Recompose a list of characters back into normal Yoruba text."""
    return unicodedata.normalize("NFC", "".join(chars))


def find_tone_positions(chars):
    """Return the list indices of combining tone marks (acute/grave only)."""
    return [i for i, ch in enumerate(chars) if ch in TONE_MARKS]



# Corruption type 1: strip ALL tone marks
# Tests: "does the model even need tone information at all"

def strip_all_tones(sentence: str) -> str:
    chars = to_nfd(sentence)
    chars = [ch for ch in chars if ch not in TONE_MARKS]
    return to_nfc(chars)



# Corruption type 2: flip ONE tone mark to the opposite tone
# Tests: "does the model notice one specific, local tone error"

def corrupt_single_tone(sentence: str):
    """Returns (corrupted_sentence, changed) where changed=False if there
    was nothing to corrupt (e.g. a sentence with no tone marks at all)."""
    chars = to_nfd(sentence)
    positions = find_tone_positions(chars)
    if not positions:
        return sentence, False

    pos = random.choice(positions)
    if chars[pos] == HIGH:
        chars[pos] = LOW
    else:
        chars[pos] = HIGH

    return to_nfc(chars), True



# Corruption type 3: delete ONE tone mark (syllable becomes mid-tone)
# A second, milder variant of a local error — useful for more pairs
# per sentence without repeating the same corruption type.

def delete_single_tone(sentence: str):
    chars = to_nfd(sentence)
    positions = find_tone_positions(chars)
    if not positions:
        return sentence, False

    pos = random.choice(positions)
    del chars[pos]
    return to_nfc(chars), True



# Build the full pair dataset

def build_tone_pairs(pool_csv="yoruba_probing_pool.csv", n_sentences=400):
    # Accept either a file path (string) or an already-loaded DataFrame,
    # e.g. if you did `df = pd.read_csv(...)` yourself earlier in your
    # notebook, you can just call build_tone_pairs(pool_csv=df).
    if isinstance(pool_csv, pd.DataFrame):
        df = pool_csv.copy()
    else:
        df = pd.read_csv(pool_csv)

    # keeping sentences that actually have tone marks to corrupt
    df["nfd_check"] = df["sentence"].apply(lambda s: len(find_tone_positions(to_nfd(s))))
    df = df[df["nfd_check"] > 0].copy()

    sample = df.sample(n=min(n_sentences, len(df)), random_state=42).reset_index(drop=True)

    rows = []
    for i, row in sample.iterrows():
        sentence = row["sentence"]
        source = row["source"]
        pair_id = f"tone_{i:04d}"

        # 1. the correct (original) sentence
        rows.append({
            "pair_id": pair_id, "sentence": sentence, "label": "correct",
            "corruption_type": "none", "phenomenon": "tone", "source": source
        })

        # 2. fully stripped version
        stripped = strip_all_tones(sentence)
        rows.append({
            "pair_id": pair_id, "sentence": stripped, "label": "incorrect",
            "corruption_type": "strip_all", "phenomenon": "tone", "source": source
        })

        # 3. single tone flip
        flipped, changed = corrupt_single_tone(sentence)
        if changed:
            rows.append({
                "pair_id": pair_id, "sentence": flipped, "label": "incorrect",
                "corruption_type": "single_flip", "phenomenon": "tone", "source": source
            })

        # 4. single tone deletion
        deleted, changed = delete_single_tone(sentence)
        if changed:
            rows.append({
                "pair_id": pair_id, "sentence": deleted, "label": "incorrect",
                "corruption_type": "single_delete", "phenomenon": "tone", "source": source
            })

    out = pd.DataFrame(rows)
    return out


if __name__ == "__main__":
    pairs = build_tone_pairs(n_sentences=400)
    pairs.to_csv("yoruba_tone_pairs.csv", index=False)
    print(f"Built {len(pairs)} rows from {pairs['pair_id'].nunique()} source sentences")
    print(pairs["corruption_type"].value_counts())
    print()
    print("Example pair:")
    example_id = pairs["pair_id"].iloc[0]
    print(pairs[pairs["pair_id"] == example_id][["label", "corruption_type", "sentence"]].to_string(index=False))

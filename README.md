# On the Surface of Tone: What Multilingual LLMs Do and Don't Know About Yorùbá Diacritics

A probing study testing whether multilingual language models genuinely represent Yorùbá tone, or merely detect the presence of tone-marking diacritics without understanding what they mean.

## What is this project? (plain-language summary)

Yorùbá is a tonal language — the same sequence of letters can mean different things depending on pitch, which is marked in writing using accent marks (tone marks) over vowels. This project asks: when an AI language model reads Yorùbá text, does it actually understand what those tone marks mean, or does it just notice that *some* marks are present without understanding *which* ones?

We tested this by taking real Yorùbá sentences and creating damaged versions of each — some with all tone marks removed, some with just one mark deleted, and some with just one mark swapped for the wrong one (a change that leaves the sentence looking almost identical but makes it linguistically incorrect). We then checked whether two AI models could tell correct sentences apart from each type of damage.

**Finding:** the models easily detect large-scale tone-mark removal, but perform at chance level (no better than random guessing) when just one mark is swapped for the wrong one. This suggests these models detect the *presence* of tone marking as a surface/orthographic cue, rather than genuinely representing *what the tone marks mean*.

## Repository structure

```
yoruba-tone-probing/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── src/                              # all analysis code, run in this order
│   ├── build_probing_pool.py         # Step 1: collect + filter source sentences
│   ├── build_tone_pairs.py           # Step 2: generate correct/incorrect tone pairs
│   ├── week3_probing.py              # Step 3: extract hidden states, train layer-wise probes
│   ├── week3b_breakdown.py           # Step 4: break results down by corruption type
│   └── week3c_symmetry_check.py      # Step 5: check high-to-low vs low-to-high flip symmetry
│
├── data/
│   └── processed/                    # small, git-friendly CSVs — safe to commit
│       ├── yoruba_probing_pool.csv
│       └── yoruba_tone_pairs.csv
│
├── results/
│   ├── tables/
│   │   ├── tone_probing_results.csv
│   │   ├── tone_probing_breakdown_by_corruption_type.csv
│   │   └── tone_flip_symmetry_results.csv
│   └── figures/
│       ├── tone_probing_by_layer.png
│       ├── breakdown_afro-xlmr-base.png
│       └── breakdown_xlm-roberta-base.png
│
└── paper/
    └── yoruba_tone_paper_draft.docx
```

### What's deliberately NOT committed to the repo (see `.gitignore`)

- **`embedding_cache/`** — the cached model hidden-state vectors (`.npz` files). These are large (can be hundreds of MB), and fully reproducible by rerunning `week3_probing.py` / `week3b_breakdown.py`. Committing large binary files to git is bad practice — they bloat the repo permanently, even after deletion.
- **`menyo-20k_MT/`** (your local clone of the MENYO-20k source data) — this is someone else's repository; link to it instead of duplicating it. Anyone reproducing your work should clone it themselves (see Setup below).
- Any raw, unfiltered downloads — only the final processed CSVs need to be committed.

## Setup

```bash
git clone https://github.com/uds-lsv/menyo-20k_MT.git   # needed by build_probing_pool.py
pip install -r requirements.txt
```

## Reproducing the results

Run the scripts in `src/` in order. Each one reads the output of the previous step:

```bash
python src/build_probing_pool.py      # -> data/processed/yoruba_probing_pool.csv
python src/build_tone_pairs.py        # -> data/processed/yoruba_tone_pairs.csv
python src/week3_probing.py           # -> results/tables/, results/figures/ (needs a GPU)
python src/week3b_breakdown.py        # -> results/tables/, results/figures/
python src/week3c_symmetry_check.py   # -> results/tables/
```

Note: `week3_probing.py` and `week3b_breakdown.py` download and run two open multilingual models (`Davlan/afro-xlmr-base`, `xlm-roberta-base`) — a GPU is recommended (Google Colab's free tier is sufficient).

## Summary of results

| Corruption type | Balanced accuracy (both models) | Interpretation |
|---|---|---|
| All tone marks removed | 0.86 – 0.91 | Easily detected |
| One tone mark deleted | 0.55 – 0.61 | Weak but real signal |
| One tone mark flipped (wrong tone) | 0.50 – 0.54 | At chance — undetectable |

Full methodology, discussion, and limitations are in `paper/yoruba_tone_paper_draft.docx`.



# Fake News Detection — ISOT (binary text classification)

**English** | [简体中文](README.zh.md)

Detect whether a news article is **fake** (`label=1`) or **real** (`label=0`) from its
**title + body**, using the ISOT Fake and Real News dataset
([Kaggle: clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)).

This repository is being built stage by stage following
[`docs/coding_plan.md`](docs/coding_plan.md). Current status:

| Stage | Content | Status |
|---|---|---|
| 0 | repo skeleton, config system, tests, CI | ✅ |
| 1 | dataset download + EDA report | ✅ |
| 2 | cleaning, dedup/conflict audit, 70/10/20 split | ✅ |
| 3 | TF-IDF + Logistic Regression / Linear SVM | ✅ |
| 4 | shortcut-feature baseline | ✅ |
| 5 | Bi-LSTM (deferred) | ⏸ |
| 6 | DistilBERT (deferred) | ⏸ |
| 7 | evaluation, bootstrap CI, error analysis | 🟡 |
| 8 | masked source-cue experiment | ✅ |
| 9 | figures + docs | ⬜ |
| 10 | temporal / LIAR / near-dup (deferred) | ⏸ |

## Data

- `True.csv` — Reuters articles, `label = 0` (real)
- `Fake.csv` — articles from flagged sites, `label = 1` (fake)
- License: **CC BY-NC-SA 4.0** (data); code is MIT. The CSVs are **not** in this repo.

## Quickstart

Python **3.10+** (tested with 3.11 on Windows). From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

Neural stages (Bi-LSTM / DistilBERT) additionally need torch + transformers:

```bash
pip install -e ".[dev,neural]"
```

### 1. Get the data

Create a Kaggle API token (Kaggle → *Settings* → *API* → *Create New Token*) and
export it (never commit it):

```powershell
$env:KAGGLE_API_TOKEN = "KGAT_xxx"        # PowerShell
```

```bash
export KAGGLE_API_TOKEN="KGAT_xxx"        # bash
```

```bash
python -m fakenews.data.download --config configs/base.yaml
```

`download.py` validates both CSVs, writes `data/raw/CHECKSUMS.txt`, and prefers
the Kaggle API. If TLS to Google Storage is blocked (common on some networks),
it retries the same signed URL over HTTP and logs a warning. Manual fallback:

```bash
python -m fakenews.data.download --archive path/to/archive.zip
```

### 2. EDA → clean → split

```bash
python -m fakenews.data.eda    --config configs/base.yaml
python -m fakenews.data.clean  --config configs/base.yaml --csv
python -m fakenews.data.split  --config configs/base.yaml
```

Outputs:

```text
results/eda_report.md
results/figures/fig_eda_overview.{png,pdf}
data/processed/articles_clean.parquet   # canonical, used by training
data/processed/articles_clean.csv       # human-readable copy
data/processed/clean_log.json           # cleaning counters
data/processed/{train,val,test}.parquet # 70 / 10 / 20, stratified, seed 42
data/processed/split_manifest.json      # sizes, label counts, sha256
```

The cleaning step, in order: strip HTML/control characters → collapse whitespace
→ drop bodies `<= 50` chars → drop **all** rows of a `dup_key` that occurs with
both labels → drop exact duplicates (`keep="first"`). URLs and `(Reuters)`
datelines are **kept** here; masking them is a separate experiment.

## Current data snapshot

Run on 2026-10-23 with `min_text_len=50`, `seed=42`:

| Quantity | Value |
|---|---|
| raw rows | 44,898 (`True.csv` 21,417 + `Fake.csv` 23,481) |
| empty bodies | 631 |
| dropped: body ≤ 50 chars | 844 |
| dropped: cross-label conflicts | 0 |
| dropped: exact duplicates | 5,551 |
| **cleaned rows** | **38,503** (21,194 real / 17,309 fake) |
| split train / val / test | 26,951 / 3,851 / 7,701 |

`subject` is perfectly disjoint between the classes (real: `politicsNews`,
`worldnews`; fake: the other six values) and dates differ, so both are excluded
from every feature matrix. The cleaned CSV is ~293 MB because it keeps
`text_raw`, `text_clean` and `dup_key`; use the Parquet file for training.

## Classical baselines (stage 3)

`python -m fakenews.train.classical --config configs/base.yaml --model {lr,svm} --final`

One seed (42), TF-IDF fit on train only, `alpha` and thresholds selected on val,
test read once after re-fitting on train+val. F1 (positive class = fake):

| model | title_only | body_only | concat | fusion | alpha |
|---|---|---|---|---|---|
| LR val | 0.9387 | 0.9916 | 0.9939 | **0.9951** | 0.40 |
| LR test | 0.9429 | 0.9929 | 0.9962 | **0.9962** | 0.40 |
| SVM val | 0.9396 | 0.9939 | 0.9959 | **0.9960** | 0.50 |
| SVM test | 0.9425 | 0.9957 | 0.9975 | **0.9970** | 0.50 |

Main table uses threshold 0.5. The per-condition val-tuned thresholds are a
sensitivity check: they change test F1 by at most 0.0012 (e.g. SVM concat
0.9963 tuned vs 0.9975 at 0.5; LR body 0.9940 tuned vs 0.9929 at 0.5).
`alpha` is a probability-mix weight, not branch importance: the title branch is
less calibrated, and logit averaging selects 0.30/0.45 with essentially the same
test F1 (0.9961/0.9975).

Per-sample probabilities live in `results/predictions/` (git-ignored); the
unified @0.5 table for every model/condition/split is `results/all_metrics.csv`.
Title-only is clearly weaker than body-only, but fusing both still adds a small
amount.

Stage 4 / 8 reproducibility:

```bash
python -m fakenews.train.shortcut --config configs/shortcut.yaml --final
python -m fakenews.experiments.source_cue --config configs/base.yaml --model lr
python -m fakenews.experiments.source_cue --config configs/base.yaml --model svm
python -m fakenews.eval.collect_metrics --config configs/base.yaml
```

Full results, statistical checks and the interpretation caveats are kept in
[`results/RESULTS.md`](results/RESULTS.md). The short version: body dominates,
fusion/early-concat are statistically indistinguishable, and a one-line
`(Reuters)` rule already reaches F1 0.994 — the benchmark is source-cue
saturated, so these numbers are not evidence of real-world fact-checking.

## Why the benchmark is saturated: five raw examples

The first 100 rows of each raw file already show the shortcut. `True.csv`:
91/100 bodies contain `(Reuters)`, 0 contain a URL, 0 contain `getty`.
`Fake.csv`: 0 contain `(Reuters)`, 23 contain a URL, 79 contain `getty`,
58 contain `featured image`.

| # | label | source | excerpt | giveaway |
|---|---|---|---|---|
| 1 | real | `True.csv` #0 | `WASHINGTON (Reuters) - The head of a conservative Republican faction...` | city + `(Reuters)` dateline |
| 2 | real | `True.csv` #4 | `SEATTLE/WASHINGTON (Reuters) - President Donald Trump called on...` | multi-city dateline |
| 3 | fake | `Fake.csv` #0 | title: `Donald Trump Sends Out Embarrassing New Year’s Eve Message; This is Disturbing`; body ends `Photo by Andrew Burton/Getty Images.` | clickbait title + Getty credit |
| 4 | fake | `Fake.csv` #5 | title: `...Black Boy While He Is In Handcuffs (GRAPHIC IMAGES)`; body ends `Featured image via David McNew/Stringer/Getty Images` | `Featured image` + Getty |
| 5 | fake | `Fake.csv` #3 | title: `...Obama’s Name Coded Into His Website (IMAGES)`; body has `https://t.co/Fg7VacxRtJ pic.twitter.com/5gEMcjQTbH` | `t.co` / Twitter embed |

The diagnostic result is blunt: a single `(Reuters)` rule reaches test F1
0.9943, the 18-format-feature baseline reaches 0.9953, and the full TF-IDF
models reach 0.9962 (LR) / 0.9970 (SVM). Paired bootstrap says the 18-feature
baseline is statistically indistinguishable from both full-text models
(p=0.43 vs LR, p=0.13 vs SVM). Removing the dateline, URLs and media credits
lowers fusion F1 only to 0.9903–0.9944. The dataset is separable almost entirely
by source formatting, not by whether the article is true.

## Configuration

`configs/base.yaml` is the single source of truth; model configs only contain
overrides and are merged recursively at load time. Every CLI accepts:

```bash
--config configs/base.yaml --override data.min_text_len=80
```

## Tests and lint

```bash
pytest -q
ruff check .
```

CI (`.github/workflows/ci.yml`) runs `ruff` and `pytest -m "not slow"` on Ubuntu.

## Hardware

- TF-IDF + LR/SVM: laptop CPU, minutes.
- Shortcut / masking experiments: laptop CPU, seconds to a few minutes.
- Bi-LSTM: CPU/GPU, 20–60 min per epoch.
- DistilBERT: Colab T4, 20–40 min per seed.

## Known limitations

- Labels come from the **source**, not from per-article fact checking; the model
  can learn source style rather than truthfulness.
- `subject` and `date` are audit columns only and must never be features.
- Single dataset, English only, body truncated for neural models.

## License

Code: MIT (see `LICENSE`). Data: CC BY-NC-SA 4.0 — see `data/README.md` and
`docs/data_card.md`.

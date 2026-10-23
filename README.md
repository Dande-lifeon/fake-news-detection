# Fake News Detection — ISOT (binary text classification)

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
| 3 | TF-IDF + Logistic Regression / Linear SVM | ⬜ |
| 4 | shortcut-feature baseline | ⬜ |
| 5 | Bi-LSTM | ⬜ |
| 6 | DistilBERT | ⬜ |
| 7 | evaluation, bootstrap CI, error analysis | ⬜ |
| 8 | masked source-cue experiment | ⬜ |
| 9 | figures + docs | ⬜ |

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

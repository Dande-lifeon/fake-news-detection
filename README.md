# Fake news detection on ISOT

[简体中文](README.zh.md)

This repository trains and compares four models on the ISOT Fake and Real News
dataset: TF-IDF with logistic regression, TF-IDF with a linear SVM, a Bi-LSTM,
and a fine-tuned DistilBERT. Title and body are encoded separately and fused,
and the code keeps a concatenation baseline for comparison.

The top F1 is not the interesting result. A one-line rule based on the Reuters
dateline already scores 0.994, and an 18-feature format-only model scores 0.995.
The full TF-IDF models reach 0.996. ISOT is separable by source and layout cues,
so a high score here is not evidence of fact-checking.

Dataset: <https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset>

## Status

| Stage | Content | State |
|---|---|---|
| 0 | package skeleton, config, tests, CI | ✅ done |
| 1 | data download and EDA | ✅ done |
| 2 | cleaning, deduplication, 70/10/20 split | ✅ done |
| 3 | TF-IDF LR and linear SVM | ✅ done |
| 4 | format-feature shortcut baseline | ✅ done |
| 5 | Bi-LSTM | 🟡 code ready, not trained |
| 6 | DistilBERT | 🟡 code ready, not trained |
| 7 | bootstrap, McNemar, calibration, error analysis | 🟡 partial |
| 8 | source-cue masking | ✅ done |
| 9 | result figures and docs | 🟡 in progress |
| 10 | temporal split, LIAR, near-duplicate split | ⏸ deferred |

The runnable path today is stages 0-4 and 8, plus the metrics collector and
result figures. The neural stages have code but no trained results yet: torch
and transformers are not installed and PyPI is unreachable from this machine.

## Repository layout

```text
configs/            YAML defaults and per-model overrides
data/               raw and processed data (git-ignored)
docs/               data card, protocol, coding plan, environment record
environment.yml     base conda environment
.python-version     reference Python version
fakenews/
  data/             download, EDA, clean, split
  features/         TF-IDF, shortcut features, masking
  models/           classical model, fusion layers
  train/            classical and shortcut training entry points
  eval/             metrics, bootstrap, metrics collector
  experiments/      source-cue masking
scripts/            run_all.sh, smoke.sh, env_report.py
tests/              unit and integration tests
results/            tracked reports and metrics; predictions are git-ignored
```

## Setup

Python 3.11 is the reference version. The lockfile in `requirements.txt` was
generated on 3.11 and the project now requires 3.11 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

To use the pinned set explicitly:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

Conda users can start from the base environment:

```bash
conda env create -f environment.yml
conda activate fakenews
pip install -e ".[dev]"
```

The exact versions used for the reported experiments are in
[`docs/environment.md`](docs/environment.md). Regenerate that file with:

```bash
python scripts/env_report.py --output docs/environment.md
```

`Makefile` targets are optional. On Windows without `make`, run the
`python -m ...` commands directly.

Neural stages need extra packages:

```bash
pip install -e ".[dev,neural]"
```

## Data

The two CSVs are not stored in the repository.

- `True.csv`: Reuters articles, `label=0`
- `Fake.csv`: articles from flagged sites, `label=1`

Download with a Kaggle API token:

```powershell
$env:KAGGLE_API_TOKEN = "KGAT_xxx"
```

```bash
export KAGGLE_API_TOKEN="KGAT_xxx"
python -m fakenews.data.download --config configs/base.yaml
```

If the download is blocked, fetch `archive.zip` from Kaggle and unpack it:

```bash
python -m fakenews.data.download --archive path/to/archive.zip
```

The loader checks both files, records their SHA-256 in `data/raw/CHECKSUMS.txt`,
and writes `data/raw/raw_summary.json`.

## Run the pipeline

### 1. EDA, cleaning, and split

```bash
python -m fakenews.data.eda   --config configs/base.yaml
python -m fakenews.data.clean --config configs/base.yaml --csv
python -m fakenews.data.split --config configs/base.yaml
```

Outputs:

```text
results/eda_report.md
results/figures/fig_eda_overview.{png,pdf}
data/processed/articles_clean.parquet
data/processed/articles_clean.csv
data/processed/clean_log.json
data/processed/{train,val,test}.parquet
data/processed/split_manifest.json
```

Cleaning runs in this order: strip HTML and control characters, collapse
whitespace, drop bodies of 50 characters or fewer, drop every row whose
`dup_key` appears with both labels, then keep the first exact duplicate. URLs and
Reuters datelines are kept here; removing them is a separate experiment.

The current snapshot, built with `min_text_len=50` and `seed=42`:

| Quantity | Value |
|---|---:|
| raw rows | 44,898 (`True.csv` 21,417, `Fake.csv` 23,481) |
| empty bodies | 631 |
| dropped: body <= 50 chars | 844 |
| dropped: cross-label conflicts | 0 |
| dropped: exact duplicates | 5,551 |
| cleaned rows | 38,503 (21,194 real, 17,309 fake) |
| train / val / test | 26,951 / 3,851 / 7,701 |

`subject` is disjoint between the classes and `date` differs, so neither is used
as a feature. The CSV copy is about 293 MB because it keeps `text_raw`,
`text_clean`, and `dup_key`; training reads the Parquet file.

The 631 empty bodies are a subset of the 844 bodies dropped for being 50
characters or shorter, so they are not subtracted twice:
44,898 - 844 - 5,551 = 38,503.

### 2. Classical baselines

```bash
python -m fakenews.train.classical --config configs/base.yaml --model lr  --final
python -m fakenews.train.classical --config configs/base.yaml --model svm --final
```

Both models see the same split. TF-IDF is fit on train only; `C` is chosen by
5-fold CV on train; `alpha` and the decision threshold are chosen on val; the
model is refit on train+val before the single test evaluation.

Test F1 for the fake class:

| Model | title_only | body_only | concat | fusion | alpha |
|---|---:|---:|---:|---:|---:|
| LR | 0.9429 | 0.9929 | 0.9962 | 0.9962 | 0.40 |
| SVM | 0.9425 | 0.9957 | 0.9975 | 0.9970 | 0.50 |

The main table uses threshold 0.5. Val-tuned thresholds change test F1 by at
most 0.0012, so 0.5 is the safer default. `alpha` is a probability-mix weight,
not a measure of branch importance.

Per-sample probabilities are written to `results/predictions/` (git-ignored).
The consolidated table is `results/all_metrics.csv`. Per-condition 95% CIs are
in `results/cis.csv`, the four result figures are under `results/figures/`, and
the FP/FN table is `results/error_examples.md`.

### 3. Shortcut and masking experiments

```bash
python -m fakenews.train.shortcut --config configs/shortcut.yaml --final
python -m fakenews.experiments.source_cue --config configs/base.yaml --model lr
python -m fakenews.experiments.source_cue --config configs/base.yaml --model svm
python -m fakenews.eval.evaluate --config configs/base.yaml
```

`shortcut` trains logistic regression on 18 surface features: dateline, URL,
Getty and "featured image" credits, casing, punctuation counts, and lengths. It
never reads the article text.

`source_cue` masks the cleaned body, retrains the same LR/SVM configuration on
the masked train split, and compares the masked test predictions with the
original ones. `C`, `alpha`, and the threshold are fixed, so the only change is
the removed cue.

## Results

The full result ledger, confidence intervals, McNemar tests, calibration, and
error analysis are in [`results/RESULTS.md`](results/RESULTS.md). The Week 9
summary used for the 1-page report and slides is
[`results/MILESTONE.md`](results/MILESTONE.md).

The short version:

- Body is much stronger than title. Title-only F1 is about 0.94; body-only is
  about 0.99.
- Fusion and early concatenation are statistically indistinguishable.
- A single `(Reuters)` rule reaches test F1 0.9943.
- The 18-feature format model reaches 0.9953.
- Full-text LR and SVM reach 0.9962 and 0.9970.
- Masking datelines, URLs, and media credits lowers fusion F1 only to
  0.9903-0.9944.

The masking result is the main finding. Removing the obvious cue does not remove
the shortcut, because the corpus has several redundant source and topic signals.

### Raw examples

The first 100 rows already show the pattern. In `True.csv`, 91 of 100 bodies
contain `(Reuters)`, none contain a URL, and none contain `getty`. In `Fake.csv`,
none contain `(Reuters)`, 23 contain a URL, 79 contain `getty`, and 58 contain
`featured image`.

| Label | Source | Excerpt | Cue |
|---|---|---|---|
| real | `True.csv` #0 | `WASHINGTON (Reuters) - The head of a conservative Republican faction...` | city plus Reuters dateline |
| real | `True.csv` #4 | `SEATTLE/WASHINGTON (Reuters) - President Donald Trump called on...` | multi-city dateline |
| fake | `Fake.csv` #0 | title ends with `This is Disturbing`; body ends `Photo by Andrew Burton/Getty Images.` | clickbait title, Getty credit |
| fake | `Fake.csv` #5 | body ends `Featured image via David McNew/Stringer/Getty Images` | featured image, Getty credit |

## Reproducibility

- Every entry point calls `seed_everything(42)`.
- The split is written once and the test file is made read-only.
- `split_manifest.json` stores sizes, class counts, SHA-256 hashes, and the
  pandas version.
- Each run writes `config.yaml`, `environment.txt`, `git_commit.txt`, and
  `train.log` under `results/logs/{run_id}/`.
- `eval.collect_metrics` reports every prediction file it finds. Keep one test
  run per `(model, condition, seed)`; otherwise the ledger will contain duplicate
  test rows.

The protocol is documented in [`docs/protocol.md`](docs/protocol.md).

## Configuration

`configs/base.yaml` holds the defaults. Model configs contain only overrides and
are merged at load time. Every CLI accepts:

```bash
--config configs/base.yaml --override data.min_text_len=80
```

## Tests and CI

```bash
pytest -q
ruff check .
```

CI runs `ruff` and `pytest -m "not slow"` on Ubuntu with Python 3.11. Unit tests
cover cleaning, splitting, TF-IDF, fusion, metrics, masking, and the Kaggle CLI
fallback; they do not need the ISOT CSVs.

## Hardware

The classical pipeline runs on a laptop CPU in minutes. The masking and shortcut
experiments take seconds to a few minutes. Bi-LSTM and DistilBERT have code but
no results yet; training needs torch and transformers, which are not installed
here. The plan targets a laptop CPU for Bi-LSTM and a free Colab T4 for
DistilBERT.

## Limitations

- Labels come from the publishing source, not from per-article fact checking.
  The model can learn source style instead of truthfulness.
- `subject` and `date` are audit columns and must not be features.
- One dataset, English only; the neural models truncate long bodies.
- The dataset is close to saturated, so small F1 differences between models
  should not be read as meaningful without a paired test.

## License

Code is MIT (see `LICENSE`). The data is CC BY-NC-SA 4.0; see `data/README.md`
and [`docs/data_card.md`](docs/data_card.md).

## Troubleshooting

- `download` fails: check `KAGGLE_API_TOKEN`, or use `--archive` / copy the CSVs
  into `data/raw/`.
- `split` refuses to run: it protects an existing split. Pass `--force` only if
  you intend to regenerate train/val/test.
- `make` is missing on Windows: run the `python -m ...` commands above.
- `source_cue` cannot find predictions: run the classical baselines first.

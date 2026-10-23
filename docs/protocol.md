# Evaluation protocol

## Splits

- `train : val : test = 70 : 10 : 20`, stratified on `label`, `seed = 42`.
- Generated once by `python -m fakenews.data.split`; `test.parquet` is written
  read-only.
- `id` is the join key across all files. Never reorder rows without keeping `id`.
- Invariants enforced by `assert_split_invariants`: no `id` overlap, complete
  union, class ratio within ±0.5 % of the full data, no empty `text_clean`.

## Test-set discipline

- Models are fit on `train`; hyper-parameters, fusion weight `α` and the decision
  threshold are selected on `val` only.
- `test` is read **once** per `(model, condition, seed)`, only after `--final`,
  for the final model refit on `train + val`.
- Pre-registered analyses (`source_cue`, `temporal`, `cross_dataset`) may evaluate
  on test once, but their results may never feed back into tuning.
- `evaluate.py` refuses to append a second test row for the same
  `(model, condition, seed)`.

## Leakage guards

- `subject` and `date` never enter a feature matrix (audit only).
- TF-IDF vectorisers are fit on `train` only (`fit_transform`); `val`/`test`
  only call `transform`.
- Deduplication happens **before** splitting.
- Source-cue features are extracted from `text_raw` only.

## Metrics

Positive class = `label=1` (fake). Report precision, recall, F1, accuracy,
ROC-AUC, PR-AUC, confusion counts, and bootstrap 95 % CIs (1000 resamples,
stratified, `seed=42`). Every final claim needs a CI or a paired test
(McNemar, paired bootstrap).

## Reproducibility

- `seed_everything(42)` in every entry point.
- Each run stores `results/logs/{run_id}/config.yaml`, `environment.txt`,
  `git_commit.txt`, `train.log`.
- `data/raw/CHECKSUMS.txt` and `data/processed/split_manifest.json` pin the exact
  data used.

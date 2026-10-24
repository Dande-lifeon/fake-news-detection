# Evaluation protocol

## Splits

`train : val : test = 70 : 10 : 20`, stratified on `label`, `seed = 42`.

The split is generated once by `python -m fakenews.data.split`, and
`test.parquet` is written read-only. `id` is the join key across all files; do not
reorder rows without keeping it. `assert_split_invariants` checks that ids do not
overlap, that the union covers every row, that class rates stay within 0.5
percentage points (or two rows on a tiny fixture), and that no body is empty.

## Test-set discipline

Models are fit on `train`. Hyper-parameters, the fusion weight `alpha`, and the
decision threshold are selected on `val` only. `test` is read once per
`(model, condition, seed)`, after `--final`, when the model has been refit on
`train + val`.

Pre-registered analyses (`source_cue`, `temporal`, `cross_dataset`) may evaluate
on test once, but their results must not feed back into tuning.

The project keeps one test run per `(model, condition, seed)`.
`eval.collect_metrics` reports every prediction file it finds, so delete stale
prediction files before rebuilding the ledger.

## Leakage guards

- `subject` and `date` never enter a feature matrix. They are audit columns.
- TF-IDF vectorisers are fit on `train` only. `val` and `test` call `transform`.
- Deduplication happens before splitting.
- Shortcut features are extracted from `text_raw`. The masking experiment
  operates on `text_clean`, which is what the baseline models were trained on.

## Metrics

The positive class is `label=1` (fake). Report precision, recall, F1, accuracy,
ROC-AUC, PR-AUC, confusion counts, and bootstrap 95% CIs (1,000 stratified
resamples, `seed=42`). Every final claim needs a CI or a paired test (McNemar or
paired bootstrap).

## Reproducibility

- Every entry point calls `seed_everything`.
- Each run stores `results/logs/{run_id}/config.yaml`, `environment.txt`,
  `git_commit.txt`, and `train.log`.
- `data/raw/CHECKSUMS.txt` and `data/processed/split_manifest.json` pin the data
  and split used.
- The reference package versions are in `docs/environment.md`.

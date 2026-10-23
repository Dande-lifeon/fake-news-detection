# Data

## Source

ISOT Fake and Real News Dataset (University of Victoria, ISOT Lab), distributed on Kaggle:

<https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset>

- `True.csv` — Reuters articles, `label = 0` (real)
- `Fake.csv` — articles from flagged sites, `label = 1` (fake)
- License: **CC BY-NC-SA 4.0** (data only; code is MIT). Do not redistribute the
  CSVs in this repository; download them yourself.

## How to get it

### Option A — Kaggle API token (recommended)

1. Sign in to Kaggle → *Settings* → *API* → *Create New Token*.
2. Put the token in the environment (never commit it):

   ```powershell
   # PowerShell
   $env:KAGGLE_API_TOKEN = "KGAT_xxx"
   ```

   ```bash
   # bash
   export KAGGLE_API_TOKEN="KGAT_xxx"
   ```

   The classic `~/.kaggle/kaggle.json` (`username` + `key`) and the `kaggle`
   CLI are also supported.

3. Run:

   ```bash
   python -m fakenews.data.download --config configs/base.yaml
   ```

### Option B — manual download

Download `archive.zip` from the Kaggle page, unzip it, and copy `True.csv` and
`Fake.csv` into `data/raw/`. Then run the download command again: it validates
the files, checks the column set and writes `data/raw/CHECKSUMS.txt`.

## Layout

```text
data/raw/True.csv            # read-only input
data/raw/Fake.csv            # read-only input
data/raw/CHECKSUMS.txt       # sha256 of both files
data/processed/articles_clean.parquet
data/processed/articles_clean.csv
data/processed/{train,val,test}.parquet
data/processed/split_manifest.json
data/processed/clean_log.json
```

Files under `data/` are git-ignored on purpose.

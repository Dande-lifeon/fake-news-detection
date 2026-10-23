# Data card — ISOT Fake and Real News

## Identity

- **Name**: ISOT Fake and Real News Dataset
- **Authors / institution**: ISOT Lab, University of Victoria
- **Kaggle**: <https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset>
- **License**: CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike)
- **Language**: English
- **Task**: binary text classification, `label=1` fake / `label=0` real

## Files

| File | Label | Rows | Description |
|---|---|---|---|
| `True.csv` | 0 | 21,417 | Reuters articles (2016–2017) |
| `Fake.csv` | 1 | 23,481 | articles from sites flagged as unreliable |

Both have the columns `title, text, subject, date`. Row counts are read from
`pandas.read_csv` and recorded in `data/processed/clean_log.json`; do not
hard-code them elsewhere.

## Construction

Real news was scraped from Reuters.com; fake news was collected from a set of
websites flagged by fact-checking organisations (e.g. PolitiFact, Wikipedia
lists). The label is therefore a **source label**, not a per-article judgment.

## Known issues

1. **Source shortcut**: `subject` is nearly disjoint between the classes and
   `(Reuters)` datelines appear only in real articles — both are label leaks.
2. **Duplicates**: many exact duplicate bodies, plus copies of the same story
   with different titles.
3. **Cross-label conflicts**: the same text occasionally appears as both real
   and fake; these are removed during cleaning.
4. **Temporal drift**: real articles are mostly 2016–2017; fake articles cover a
   wider range.
5. **Truncation**: DistilBERT sees only head+tail tokens of a long article.

## Intended use / non-use

Intended for NLP research and teaching. **Not** a fact-checking tool: a high F1
on this dataset says little about detecting novel misinformation, because the
task can be solved by source-style cues.

## Privacy / ethics

The dataset contains public news text. No personal data was added by this
project. Do not redistribute the CSVs; download them from Kaggle.

## Cleaning policy (this repo)

See `fakenews/data/clean.py`. In short: HTML/control chars removed, whitespace
collapsed, bodies `<= 50` chars dropped, cross-label conflict groups dropped,
exact duplicates dropped (`keep="first"`). Raw `text` is preserved verbatim in
`text_raw`; only `text_clean` is normalised.

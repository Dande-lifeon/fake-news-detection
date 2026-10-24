# Data card: ISOT Fake and Real News

## Identity

- Name: ISOT Fake and Real News Dataset
- Authors: ISOT Lab, University of Victoria
- Kaggle: <https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset>
- License: CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike)
- Language: English
- Task: binary text classification, `label=1` fake and `label=0` real

## Files

| File | Label | Rows | Description |
|---|---|---|---|
| `True.csv` | 0 | 21,417 | Reuters articles, mostly 2016-2017 |
| `Fake.csv` | 1 | 23,481 | articles from sites flagged as unreliable |

Both files have the columns `title, text, subject, date`. The row counts come from
`pandas.read_csv` and are recorded in `data/processed/clean_log.json`; do not
hard-code them elsewhere.

## Construction

Real news was scraped from Reuters.com. Fake news was collected from websites
flagged by fact-checking organisations such as PolitiFact and Wikipedia. The
label is a source label, not a per-article judgment.

## Known issues

1. Source shortcut: `subject` is disjoint between the classes, and `(Reuters)`
   datelines appear almost only in real articles. Both leak the label.
2. Duplicates: many exact duplicate bodies, plus copies of the same story with
   different titles.
3. Cross-label conflicts: the same text occasionally appears with both labels.
   These rows are removed during cleaning.
4. Temporal drift: real articles are mostly 2016-2017; fake articles cover a
   wider range.
5. Truncation: DistilBERT sees only the head and tail tokens of a long article.

## Intended use

For NLP research and teaching. It is not a fact-checking tool. A high F1 on this
dataset says little about detecting new misinformation, because the task can be
solved with source-style cues.

## Privacy and ethics

The dataset contains public news text. This project added no personal data. Do
not redistribute the CSVs; download them from Kaggle.

## Cleaning policy

See `fakenews/data/clean.py`. HTML and control characters are removed, whitespace
is collapsed, bodies of 50 characters or fewer are dropped, cross-label conflict
groups are dropped, and exact duplicates keep the first row. `text_raw` preserves
the original body; `text_clean` is the normalised version used by models.

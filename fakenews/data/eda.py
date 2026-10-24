"""Exploratory data analysis for the raw ISOT data.

Produces ``results/eda_report.md`` plus a couple of PNG figures. The report is
meant to answer four questions before modelling:

* how many rows / how much text is missing or very short;
* is ``subject`` a label leak (do real and fake subjects overlap?);
* do the two sources cover different time ranges;
* how many exact duplicates and cross-label conflicts exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fakenews.config import load_config
from fakenews.data.clean import _normalize_key, clean_text
from fakenews.utils import ensure_dir, get_logger


def _markdown_table(frame: pd.DataFrame, index_name: str | None = None) -> str:
    """Render a small DataFrame as a GitHub-flavoured Markdown table."""
    label = index_name or frame.index.name or "index"
    header = [str(label), *[str(c) for c in frame.columns]]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for idx, row in frame.iterrows():
        cells = [str(idx)]
        cells.extend(str(value) for value in row.to_numpy())
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _length_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, name in ((0, "real"), (1, "fake")):
        subset = frame[frame["label"] == label]
        for column in ("title", "text"):
            lengths = subset[column].fillna("").map(clean_text).str.len()
            rows.append(
                {
                    "label": name,
                    "field": column,
                    "n": int(len(lengths)),
                    "mean": round(float(lengths.mean()), 1),
                    "median": round(float(lengths.median()), 1),
                    "p95": round(float(lengths.quantile(0.95)), 1),
                    "empty": int((lengths == 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def run_eda(df: pd.DataFrame, out_dir: Path, figures_dir: Path | None = None) -> dict:
    """Write the EDA report and figures; return a small summary dict."""
    logger = get_logger("fakenews.eda")
    out_dir = ensure_dir(out_dir)
    figures_dir = ensure_dir(figures_dir or out_dir / "figures")

    frame = df.copy()
    for column in ("title", "text", "subject", "date"):
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)

    report: list[str] = ["# EDA: ISOT Fake and Real News", ""]

    report += [
        "## 1. Overview",
        "",
        f"- rows: **{len(frame):,}**",
        f"- columns: {', '.join(map(str, frame.columns))}",
        f"- real (label=0): **{(frame['label'] == 0).sum():,}**",
        f"- fake (label=1): **{(frame['label'] == 1).sum():,}**",
        "",
        "### Missing values",
        "",
        _markdown_table(
            frame[["title", "text", "subject", "date"]]
            .isna()
            .sum()
            .rename_axis("column")
            .to_frame("missing")
        ),
        "",
    ]

    empty_text = int((frame["text"].str.strip() == "").sum())
    report += [f"- empty body: **{empty_text:,}**", ""]

    report += [
        "## 2. Subject × label (label-leak check)",
        "",
        _markdown_table(
            pd.crosstab(
                frame["subject"], frame["label"], margins=True, margins_name="total"
            ).rename(columns={0: "real", 1: "fake"}),
            index_name="subject",
        ),
        "",
    ]
    real_subjects = set(frame.loc[frame["label"] == 0, "subject"])
    fake_subjects = set(frame.loc[frame["label"] == 1, "subject"])
    report += [
        f"- subjects only in real: {sorted(real_subjects) or 'none'}",
        f"- subjects only in fake: {sorted(fake_subjects) or 'none'}",
        f"- shared subjects: {sorted(real_subjects & fake_subjects) or 'none'}",
        "- `subject` is audit-only and must never enter a model feature.",
        "",
    ]

    report += ["## 3. Dates", ""]
    parsed = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
    valid = parsed.notna()
    if valid.any():
        report += [
            f"- parseable dates: {int(valid.sum()):,} / {len(frame):,}",
            f"- range: {parsed[valid].min():%Y-%m-%d} → {parsed[valid].max():%Y-%m-%d}",
            "",
        ]
        months = parsed[valid].dt.to_period("M").astype(str)
        labels = frame.loc[valid, "label"].to_numpy()
        by_month = pd.crosstab(months, labels).rename(columns={0: "real", 1: "fake"})
        by_month.index.name = "month"
        report += [_markdown_table(by_month.tail(12)), ""]
    else:
        report += ["- no parseable dates", ""]

    report += [
        "## 4. Text lengths (characters, after cleaning)",
        "",
        _markdown_table(_length_table(frame)),
        "",
    ]

    dup_keys = (
        frame["title"].map(_normalize_key) + "\n" + frame["text"].map(clean_text).map(_normalize_key)
    )
    duplicated_rows = int(dup_keys.duplicated().sum())
    conflicts = frame.assign(dup_key=dup_keys).groupby("dup_key")["label"].nunique()
    conflict_keys = int((conflicts > 1).sum())
    report += [
        "## 5. Duplicates",
        "",
        f"- exact duplicate rows (title+body): **{duplicated_rows:,}**",
        f"- cross-label conflict keys: **{conflict_keys:,}**",
        "",
    ]

    unbalanced = abs((frame["label"] == 1).mean() - 0.5)
    report += [
        "## 6. Class balance",
        "",
        f"- fake share: **{(frame['label'] == 1).mean():.4f}** "
        f"(distance from 0.5: {unbalanced:.4f})",
        "",
    ]

    report_path = out_dir / "eda_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    logger.info("wrote %s", report_path)

    try:
        _plot_distributions(frame, figures_dir)
    except Exception as exc:  # noqa: BLE001 - figures must not break the report
        logger.warning("figure generation skipped: %s", exc)

    return {
        "rows": int(len(frame)),
        "real": int((frame["label"] == 0).sum()),
        "fake": int((frame["label"] == 1).sum()),
        "empty_text": empty_text,
        "duplicate_rows": duplicated_rows,
        "conflict_keys": conflict_keys,
        "report": str(report_path),
    }


def _plot_distributions(frame: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
    counts = frame["label"].value_counts().sort_index()
    axes[0].bar(["real (0)", "fake (1)"], [counts.get(0, 0), counts.get(1, 0)], color="#4c72b0")
    axes[0].set_title("Class distribution")
    axes[0].set_ylabel("articles")

    for label, name, color in ((0, "real", "#4c72b0"), (1, "fake", "#c44e52")):
        lengths = frame.loc[frame["label"] == label, "text"].map(clean_text).str.len()
        axes[1].hist(
            lengths.clip(upper=lengths.quantile(0.99)),
            bins=50,
            alpha=0.6,
            label=name,
            color=color,
        )
    axes[1].set_title("Body length (clipped at p99)")
    axes[1].set_xlabel("characters")
    axes[1].legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures_dir / f"fig_eda_overview.{suffix}")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EDA report for the raw ISOT data")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    from fakenews.data.clean import load_raw

    cfg = load_config(args.config)
    raw_dir = args.raw_dir or Path(cfg.paths.raw_dir)
    out_dir = args.out_dir or Path(cfg.paths.results_dir)
    summary = run_eda(load_raw(raw_dir), out_dir, out_dir / "figures")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

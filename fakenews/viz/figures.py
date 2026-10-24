"""Generate the figures used in the milestone slides.

Usage:
    python -m fakenews.viz.figures --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fakenews.config import load_config  # noqa: E402
from fakenews.utils import ensure_dir, get_logger  # noqa: E402

COLORS = {
    "title": "#8da0cb",
    "body": "#66c2a5",
    "concat": "#fc8d62",
    "fusion": "#e78ac3",
    "lr": "#4c72b0",
    "svm": "#dd8452",
    "masked": "#c44e52",
    "reference": "#55a868",
}


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _metrics(results: Path) -> pd.DataFrame:
    table = pd.read_csv(results / "all_metrics.csv")
    return table[table["split"] == "test"]


def plot_four_way(metrics: pd.DataFrame, out: Path) -> None:
    conditions = ["title_only", "body_only", "concat", "fusion"]
    labels = ["title", "body", "concat", "fusion"]
    width = 0.18
    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for offset, model in zip((-0.5, 0.5), ("lr", "svm"), strict=False):
        values = []
        for condition in conditions:
            row = metrics[
                (metrics.model == model)
                & (metrics.base_condition == condition)
                & (metrics["mask"] == "none")
            ]
            values.append(float(row["f1"].iloc[0]))
        ax.bar(x + offset * width, values, width=width, label=model.upper(),
               color=COLORS[model], edgecolor="white")
        for xi, value in zip(x + offset * width, values, strict=False):
            ax.text(xi, value + 0.0008, f"{value:.4f}", ha="center", va="bottom",
                    rotation=90, fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("test F1 (fake class)")
    ax.set_ylim(0.92, 1.01)
    ax.set_title("Title, body, concat, and late fusion")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out / "fig_four_way_f1")


def plot_probes(metrics: pd.DataFrame, shortcut: dict, out: Path) -> None:
    rows = [
        ("single rule\n(no Reuters)", shortcut["rule_baselines"]["no_reuters"]["f1"]),
        ("18 format features", shortcut["test"]["f1_at_0.5"]["f1"]),
    ]
    for model in ("lr", "svm"):
        row = metrics[
            (metrics.model == model)
            & (metrics.base_condition == "fusion")
            & (metrics["mask"] == "none")
        ]
        rows.append((f"{model.upper()} fusion", float(row["f1"].iloc[0])))
    for model in ("lr", "svm"):
        row = metrics[
            (metrics.model == model)
            & (metrics.base_condition == "fusion")
            & (metrics["mask"] == "format")
        ]
        rows.append((f"{model.upper()} fusion\nmasked (format)", float(row["f1"].iloc[0])))

    labels = [label for label, _ in rows]
    values = [value for _, value in rows]
    colors = [COLORS["reference"], COLORS["reference"], COLORS["lr"], COLORS["svm"],
              COLORS["masked"], COLORS["masked"]]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bars = ax.barh(range(len(rows)), values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.98, 1.001)
    ax.set_xlabel("test F1 (fake class)")
    ax.set_title("A format-only model is almost as strong as full text")
    for bar, value in zip(bars, values, strict=False):
        ax.text(value + 0.0004, bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    _save(fig, out / "fig_probe_comparison")


def plot_confusion(metrics: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for ax, model in zip(axes, ("lr", "svm"), strict=False):
        row = metrics[
            (metrics.model == model)
            & (metrics.base_condition == "fusion")
            & (metrics["mask"] == "none")
        ].iloc[0]
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=int)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred real", "pred fake"])
        ax.set_yticklabels(["actual real", "actual fake"])
        ax.set_title(f"{model.upper()} fusion")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    fig.suptitle("Test confusion matrices", y=1.02)
    fig.subplots_adjust(wspace=0.45)
    _save(fig, out / "fig_confusion_fusion")


def plot_alpha(fusion: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for model in ("lr", "svm"):
        curve = fusion[model]["alpha_curve"]
        alphas = [point["alpha"] for point in curve]
        scores = [point["val_f1"] for point in curve]
        best = fusion[model]["alpha"]
        ax.plot(alphas, scores, marker="o", markersize=3,
                label=f"{model.upper()} (alpha={best:.2f})", color=COLORS[model])
        ax.axvline(best, color=COLORS[model], linestyle=":", alpha=0.5)
    ax.set_xlabel("alpha (title weight)")
    ax.set_ylabel("validation F1")
    ax.set_title("Fusion weight selected on validation")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, out / "fig_alpha_curve")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate result figures")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args(argv)

    logger = get_logger("fakenews.figures")
    cfg = load_config(args.config)
    results = Path(cfg.paths.results_dir)
    figures = ensure_dir(results / "figures")

    metrics = _metrics(results)
    shortcut = json.loads((results / "shortcut.json").read_text(encoding="utf-8"))
    fusion = json.loads((results / "fusion.json").read_text(encoding="utf-8"))

    plot_four_way(metrics, figures)
    plot_probes(metrics, shortcut, figures)
    plot_confusion(metrics, figures)
    plot_alpha(fusion, figures)
    logger.info("wrote four figures under %s", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

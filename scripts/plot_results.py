"""
Bar and radar charts comparing evaluation variants from a MetricsLogger
summary.json (scripts/metrics_logger.py).

Usage:
    python scripts/plot_results.py --summary <path/to/summary.json> \
        [--variants v1 v2 ...] [--out-dir <dir>]

By default, plots the core comparison matrix from the thesis Design
chapter (\\autoref{sec: design_variants}): the unconstrained diffusion
baseline, the non-learned baseline, post-hoc, gradient, and the three
in-loop DiMPC selection rules.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_VARIANTS = [
    "diffuser",
    "model_free",
    "post_processing",
    "gradient",
    "dpcc-r",
    "dpcc-c",
    "dpcc-t",
]

# raw summary.json keys -> display labels used in the thesis (DiMPC naming)
LABELS = {
    "diffuser": "Diffuser",
    "model_free": "Model-free",
    "post_processing": "Post-hoc",
    "gradient": "Gradient",
    "dpcc-r": "DiMPC-R",
    "dpcc-c": "DiMPC-C",
    "dpcc-t": "DiMPC-T",
}

# (summary.json field, "higher_better")
RADAR_METRICS = [
    ("success_rate_%", True),
    ("collision_rate_%", False),
    ("mean_x_progress_%", True),
    ("mean_path_length", False),
    ("mean_smoothness", False),
    ("mean_corridor_violations", False),
]
RADAR_LABELS = [
    "Success rate",
    "Collision-free rate",
    "Goal progress",
    "Path efficiency",
    "Smoothness",
    "Constraint compliance",
]


def load_summary(path, variants):
    with open(path) as f:
        data = json.load(f)
    missing = [v for v in variants if v not in data]
    if missing:
        raise KeyError(f"Variant(s) not found in {path}: {missing}")
    return {v: data[v] for v in variants}


def label_for(variant):
    return LABELS.get(variant, variant)


def plot_bar(results, out_path):
    variants = list(results.keys())
    labels = [label_for(v) for v in variants]
    success = [results[v]["success_rate_%"] for v in variants]
    collision = [results[v]["collision_rate_%"] for v in variants]

    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, success, width, label="Success rate", color="#2a9d8f")
    bars2 = ax.bar(x + width / 2, collision, width, label="Collision rate", color="#e76f51")

    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(loc="upper right")
    ax.set_title("Task success vs. constraint violation, by evaluation variant")

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    add_labels(bars1)
    add_labels(bars2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot_results] wrote {out_path}")


def normalize_radar(results, variants):
    """Min-max normalize every metric to [0, 100], flipping sign for
    lower-is-better metrics so every axis reads 'bigger = better'."""
    raw = {m: np.array([results[v][m] for v in variants], dtype=float) for m, _ in RADAR_METRICS}
    scores = np.zeros((len(variants), len(RADAR_METRICS)))
    for j, (metric, higher_better) in enumerate(RADAR_METRICS):
        vals = raw[metric]
        lo, hi = vals.min(), vals.max()
        if hi - lo < 1e-12:
            scores[:, j] = 100.0
            continue
        norm = (vals - lo) / (hi - lo)
        if not higher_better:
            norm = 1.0 - norm
        scores[:, j] = norm * 100.0
    return scores


def plot_radar(results, out_path):
    variants = list(results.keys())
    scores = normalize_radar(results, variants)

    n_axes = len(RADAR_METRICS)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    cmap = plt.get_cmap("tab10")

    for i, variant in enumerate(variants):
        values = scores[i].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, label=label_for(variant), color=cmap(i % 10))
        ax.fill(angles, values, alpha=0.08, color=cmap(i % 10))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_LABELS)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_ylim(0, 100)
    ax.set_title("Normalized per-variant performance profile\n(every axis: further out = better)", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot_results] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="Path to a MetricsLogger summary.json")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--out-dir", default="isaac/logs/MasterThesis/figures")
    parser.add_argument("--prefix", default="results", help="Output filename prefix")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = load_summary(args.summary, args.variants)

    plot_bar(results, os.path.join(args.out_dir, f"{args.prefix}_bar.pdf"))
    plot_radar(results, os.path.join(args.out_dir, f"{args.prefix}_radar.pdf"))


if __name__ == "__main__":
    main()

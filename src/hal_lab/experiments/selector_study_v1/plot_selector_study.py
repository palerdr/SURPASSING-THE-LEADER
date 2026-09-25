"""Render frozen selector results without changing training or selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "small": "Small MLP · 3 × 7,256",
    "wide": "Residual MLP · 3 × 161,560",
    "attention": "Expert attention · 3 × 24,001",
    "old": "Old Hal", "translated": "Translated Hal",
    "pilot": "Prior pilot · 7,256",
    "no_errors": "Remove recent-error inputs",
    "no_context": "Remove public-context inputs",
    "no_prior": "Remove inherited expert weights",
    "no_shapes": "Remove forecast-shape inputs",
    "static": "Use 24 learned constants",
    "untrained": "Leave network untrained",
    "single_seed": "Use one training seed",
}


def interval(ax, row, value, color, *, scale=100):
    mean = scale * value["mean"]
    lower, upper = np.asarray(value["interval_95"]) * scale
    ax.errorbar(mean, row, xerr=[[mean - lower], [upper - mean]],
        fmt="o", color=color, capsize=4, markersize=7, linewidth=1.8)
    return mean, lower, upper


def plot(output):
    result = json.loads((output / "results.json").read_text())
    selected = result["selected"]
    summary = result["summary"]["all"]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False,
        "axes.titleweight": "bold", "figure.facecolor": "#fafaf7",
        "axes.facecolor": "#fafaf7", "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.8), gridspec_kw={"width_ratios": [1, 1.2]})
    fig.subplots_adjust(left=.15, right=.97, bottom=.19, top=.78, wspace=.8)
    fig.suptitle("Neural selectors: capacity and ablation", x=.035, y=.96, ha="left", fontsize=23, weight="bold")
    games = summary[selected]["games"]
    fig.text(.035, .9, f"Frozen validation choice: {LABELS[selected]} parameters  |  {games:,} fresh games per variant", fontsize=12)
    ax = axes[0]
    names = ["old", "translated", "pilot", "small", "wide", "attention"]
    for row, name in enumerate(names):
        color = "#008473" if name == selected else "#42566d"
        interval(ax, row, summary[name]["score"], color)
        ax.text(1.02, row, f"{summary[name]['wins']:,}/{games:,}", transform=ax.get_yaxis_transform(), va="center", fontsize=10, color=color)
    ax.set_yticks(range(len(names)), [LABELS[name] + ("  ★" if name == selected else "") for name in names])
    ax.invert_yaxis()
    ax.set_title("Architecture holdout", loc="left", pad=18)
    ax.set_xlabel("Game score (%)")
    ax.xaxis.grid(True, alpha=.18)
    ax.set_axisbelow(True)

    ax = axes[1]
    controls = result["selected_minus_controls"]
    names = [name for name in ("no_errors", "no_context", "no_prior", "no_shapes", "static", "untrained", "single_seed") if name in controls]
    for row, name in enumerate(names):
        value = controls[name]["score"]
        color = "#008473" if value["interval_95"][0] > 0 else "#42566d"
        mean, lower, upper = interval(ax, row, value, color)
        ax.text(1.02, row, f"{mean:+.2f} [{lower:+.2f}, {upper:+.2f}]", transform=ax.get_yaxis_transform(), va="center", fontsize=10, color=color)
    ax.axvline(0, color="#8b8e93", linewidth=1, linestyle="--")
    ax.set_yticks(range(len(names)), [LABELS[name] for name in names])
    ax.invert_yaxis()
    ax.set_title("Retrained ablations and controls", loc="left", pad=18)
    ax.set_xlabel("Selected model minus control\n(percentage points; positive favors full model)")
    ax.xaxis.grid(True, alpha=.18)
    ax.set_axisbelow(True)
    fig.text(.035, .075, "Dots: observed means. Bars: unadjusted 95% paired identity-bootstrap intervals on the right; identity intervals on the left.", fontsize=10, color="#586473")
    fig.text(.035, .04, "Three training seeds per new fitted variant. Four games per opponent identity. Synthetic opponents; no human win-rate claim.", fontsize=10, color="#586473")
    for extension in ("png", "svg"):
        fig.savefig(output / f"ablation.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    training = json.loads((output / "training.json").read_text())["history"]
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"small": "#42566d", "wide": "#008473", "attention": "#b15f29"}
    for name, color in colors.items():
        histories = np.asarray([[r["validation_nll"] for r in member["history"]] for member in training[name]])
        epochs = np.arange(1, histories.shape[1] + 1)
        for history in histories:
            ax.plot(epochs, history, color=color, alpha=.22, linewidth=1)
        ax.plot(epochs, histories.mean(0), color=color, linewidth=2.5, label=LABELS[name])
    ax.set(title="Validation learning curves", xlabel="Training epoch", ylabel="Next-action NLL (nats; lower is better)")
    ax.legend(frameon=False)
    ax.grid(alpha=.18)
    fig.text(.1, .01, "Thin lines: individual seeds. Thick lines: mean across seeds. Equal weight per opponent identity.", fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, 1))
    for extension in ("png", "svg"):
        fig.savefig(output / f"learning-curves.{extension}", dpi=170)
    plt.close(fig)

    families = list(result["mean_selected_weights_on_test_corpus"])
    weights = np.asarray([result["mean_selected_weights_on_test_corpus"][family] for family in families])
    top = np.argsort(weights.mean(0))[-10:][::-1]
    fig, ax = plt.subplots(figsize=(15, 8))
    heatmap = ax.imshow(100 * weights[:, top].T, cmap="YlGnBu", aspect="auto", vmin=0)
    ax.set_xticks(range(len(families)), [name.replace("_", " ") for name in families], rotation=55, ha="right", fontsize=9)
    ax.set_yticks(range(len(top)), [result["expert_names"][i].replace("_", " ") for i in top], fontsize=10)
    ax.set_title(f"{selected.title()} selector: expert weights across opponent families", loc="left", pad=18)
    fig.colorbar(heatmap, ax=ax, shrink=.7, label="Mean forecast weight (%)")
    fig.text(.02, .018, "Ten largest mean weights across families. Public test histories from fixed behavior policies. Weights describe the model; they do not measure causal importance.", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for extension in ("png", "svg"):
        fig.savefig(output / f"expert-weights.{extension}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    plot(parser.parse_args().output)

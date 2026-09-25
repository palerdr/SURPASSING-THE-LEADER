"""Reconstruct Hal's clock-dependent strategies from the completed STL table.

Run from the repository root with:
    uv run --with matplotlib python paper/make_stl_figures.py

The script reads the table through ``stl.reader``, which checks the manifest
and the DTH tail hash. Set STL_LEAP_ARTIFACT or STL_LEAP_DTH to an absolute
path to read another copy.
"""

from pathlib import Path
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stl.reader import open_leap

OUTPUT = ROOT / "paper/build/figures/stl"


def source(path):
    """Name the artifact by its repository path when it lies inside the repository."""
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def reconstruct():
    """Hold both profiles fixed and certify each reachable clock slice."""
    table = open_leap()
    identity = table.identity
    checker = table.profile_index(st=60, ttd=120)
    dropper = table.profile_index(st=0, ttd=180)
    # stage() raises when a Bellman residual or a saddle gap exceeds 1e-6.
    records = [{"minute": minute, **table.stage(("H1", minute), checker, dropper)}
               for minute in range(44, 60)]
    evidence = {
        "source": source(identity.artifact),
        "builder_sha256": identity.builder_sha256,
        "dth_sha256": identity.dth_sha256,
        "hal": {"st": 0, "ttd": 180, "role": "dropper"},
        "baku": {"st": 60, "ttd": 120, "role": "checker"},
        "records": records,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "strategy_surface.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return records


def render(records):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.linewidth": 0.5, "pdf.fonttype": 42,
    })
    seconds, minutes = np.meshgrid(np.arange(1, 61), [r["minute"] for r in records])
    probability = 100 * np.array([r["drop"] for r in records])
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 100):
        raise ValueError("surface rows must be probability distributions")
    fig = plt.figure(figsize=(3.15, 3.25))
    ax = fig.add_axes([0.12, 0.15, 0.84, 0.79], projection="3d")
    fig.text(0.52, 0.96, "Hal's drop probability", ha="center", fontsize=9)
    colors = LinearSegmentedColormap.from_list("paper", ["#e5f2f3", "#269b9d", "#234c7c"])
    ax.plot_surface(seconds, minutes, probability, cmap=colors,
                    rcount=len(records), ccount=60, vmin=0, vmax=100,
                    linewidth=0.18, edgecolor="#48757e", antialiased=True)
    ax.set(xlim=(1, 60), ylim=(44, 59), zlim=(0, 100))
    ax.set_xticks([1, 20, 40, 60])
    ax.set_yticks([44, 49, 54, 59], ["8:44", "8:49", "8:54", "8:59"])
    ax.set_zticks([0, 50, 100], ["0%", "50%", "100%"])
    ax.set_xlabel("Drop second", labelpad=5)
    ax.set_ylabel("Clock", labelpad=7)
    ax.tick_params(pad=0)
    ax.view_init(elev=27, azim=-128)
    ax.set_box_aspect((1.5, 1.25, 0.85))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
    fig.savefig(OUTPUT / "strategy_surface.pdf")
    fig.savefig(OUTPUT / "strategy_surface.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    data = reconstruct()
    render(data)
    print(f"Certified {len(data)} clock slices; maximum gap {max(r['gap'] for r in data):.3g}")

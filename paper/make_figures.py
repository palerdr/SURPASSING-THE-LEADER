"""Render the paper figures with Seaborn and Matplotlib.

The numerical figures fail closed when their generated inputs are missing or
malformed. Run from anywhere with, for example:

    uv run --with matplotlib --with seaborn --with pandas \
        python paper/make_figures.py

The earlier monolithic Gemini plotting draft is deprecated. It silently
substituted mock strategies and values when solver outputs were absent, so its
figures were not evidence-bearing. This renderer is the canonical replacement:
``paper/generate_figure_data.py`` produces the sampled solver data, and every
consumer below validates those inputs before drawing.

Chart contracts
---------------
1. Revival: expose the two decay mechanisms and the survivability cliff.
2. Toeplitz: connect matrix structure to the O(60) saddle scan.

The quotient, potential-DAG, and root-strategy figures are deliberately
rendered in TeX; their simpler geometry is clearer in the paper than the
Seaborn alternatives.

All figures use a white paper surface, explicit palette roots, and line style
or geometry in addition to color. Outputs are vector PDF/SVG plus PNG previews.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd
import seaborn as sns


PAPER_DIR = Path(__file__).resolve().parent
DATA_DIR = PAPER_DIR / "build" / "figures"
OUTPUT_DIR = DATA_DIR / "seaborn"
TEXT_WIDTH = 6.5

BLUE = "#2A6FB0"
ORANGE = "#D1622B"
TEAL = "#1D9E75"
PURPLE = "#534AB7"
INK = "#242424"
MID_GRAY = "#707070"
LIGHT_GRAY = "#E8E8E8"
PALE_GRAY = "#F5F5F5"


def configure_style() -> None:
    """Apply one restrained Seaborn style across every figure."""

    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        font_scale=0.9,
        rc={
            "axes.edgecolor": "#777777",
            "axes.labelcolor": INK,
            "axes.linewidth": 0.6,
            "axes.titleweight": "bold",
            "axes.titlesize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "text.color": INK,
            "xtick.color": "#555555",
            "ytick.color": "#555555",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def load_table(name: str, expected_columns: tuple[str, ...]) -> pd.DataFrame:
    """Load one generated table and reject missing, incomplete, or nonfinite data."""

    path = DATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run paper/generate_figure_data.py first"
        )
    frame = pd.read_csv(path, sep=r"\s+")
    if tuple(frame.columns) != expected_columns:
        raise ValueError(
            f"{path} has columns {tuple(frame.columns)}, expected {expected_columns}"
        )
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{path} is empty or contains nonfinite values")
    return frame


def save_figure(fig: plt.Figure, stem: str, *, tight: bool = True) -> None:
    """Save one figure in the paper format and two review formats."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        kwargs: dict[str, object] = {}
        if tight:
            kwargs.update({"bbox_inches": "tight", "pad_inches": 0.04})
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(OUTPUT_DIR / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def make_revival_probability() -> None:
    """Show the revival law and its zero plateau over the full ST-TTD domain."""

    fig = plt.figure(figsize=(TEXT_WIDTH, 3.55))
    ax = fig.add_subplot(111, projection="3d")

    st = np.linspace(0, 240, 61)
    fraction = np.linspace(0, 1, 51)
    ST, U = np.meshgrid(st, fraction)
    TTD = U * (240 - ST)
    P = 0.95 * (1 - ST / 240) * 0.75 ** (TTD / 60)

    # The formula applies on ST + TTD <= 240.  Parameterize the complementary
    # triangle separately so the full-domain plot shows the exact p=0 plateau,
    # including the (240, 240) corner, without interpolating across the cliff.
    zero_ttd = 240 - ST + U * ST
    ax.plot_surface(
        ST,
        zero_ttd,
        np.zeros_like(ST),
        color=PALE_GRAY,
        edgecolor="#D6D6D6",
        linewidth=0.18,
        antialiased=True,
        alpha=0.90,
        shade=False,
    )

    surface = ax.plot_surface(
        ST,
        TTD,
        P,
        cmap=sns.color_palette("crest", as_cmap=True),
        vmin=0,
        vmax=0.95,
        linewidth=0,
        antialiased=True,
        alpha=0.97,
    )

    boundary_s = np.linspace(0, 240, 100)
    boundary_t = 240 - boundary_s
    boundary_p = 0.95 * (1 - boundary_s / 240) * 0.75 ** (boundary_t / 60)
    curtain_z = np.vstack((np.zeros_like(boundary_p), boundary_p))
    ax.plot_surface(
        np.vstack((boundary_s, boundary_s)),
        np.vstack((boundary_t, boundary_t)),
        curtain_z,
        color=ORANGE,
        alpha=0.42,
        linewidth=0,
        shade=False,
    )
    ax.plot(boundary_s, boundary_t, boundary_p, color=ORANGE, lw=1.5)
    ax.plot(st, np.zeros_like(st), 0.95 * (1 - st / 240), color=BLUE, lw=1.7)
    ax.plot(
        np.zeros_like(st),
        st,
        0.95 * 0.75 ** (st / 60),
        color=PURPLE,
        lw=1.7,
        ls="--",
    )

    ax.set(
        xlim=(0, 240),
        ylim=(0, 240),
        zlim=(0, 1),
        xlabel="ST $s$",
        ylabel="TTD $t$",
        zlabel="$p(s,t)$",
        title="Frozen revival probability over ST and TTD",
    )
    ax.set_xticks((0, 60, 120, 180, 240))
    ax.set_yticks((0, 60, 120, 180, 240))
    ax.set_zticks((0, 0.25, 0.5, 0.75, 1))
    ax.view_init(elev=27, azim=-45)
    ax.set_proj_type("persp", focal_length=0.90)
    ax.set_box_aspect((1.30, 1.0, 1.0))
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.scatter([0], [0], [0.95], color=ORANGE, edgecolor="white", s=28, zorder=8)
    ax.text(4, 6, 0.98, "$p(0,0)=0.95$", color=ORANGE, fontsize=8.5)
    ax.text(98, 142, 0.36, "$s+t=240$", color=ORANGE, fontsize=8)
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.16, top=0.90)
    save_figure(fig, "fig1_revival_probability", tight=False)


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    textcolor: str = INK,
) -> patches.FancyBboxPatch:
    """Draw a consistently styled flow box in data coordinates."""

    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.8,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color=textcolor,
        fontsize=8.2,
        linespacing=1.25,
    )
    return box


def make_toeplitz_structure() -> None:
    """Pair the Toeplitz heatmap with the exact linear-time reduction flow."""

    fig = plt.figure(figsize=(TEXT_WIDTH, 3.25))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.15, 0.85),
        left=0.075,
        right=0.985,
        bottom=0.15,
        top=0.90,
        wspace=0.28,
    )
    ax = fig.add_subplot(grid[0, 0])

    n = 8
    values = np.zeros((n, n), dtype=int)
    labels = np.empty((n, n), dtype=object)
    for row in range(n):
        for column in range(n):
            if column >= row:
                lag = column - row + 1
                values[row, column] = lag
                labels[row, column] = f"$S_{{{lag}}}$"
            else:
                values[row, column] = 0
                labels[row, column] = "$F$"

    teal_ramp = sns.light_palette(TEAL, n_colors=n, reverse=False)
    cmap = ListedColormap([LIGHT_GRAY, *teal_ramp])
    norm = BoundaryNorm(np.arange(-0.5, n + 1.5), cmap.N)
    sns.heatmap(
        values,
        annot=labels,
        fmt="",
        cmap=cmap,
        norm=norm,
        cbar=False,
        square=True,
        linewidths=0,
        annot_kws={"fontsize": 7.2, "color": INK},
        ax=ax,
    )
    boundaries = np.arange(n + 1)
    ax.vlines(boundaries, 0, n, colors="white", linewidth=0.7, zorder=3)
    ax.hlines(boundaries, 0, n, colors="white", linewidth=0.7, zorder=3)
    ax.set(
        title="Eight-action view of the stage matrix",
        xlabel="Checker action $c$",
        ylabel="Dropper action $d$",
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticklabels(range(1, n + 1), rotation=0)
    ax.set_yticklabels(range(1, n + 1), rotation=0)
    ax.add_patch(
        patches.Rectangle(
            (0.04, 3.04),
            n - 0.08,
            0.92,
            fill=False,
            ec=ORANGE,
            lw=1.35,
            zorder=4,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (5.04, 0.04),
            0.92,
            n - 0.08,
            fill=False,
            ec=PURPLE,
            lw=1.35,
            zorder=4,
        )
    )

    flow = fig.add_subplot(grid[0, 1])
    flow.axis("off")
    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)
    rounded_box(
        flow,
        (0.06, 0.73),
        0.88,
        0.19,
        "$60\\times60$ stage matrix\n$3{,}600$ cells",
        facecolor=PALE_GRAY,
        edgecolor="#A0A0A0",
    )
    rounded_box(
        flow,
        (0.06, 0.42),
        0.88,
        0.19,
        "at most $61$ distinct values\n$S_1,\\ldots,S_{60}$ and $F$",
        facecolor="#E8F4F0",
        edgecolor=TEAL,
    )
    rounded_box(
        flow,
        (0.06, 0.11),
        0.88,
        0.19,
        "prefix minima and maxima\nexact saddle test in $O(60)$",
        facecolor="#F1EEFB",
        edgecolor=PURPLE,
    )
    for y0, y1 in ((0.73, 0.61), (0.42, 0.30)):
        flow.annotate(
            "",
            xy=(0.5, y1 + 0.10),
            xytext=(0.5, y0),
            arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.2},
        )
    save_figure(fig, "fig2_toeplitz_structure")


def make_quotient_geometry() -> None:
    """Show the dead region, representative fiber collapses, and squared count."""

    fig = plt.figure(figsize=(TEXT_WIDTH, 3.65))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.05, 0.9, 1.15),
        height_ratios=(1, 0.34),
        left=0.055,
        right=0.985,
        bottom=0.08,
        top=0.92,
        wspace=0.32,
        hspace=0.30,
    )

    raw = fig.add_subplot(grid[0, 0])
    raw.set_xlim(0, 300)
    raw.set_ylim(0, 300)
    raw.fill(
        [0, 300, 300, 0],
        [0, 0, 300, 300],
        color=LIGHT_GRAY,
        ec="#A5A5A5",
        lw=0.7,
    )
    raw.fill([0, 240, 0], [0, 0, 240], color="#C9E9DF", ec=TEAL, lw=0.8)
    raw.plot([0, 240], [240, 0], color=TEAL, lw=1.2)
    raw.text(70, 62, "alive\n16,711", color="#136A50", ha="center", fontsize=8.2)
    raw.annotate(
        "death-eligible\n55,889",
        xy=(205, 215),
        xytext=(120, 262),
        arrowprops={"arrowstyle": "->", "color": MID_GRAY, "lw": 0.8},
        color=MID_GRAY,
        ha="center",
        fontsize=8.0,
    )
    raw.set(
        title="Raw one-player profiles",
        xlabel="ST $s$",
        ylabel="TTD $t$",
        xticks=(0, 120, 240, 300),
        yticks=(0, 120, 240, 300),
    )
    raw.set_aspect("equal")
    raw.grid(False)

    fibers = fig.add_subplot(grid[0, 1])
    fibers.set_xlim(-0.05, 1.05)
    fibers.set_ylim(-0.08, 1.05)
    fibers.axis("off")
    fibers.set_title("Collapse each dead TTD fiber", pad=5)
    sample_x = (0.17, 0.50, 0.83)
    sample_st = (60, 150, 230)
    for x, st_value in zip(sample_x, sample_st, strict=True):
        threshold = (240 - st_value) / 300
        alive_y = np.linspace(0.08, max(0.09, threshold), 3)
        dead_y = np.linspace(max(threshold + 0.08, 0.27), 0.92, 5)
        fibers.scatter(
            np.full_like(alive_y, x),
            alive_y,
            s=14,
            facecolors="#C9E9DF",
            edgecolors=TEAL,
            linewidths=0.6,
            zorder=3,
        )
        fibers.scatter(
            np.full_like(dead_y, x),
            dead_y,
            s=13,
            facecolors=LIGHT_GRAY,
            edgecolors=MID_GRAY,
            linewidths=0.5,
            zorder=3,
        )
        sentinel_y = 0.02
        for source_y in dead_y[1::2]:
            fibers.annotate(
                "",
                xy=(x, sentinel_y + 0.025),
                xytext=(x, source_y - 0.025),
                arrowprops={"arrowstyle": "->", "color": PURPLE, "lw": 0.55},
            )
        fibers.scatter(
            [x],
            [sentinel_y],
            s=32,
            facecolors=PURPLE,
            edgecolors="white",
            linewidths=0.6,
            zorder=4,
        )
        fibers.text(x, -0.055, f"$s={st_value}$", ha="center", fontsize=7.2)
    counts = fig.add_subplot(grid[0, 2])
    counts.axis("off")
    counts.set_xlim(0, 1)
    counts.set_ylim(0, 1)
    counts.set_title("One-player quotient", pad=5)
    rounded_box(
        counts,
        (0.05, 0.65),
        0.90,
        0.20,
        "$72{,}600$ profiles\n$16{,}711$ alive + $55{,}889$ dead",
        facecolor=PALE_GRAY,
        edgecolor="#A0A0A0",
    )
    counts.annotate(
        "",
        xy=(0.5, 0.48),
        xytext=(0.5, 0.65),
        arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.3},
    )
    counts.text(0.56, 0.565, "quotient", color=ORANGE, fontsize=7.3, va="center")
    rounded_box(
        counts,
        (0.05, 0.27),
        0.90,
        0.20,
        "$17{,}011$ classes\n$16{,}711$ alive + $300$ sentinels",
        facecolor="#F1EEFB",
        edgecolor=PURPLE,
    )

    combined = fig.add_subplot(grid[1, :])
    combined.axis("off")
    combined.set_xlim(0, 1)
    combined.set_ylim(0, 1)
    rounded_box(
        combined,
        (0.02, 0.08),
        0.29,
        0.70,
        "$72{,}600^2$ states\n$\\approx 5.27$ billion",
        facecolor=PALE_GRAY,
        edgecolor="#A0A0A0",
    )
    rounded_box(
        combined,
        (0.69, 0.08),
        0.29,
        0.70,
        "$17{,}011^2$ classes\n$289{,}374{,}121$",
        facecolor="#F1EEFB",
        edgecolor=PURPLE,
    )
    combined.annotate(
        "",
        xy=(0.68, 0.43),
        xytext=(0.32, 0.43),
        arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.5},
    )
    combined.text(
        0.50,
        0.55,
        "apply independently to both players",
        ha="center",
        color=ORANGE,
        fontsize=7.5,
    )
    combined.text(
        0.50,
        0.20,
        "reduction factor $18.2$",
        ha="center",
        color=PURPLE,
        fontsize=8.4,
        weight="bold",
    )

    bridge = ConnectionPatch(
        xyA=(300, 150),
        coordsA=raw.transData,
        xyB=(-0.02, 0.53),
        coordsB=fibers.transData,
        arrowstyle="-|>",
        color=ORANGE,
        lw=1.1,
        connectionstyle="arc3,rad=-0.04",
    )
    fig.add_artist(bridge)
    save_figure(fig, "fig3_quotient_geometry")


def make_root_strategies() -> None:
    """Plot the exact root policies with an honest log probability scale."""

    frame = load_table(
        "root_strategies.dat", ("action", "drop", "check")
    ).sort_values("action")
    if not np.array_equal(frame["action"].to_numpy(), np.arange(1, 61)):
        raise ValueError("root_strategies.dat must contain actions 1 through 60")
    for column in ("drop", "check"):
        if (frame[column] <= 0).any():
            raise ValueError(f"{column} strategy is not full support")
        if not np.isclose(frame[column].sum(), 1.0, atol=1e-8):
            raise ValueError(f"{column} strategy does not sum to one")

    long = frame.rename(
        columns={"drop": r"Dropper $\sigma_d$", "check": r"Checker $\sigma_c$"}
    ).melt(
        id_vars="action",
        var_name="Player",
        value_name="Probability",
    )
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.15))
    sns.lineplot(
        data=long,
        x="action",
        y="Probability",
        hue="Player",
        style="Player",
        palette={r"Dropper $\sigma_d$": BLUE, r"Checker $\sigma_c$": ORANGE},
        dashes={r"Dropper $\sigma_d$": "", r"Checker $\sigma_c$": (4, 2)},
        linewidth=1.65,
        ax=ax,
    )
    ax.scatter(
        frame.loc[::5, "action"],
        frame.loc[::5, "drop"],
        s=11,
        facecolor="white",
        edgecolor=BLUE,
        linewidth=0.7,
        zorder=3,
    )
    ax.scatter(
        frame.loc[::5, "action"],
        frame.loc[::5, "check"],
        s=10,
        marker="s",
        facecolor="white",
        edgecolor=ORANGE,
        linewidth=0.7,
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set(
        title="Certified equilibrium strategies at $(0,0,0,0)$",
        xlabel="Second",
        ylabel="Probability (log scale)",
        xlim=(0.5, 60.5),
    )
    ax.set_xticks((1, 10, 20, 30, 40, 50, 60))
    root_mass = float(frame.loc[frame["action"] == 1, "drop"].iloc[0])
    minimum_probability = float(frame[["drop", "check"]].to_numpy().min())
    maximum_probability = float(frame[["drop", "check"]].to_numpy().max())
    ax.set_ylim(minimum_probability * 0.82, maximum_probability * 1.22)
    ax.annotate(
        f"Dropper places {root_mass:.1%} on second 1",
        xy=(1, root_mass),
        xytext=(8, root_mass * 0.63),
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 0.8},
        color=BLUE,
        fontsize=8,
    )
    ax.legend(loc="upper right", ncols=2, title=None)
    sns.despine(ax=ax)
    fig.tight_layout(pad=0.6)
    save_figure(fig, "fig4_root_strategies")


def main() -> None:
    configure_style()
    make_revival_probability()
    make_toeplitz_structure()
    print(f"Wrote figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

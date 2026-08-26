"""Generate the exact data behind the paper's layer-width and root-strategy figures.

Both come from the compact solver in ``src/dth_compact``: the layer widths are
a census of its profile buckets, and the root strategies are the equalizer
pair (Proposition "One recurrence, both strategies") re-derived from the
finished table ``src/dth_compact/artifacts/V.npy``, which ``uv run main.py``
inside ``src/dth_compact`` writes. Run with that project's environment:

    uv run --project src/dth_compact python paper/generate_figure_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
COMPACT = REPOSITORY / "src" / "dth_compact"
OUTPUT = REPOSITORY / "paper" / "build" / "figures"

sys.path.insert(0, str(COMPACT))
import main as solver  # noqa: E402


def write_layer_widths(table) -> None:
    """The exact number of classes in every potential layer: a convolution of the bucket sizes."""
    sizes = np.array([bucket.size for bucket in table.bucket], dtype=np.int64)
    widths = np.convolve(sizes, sizes)
    if widths.size != solver.MAX_LAYER + 1 or int(widths.sum()) != solver.CLASSES:
        raise RuntimeError("potential-layer widths do not cover the quotient table")
    with (OUTPUT / "layer_widths.dat").open("w", encoding="ascii") as handle:
        handle.write("potential classes\n")
        for potential, width in enumerate(widths):
            handle.write(f"{potential} {int(width)}\n")


def write_root_strategies(table, V: np.ndarray) -> None:
    """The root's certified equalizer pair, re-derived from the stored child values."""
    pc, pd = solver.decode_class(solver.encode_state(0, 0, 0, 0, table))
    s, f = solver.class_values(pc, pd, V, table)
    result = solver.try_rung2(s, f)
    if result is None or not result.certified:
        raise RuntimeError("the root is not equalizer-certified; the paper's figure assumes it is")
    if abs(result.value - float(V[pc, pd])) > solver.MAX_SADDLE_GAP:
        raise RuntimeError("root certificate does not match the stored value")
    with (OUTPUT / "root_strategies.dat").open("w", encoding="ascii") as handle:
        handle.write("action drop check\n")
        for action in range(solver.LAGS):
            handle.write(f"{action + 1} {result.drop[action]:.10f} {result.check[action]:.10f}\n")


def main() -> None:
    values = COMPACT / "artifacts" / "V.npy"
    if not values.is_file():
        raise SystemExit(f"missing {values}; run `uv run main.py` inside {COMPACT} first")
    table = solver.build_table()
    V = np.load(values, mmap_mode="r")
    if V.shape != (solver.N, solver.N + 1):
        raise SystemExit(f"{values} has shape {V.shape}, expected {(solver.N, solver.N + 1)}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_layer_widths(table)
    write_root_strategies(table, V)
    print(f"wrote layer_widths.dat and root_strategies.dat to {OUTPUT}")


if __name__ == "__main__":
    main()

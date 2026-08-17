"""Generate the exact data used by the paper's root-strategy figure."""

from pathlib import Path

import numpy as np

from dth.complete_tablebase import CompleteTablebase
from dth.packed import PROFILE_COUNT, build_profile_table, encode_class, profile_id
from dth.solver import reconstruct_transition_class_matrix
from dth.support_solver import solve_certified_matrix_fast


REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPOSITORY / "src" / "dth" / "artifacts" / "complete_full_v1"
OUTPUT = REPOSITORY / "paper" / "build" / "figures"


def write_layer_widths(profiles: object) -> None:
    """Write the exact number of quotient classes in every potential layer."""

    bucket_sizes = np.asarray(
        [len(bucket) for bucket in profiles.bucket_profiles], dtype=np.int64
    )
    widths = np.convolve(bucket_sizes, bucket_sizes)
    if widths.size != 1201 or int(widths.sum()) != PROFILE_COUNT**2:
        raise RuntimeError("potential-layer widths do not cover the quotient table")

    with (OUTPUT / "layer_widths.dat").open("w", encoding="ascii") as handle:
        handle.write("potential classes\n")
        for potential, width in enumerate(widths):
            handle.write(f"{potential} {int(width)}\n")


def main() -> None:
    profiles = build_profile_table()
    tablebase = CompleteTablebase(ARTIFACT_DIR)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_layer_widths(profiles)

    def class_matrix(checker: int, dropper: int) -> np.ndarray:
        success = np.empty(60, dtype=np.float64)
        for lag in range(1, 61):
            child = int(profiles.success_child_by_profile[checker, lag - 1])
            success[lag - 1] = (
                1.0
                if child < 0
                else -tablebase.value_of_class(dropper * PROFILE_COUNT + child)
            )

        failure_child = int(profiles.failure_child_by_profile[checker])
        if failure_child < 0:
            failed = 1.0
        else:
            revival = float(profiles.revival_by_profile[checker])
            continuation = -tablebase.value_of_class(
                dropper * PROFILE_COUNT + failure_child
            )
            failed = revival * continuation + (1.0 - revival)
        return reconstruct_transition_class_matrix(success, failed)

    root_profile = profile_id(0, 0)
    root_matrix = class_matrix(root_profile, root_profile)
    root_value, drop, check, _ = solve_certified_matrix_fast(root_matrix)
    stored_root = tablebase.value_of_class(encode_class((0, 0, 0, 0)))
    if abs(root_value - stored_root) > 1e-6:
        raise RuntimeError("root certificate does not match the stored value")

    with (OUTPUT / "root_strategies.dat").open("w", encoding="ascii") as handle:
        handle.write("action drop check\n")
        for action in range(60):
            handle.write(
                f"{action + 1} {drop[action]:.10f} {check[action]:.10f}\n"
            )


if __name__ == "__main__":
    main()

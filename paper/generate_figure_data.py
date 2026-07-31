"""Generate the sampled data used by the paper's strategy figures."""

from pathlib import Path

import numpy as np

from dth.packed import PROFILE_COUNT, build_profile_table, encode_class, profile_id
from dth.solver import reconstruct_transition_class_matrix
from dth.support_solver import solve_certified_matrix_fast


REPOSITORY = Path(__file__).resolve().parents[1]
VALUE_TABLE = (
    REPOSITORY / "src" / "dth" / "artifacts" / "backup_full_v1" / "value.npy"
)
OUTPUT = REPOSITORY / "paper" / "build" / "figures"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    profiles = build_profile_table()
    values = np.load(VALUE_TABLE, mmap_mode="r")

    def class_matrix(checker: int, dropper: int) -> np.ndarray:
        success = np.empty(60, dtype=np.float64)
        for lag in range(1, 61):
            child = int(profiles.success_child_by_profile[checker, lag - 1])
            success[lag - 1] = (
                1.0 if child < 0 else -float(values[dropper * PROFILE_COUNT + child])
            )

        failure_child = int(profiles.failure_child_by_profile[checker])
        if failure_child < 0:
            failed = 1.0
        else:
            revival = float(profiles.revival_by_profile[checker])
            continuation = -float(
                values[dropper * PROFILE_COUNT + failure_child]
            )
            failed = revival * continuation + (1.0 - revival)
        return reconstruct_transition_class_matrix(success, failed)

    root_profile = profile_id(0, 0)
    root_matrix = class_matrix(root_profile, root_profile)
    root_value, drop, check, _ = solve_certified_matrix_fast(root_matrix)
    stored_root = float(values[encode_class((0, 0, 0, 0))])
    if abs(root_value - stored_root) > 1e-6:
        raise RuntimeError("root certificate does not match the stored value")

    with (OUTPUT / "root_strategies.dat").open("w", encoding="ascii") as handle:
        handle.write("action drop check\n")
        for action in range(60):
            handle.write(
                f"{action + 1} {drop[action]:.10f} {check[action]:.10f}\n"
            )

    with (OUTPUT / "diag_strategy.dat").open("w", encoding="ascii") as handle:
        handle.write("s action prob\n")
        for load in range(0, 300, 10):
            pid = profile_id(load, 0)
            _, drop_at_load, _, _ = solve_certified_matrix_fast(
                class_matrix(pid, pid)
            )
            for action in range(60):
                handle.write(
                    f"{load} {action + 1} {drop_at_load[action]:.10f}\n"
                )
            handle.write("\n")

    loads = range(0, 300, 10)
    with (OUTPUT / "value_surface.dat").open("w", encoding="ascii") as handle:
        handle.write("sc sd value\n")
        for checker_load in loads:
            checker = profile_id(checker_load, 0)
            for dropper_load in loads:
                dropper = profile_id(dropper_load, 0)
                value = float(values[checker * PROFILE_COUNT + dropper])
                handle.write(f"{checker_load} {dropper_load} {value:.10f}\n")
            handle.write("\n")


if __name__ == "__main__":
    main()

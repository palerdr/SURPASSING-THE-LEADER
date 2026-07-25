"""Resumable memory-mapped tablebase generation for fine bucket grids."""

from __future__ import annotations

import importlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from abstract.artifacts import canonical_json, digest_files, digest_json, sha256_file
from abstract.matrix import saddle_gap as matrix_saddle_gap
from abstract.matrix import solve_matrix
from abstract.packed import PackedStateCodec, packed_live_successors
from abstract.rules import (
    LEGACY_REVIVAL_MODEL,
    UNIFIED_REVIVAL_MODEL,
    AbstractRuleset,
    TIMING_CONVENTION_ID,
)
from abstract.state import AbstractState
from abstract.tablebase import state_id


PACKED_TABLEBASE_SCHEMA = "abstract.packed-tablebase.v3"
PACKED_BUILD_SCHEMA = "abstract.packed-tablebase-build.v2"
UNREACHABLE_ORDINAL = np.iinfo(np.uint32).max

_ARRAY_SPECS: dict[str, tuple[str, tuple[str, ...]]] = {
    "state_index": ("uint32", ("reachable",)),
    "ordinal_by_index": ("uint32", ("physical",)),
    "value": ("float64", ("reachable",)),
    "drop_policy": ("float32", ("reachable", "actions")),
    "check_policy": ("float32", ("reachable", "actions")),
    "saddle_gap": ("float64", ("reachable",)),
    "dropper_win_probability": ("float64", ("reachable",)),
    "checker_win_probability": ("float64", ("reachable",)),
}


def _rust_revival_model_code(rules: AbstractRuleset) -> int:
    return {
        LEGACY_REVIVAL_MODEL: 0,
        UNIFIED_REVIVAL_MODEL: 1,
    }[rules.revival_model_kind]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".json",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _open_npy(path: Path, *, mode: str, dtype: str, shape: tuple[int, ...]) -> np.memmap:
    if mode == "w+":
        return np.lib.format.open_memmap(path, mode=mode, dtype=np.dtype(dtype), shape=shape)
    loaded = np.load(path, mmap_mode=mode, allow_pickle=False)
    if loaded.shape != shape or loaded.dtype != np.dtype(dtype):
        raise ValueError(
            f"packed array {path.name} has shape/dtype {loaded.shape}/{loaded.dtype}, "
            f"expected {shape}/{np.dtype(dtype)}"
        )
    return loaded


def _potential_vector(indices: np.ndarray, codec: PackedStateCodec) -> np.ndarray:
    values = np.asarray(indices, dtype=np.uint64)
    quotient, dropper_ttd = np.divmod(values, codec.ttd_size)
    quotient, dropper_load = np.divmod(quotient, codec.load_cap_units)
    checker_load, checker_ttd = np.divmod(quotient, codec.ttd_size)
    return (
        checker_load.astype(np.uint16)
        + checker_ttd.astype(np.uint16)
        + dropper_load.astype(np.uint16)
        + dropper_ttd.astype(np.uint16)
    )


@dataclass
class PackedTablebaseBuilder:
    """Checkpointed packed-index build controlled by a small progress manifest."""

    rules: AbstractRuleset
    output_dir: Path
    checkpoint_states: int = 10_000
    ordering_chunk_states: int = 1_000_000
    backend: str = "auto"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.checkpoint_states <= 0 or self.ordering_chunk_states <= 0:
            raise ValueError("checkpoint sizes must be positive")
        if self.backend not in {"auto", "python", "rust"}:
            raise ValueError("backend must be 'auto', 'python', or 'rust'")
        if self.rules.physical_state_upper_bound > int(UNREACHABLE_ORDINAL):
            raise ValueError("packed tablebase exceeds the uint32 ordinal contract")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._codec = PackedStateCodec(self.rules.load_cap_units)
        self._progress_path = self.output_dir / "build-progress.json"
        self._config_digest = digest_json(
            {
                "schema": PACKED_BUILD_SCHEMA,
                "ruleset_id": self.rules.ruleset_id,
                "action_values": self.rules.action_values,
                "bucket_seconds": self.rules.bucket_seconds,
                "load_cap_units": self.rules.load_cap_units,
                "failed_check_penalty_units": self.rules.failed_check_penalty_units,
                "revival_model": self.rules.revival_model_metadata,
            }
        )
        self._rust_kernel = self._load_rust_kernel()
        self._active_backend = "rust" if self._rust_kernel is not None else "python"
        if self._progress_path.exists():
            self._progress = json.loads(self._progress_path.read_text(encoding="utf-8"))
            if self._progress.get("config_digest") != self._config_digest:
                raise ValueError("checkpoint configuration does not match requested ruleset")
            if self._progress.get("schema_version") != PACKED_BUILD_SCHEMA:
                raise ValueError("unsupported packed tablebase checkpoint schema")
            self._progress.setdefault("execution_backends", [])
        else:
            self._progress = self._initialize_reachability()

    def _load_rust_kernel(self) -> Any | None:
        if self.backend == "python":
            return None
        try:
            module = importlib.import_module("abstract_solver_rs")
        except ImportError:
            if self.backend == "rust":
                raise RuntimeError(
                    "Rust backend requested but abstract_solver_rs is not installed; "
                    "see crates/docs/BUILD.md"
                )
            return None
        expected = "abstract-packed-parity-v2"
        if getattr(module, "PARITY_CONTRACT_VERSION", None) != expected:
            raise RuntimeError("Rust packed solver does not match the Python parity contract")
        return module

    @property
    def codec(self) -> PackedStateCodec:
        return self._codec

    @property
    def phase(self) -> str:
        return str(self._progress["phase"])

    @property
    def reachable_state_count(self) -> int | None:
        value = self._progress.get("reachable_state_count")
        return None if value is None else int(value)

    def _save_progress(self) -> None:
        _atomic_json(self._progress_path, self._progress)

    def _initialize_reachability(self) -> dict[str, Any]:
        physical = self.codec.state_count
        seen = np.memmap(
            self.output_dir / "reachability.bits",
            mode="w+",
            dtype=np.uint8,
            shape=((physical + 7) // 8,),
        )
        queue = np.memmap(
            self.output_dir / "reachability.queue.u32",
            mode="w+",
            dtype=np.uint32,
            shape=(physical,),
        )
        root = self.codec.encode(0, 0, 0, 0)
        seen[root >> 3] |= np.uint8(1 << (root & 7))
        queue[0] = root
        seen.flush()
        queue.flush()
        progress: dict[str, Any] = {
            "schema_version": PACKED_BUILD_SCHEMA,
            "config_digest": self._config_digest,
            "ruleset_id": self.rules.ruleset_id,
            "phase": "reachability",
            "queue_head": 0,
            "queue_tail": 1,
            "reachable_state_count": None,
            "solve_cursor": None,
            "states_solved": 0,
            "pure_saddle_states": 0,
            "mixed_lp_states": 0,
            "execution_backends": [],
        }
        _atomic_json(self._progress_path, progress)
        return progress

    def _open_reachability(self) -> tuple[np.memmap, np.memmap]:
        physical = self.codec.state_count
        seen = np.memmap(
            self.output_dir / "reachability.bits",
            mode="r+",
            dtype=np.uint8,
            shape=((physical + 7) // 8,),
        )
        queue = np.memmap(
            self.output_dir / "reachability.queue.u32",
            mode="r+",
            dtype=np.uint32,
            shape=(physical,),
        )
        return seen, queue

    @staticmethod
    def _rebuild_seen(seen: np.memmap, queue: np.memmap, tail: int, *, chunk: int = 1_000_000) -> None:
        # A crash may persist seen bits written after the last durable queue
        # tail. Rebuilding from the committed prefix makes resume transactional.
        seen[:] = 0
        for start in range(0, tail, chunk):
            indices = np.asarray(queue[start : min(start + chunk, tail)], dtype=np.uint64)
            byte_indices = indices >> 3
            masks = np.left_shift(np.uint8(1), (indices & 7).astype(np.uint8))
            np.bitwise_or.at(seen, byte_indices, masks)
        seen.flush()

    def enumerate_reachable(self, *, stop_after_dequeues: int | None = None) -> bool:
        """Advance packed bitset/queue reachability; return whether it finished."""

        if self.phase != "reachability":
            return True
        seen, queue = self._open_reachability()
        head = int(self._progress["queue_head"])
        tail = int(self._progress["queue_tail"])
        self._rebuild_seen(seen, queue, tail)
        completed_this_call = 0

        while head < tail:
            remaining = self.checkpoint_states
            if stop_after_dequeues is not None:
                remaining = min(remaining, stop_after_dequeues - completed_this_call)
            old_head = head
            if self._rust_kernel is not None:
                head, tail = self._rust_kernel.expand_reachability_chunk_rs(
                    queue,
                    seen,
                    head,
                    tail,
                    remaining,
                    self.rules.load_cap_units,
                    self.rules.action_size,
                    self.rules.failed_check_penalty_units,
                )
            else:
                stop = min(tail, head + remaining)
                while head < stop:
                    index = int(queue[head])
                    for child in packed_live_successors(
                        index,
                        self.rules,
                        codec=self.codec,
                    ):
                        byte_index = child >> 3
                        mask = 1 << (child & 7)
                        if int(seen[byte_index]) & mask:
                            continue
                        if tail >= self.codec.state_count:
                            raise RuntimeError(
                                "reachability queue exceeded the physical state domain"
                            )
                        seen[byte_index] = np.uint8(int(seen[byte_index]) | mask)
                        queue[tail] = child
                        tail += 1
                    head += 1
            completed_this_call += head - old_head
            seen.flush()
            queue.flush()
            self._progress["queue_head"] = head
            self._progress["queue_tail"] = tail
            if self._active_backend not in self._progress["execution_backends"]:
                self._progress["execution_backends"].append(self._active_backend)
            self._save_progress()
            should_stop = (
                stop_after_dequeues is not None
                and completed_this_call >= stop_after_dequeues
            )
            if should_stop:
                return False

        seen.flush()
        queue.flush()
        self._progress.update(
            {
                "phase": "ordering",
                "queue_head": head,
                "queue_tail": tail,
                "reachable_state_count": tail,
            }
        )
        self._save_progress()
        return True

    def prepare_storage(self) -> bool:
        """Counting-sort reachable indices by potential and create hot arrays."""

        if self.phase in {"solve", "complete"}:
            return True
        if self.phase != "ordering":
            return False

        reachable = int(self._progress["reachable_state_count"])
        _seen, queue = self._open_reachability()
        counts = np.zeros(self.codec.maximum_potential + 1, dtype=np.uint64)
        chunk = self.ordering_chunk_states
        for start in range(0, reachable, chunk):
            indices = np.asarray(queue[start : min(start + chunk, reachable)])
            counts += np.bincount(
                _potential_vector(indices, self.codec),
                minlength=counts.size,
            ).astype(np.uint64)

        offsets = np.zeros(counts.size + 1, dtype=np.uint64)
        offsets[1:] = np.cumsum(counts)
        positions = offsets[:-1].copy()
        order = _open_npy(
            self.output_dir / "state_index.npy",
            mode="w+",
            dtype="uint32",
            shape=(reachable,),
        )
        for start in range(0, reachable, chunk):
            indices = np.asarray(queue[start : min(start + chunk, reachable)])
            potentials = _potential_vector(indices, self.codec)
            permutation = np.argsort(potentials, kind="stable")
            sorted_indices = indices[permutation]
            sorted_potentials = potentials[permutation]
            boundaries = np.flatnonzero(np.diff(sorted_potentials)) + 1
            group_starts = np.r_[0, boundaries]
            group_ends = np.r_[boundaries, len(indices)]
            for group_start, group_end in zip(group_starts, group_ends):
                potential = int(sorted_potentials[group_start])
                destination = int(positions[potential])
                length = int(group_end - group_start)
                order[destination : destination + length] = sorted_indices[group_start:group_end]
                positions[potential] += length
        order.flush()
        if not np.array_equal(positions, offsets[1:]):
            raise RuntimeError("potential counting sort did not fill every reachable row")

        ordinal = _open_npy(
            self.output_dir / "ordinal_by_index.npy",
            mode="w+",
            dtype="uint32",
            shape=(self.codec.state_count,),
        )
        ordinal[:] = UNREACHABLE_ORDINAL
        for start in range(0, reachable, chunk):
            end = min(start + chunk, reachable)
            ordinal[np.asarray(order[start:end], dtype=np.int64)] = np.arange(
                start,
                end,
                dtype=np.uint32,
            )
        ordinal.flush()

        action_size = self.rules.action_size
        shapes = {
            "value": (reachable,),
            "drop_policy": (reachable, action_size),
            "check_policy": (reachable, action_size),
            "saddle_gap": (reachable,),
            "dropper_win_probability": (reachable,),
            "checker_win_probability": (reachable,),
        }
        for name, shape in shapes.items():
            dtype = _ARRAY_SPECS[name][0]
            array = _open_npy(
                self.output_dir / f"{name}.npy",
                mode="w+",
                dtype=dtype,
                shape=shape,
            )
            if np.issubdtype(np.dtype(dtype), np.floating):
                array[:] = np.nan
            else:
                array[:] = 0
            array.flush()

        self._progress.update(
            {
                "phase": "solve",
                "solve_cursor": reachable,
                "potential_counts": counts.tolist(),
                "potential_offsets": offsets.tolist(),
            }
        )
        self._save_progress()
        return True

    def _open_hot_arrays(self, *, mode: str) -> dict[str, np.memmap]:
        reachable = int(self._progress["reachable_state_count"])
        action_size = self.rules.action_size
        shapes = {
            "state_index": (reachable,),
            "ordinal_by_index": (self.codec.state_count,),
            "value": (reachable,),
            "drop_policy": (reachable, action_size),
            "check_policy": (reachable, action_size),
            "saddle_gap": (reachable,),
            "dropper_win_probability": (reachable,),
            "checker_win_probability": (reachable,),
        }
        return {
            name: _open_npy(
                self.output_dir / f"{name}.npy",
                mode=mode,
                dtype=_ARRAY_SPECS[name][0],
                shape=shape,
            )
            for name, shape in shapes.items()
        }

    def _child_ordinal(self, child: int, ordinal_by_index: np.ndarray) -> int:
        ordinal = int(ordinal_by_index[child])
        if ordinal == int(UNREACHABLE_ORDINAL):
            raise RuntimeError(f"live child {child} is missing from reachable closure")
        return ordinal

    def _cell_matrices(
        self,
        index: int,
        arrays: dict[str, np.memmap],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        checker_load, checker_ttd, dropper_load, dropper_ttd = self.codec.decode_unchecked(index)
        actions = self.rules.action_values
        size = len(actions)
        payoff = np.empty((size, size), dtype=np.float64)
        cell_dropper_win = np.empty_like(payoff)
        cell_checker_win = np.empty_like(payoff)
        value = arrays["value"]
        dropper_win = arrays["dropper_win_probability"]
        checker_win = arrays["checker_win_probability"]
        ordinal_by_index = arrays["ordinal_by_index"]

        dose = checker_load + self.rules.failed_check_penalty_units
        revive = self.rules.revival_probability(checker_ttd, dose)
        if revive > 0.0:
            failure_child = self.codec.encode_unchecked(
                dropper_load,
                dropper_ttd,
                0,
                checker_ttd + dose,
            )
            child_row = self._child_ordinal(failure_child, ordinal_by_index)
            failure_value = revive * -float(value[child_row]) + (1.0 - revive)
            failure_dropper_win = revive * float(checker_win[child_row]) + (1.0 - revive)
            failure_checker_win = revive * float(dropper_win[child_row])
        else:
            failure_value = 1.0
            failure_dropper_win = 1.0
            failure_checker_win = 0.0

        success_outcomes: dict[int, tuple[float, float, float]] = {}
        for squandered in actions:
            candidate_load = checker_load + squandered
            if candidate_load >= self.rules.load_cap_units:
                success_outcomes[squandered] = (1.0, 1.0, 0.0)
                continue
            child = self.codec.encode_unchecked(
                dropper_load,
                dropper_ttd,
                candidate_load,
                checker_ttd,
            )
            child_row = self._child_ordinal(child, ordinal_by_index)
            success_outcomes[squandered] = (
                -float(value[child_row]),
                float(checker_win[child_row]),
                float(dropper_win[child_row]),
            )

        for drop_index, drop in enumerate(actions):
            for check_index, check in enumerate(actions):
                if check < drop:
                    outcome = (
                        failure_value,
                        failure_dropper_win,
                        failure_checker_win,
                    )
                else:
                    outcome = success_outcomes[check - drop + 1]
                payoff[drop_index, check_index] = outcome[0]
                cell_dropper_win[drop_index, check_index] = outcome[1]
                cell_checker_win[drop_index, check_index] = outcome[2]

        return payoff, cell_dropper_win, cell_checker_win

    def _backup_state(
        self,
        index: int,
        arrays: dict[str, np.memmap],
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        payoff, cell_dropper_win, cell_checker_win = self._cell_matrices(
            index,
            arrays,
        )
        return solve_matrix(payoff), cell_dropper_win, cell_checker_win

    def _store_equilibrium(
        self,
        arrays: dict[str, np.memmap],
        row: int,
        equilibrium: Any,
        cell_dropper_win: np.ndarray,
        cell_checker_win: np.ndarray,
    ) -> None:
        arrays["value"][row] = equilibrium.value
        arrays["drop_policy"][row] = equilibrium.row_strategy
        arrays["check_policy"][row] = equilibrium.column_strategy
        arrays["saddle_gap"][row] = equilibrium.saddle_gap
        joint_policy = np.outer(
            equilibrium.row_strategy,
            equilibrium.column_strategy,
        )
        arrays["dropper_win_probability"][row] = float(
            np.sum(joint_policy * cell_dropper_win)
        )
        arrays["checker_win_probability"][row] = float(
            np.sum(joint_policy * cell_checker_win)
        )
        counter = (
            "pure_saddle_states"
            if equilibrium.solver_kind == "pure_saddle"
            else "mixed_lp_states"
        )
        self._progress[counter] = int(self._progress[counter]) + 1

    def _backup_rust_chunk(
        self,
        arrays: dict[str, np.memmap],
        start: int,
        cursor: int,
    ) -> None:
        assert self._rust_kernel is not None
        size = self.rules.action_size
        state_indices = np.asarray(arrays["state_index"][start:cursor])
        result = self._rust_kernel.backup_chunk_rs(
            state_indices,
            arrays["ordinal_by_index"],
            arrays["value"],
            arrays["dropper_win_probability"],
            arrays["checker_win_probability"],
            self.rules.load_cap_units,
            size,
            self.rules.failed_check_penalty_units,
            _rust_revival_model_code(self.rules),
            self.rules.revival_baseline,
            self.rules.dose_curve_exponent,
            self.rules.ttd_half_life_units,
            self.rules.ttd_curve_exponent,
            self.rules.referee_decay_per_death_dose,
            self.rules.referee_floor,
        )
        (
            pure_mask,
            pure_value,
            pure_drop_action,
            pure_check_action,
            pure_dropper_win,
            pure_checker_win,
            mixed_positions,
            mixed_payoff,
            mixed_dropper_win,
            mixed_checker_win,
        ) = (np.asarray(value) for value in result)

        for local in np.flatnonzero(pure_mask):
            row = start + int(local)
            drop_action = int(pure_drop_action[local])
            check_action = int(pure_check_action[local])
            arrays["value"][row] = pure_value[local]
            arrays["drop_policy"][row] = 0.0
            arrays["check_policy"][row] = 0.0
            arrays["drop_policy"][row, drop_action] = 1.0
            arrays["check_policy"][row, check_action] = 1.0
            arrays["saddle_gap"][row] = 0.0
            arrays["dropper_win_probability"][row] = pure_dropper_win[local]
            arrays["checker_win_probability"][row] = pure_checker_win[local]
            self._progress["pure_saddle_states"] = (
                int(self._progress["pure_saddle_states"]) + 1
            )

        mixed_count = len(mixed_positions)
        payoff_rows = mixed_payoff.reshape(mixed_count, size, size)
        dropper_rows = mixed_dropper_win.reshape(mixed_count, size, size)
        checker_rows = mixed_checker_win.reshape(mixed_count, size, size)
        for offset, local in enumerate(mixed_positions):
            row = start + int(local)
            equilibrium = solve_matrix(payoff_rows[offset])
            if equilibrium.solver_kind != "lp":
                raise RuntimeError(
                    "Rust/Python parity violation: mixed routing disagrees"
                )
            self._store_equilibrium(
                arrays,
                row,
                equilibrium,
                dropper_rows[offset],
                checker_rows[offset],
            )

    def solve(self, *, stop_after_chunks: int | None = None) -> bool:
        """Advance descending-potential backups; return whether solving finished."""

        if self.phase == "complete":
            return True
        if self.phase != "solve":
            return False
        arrays = self._open_hot_arrays(mode="r+")
        cursor = int(self._progress["solve_cursor"])
        chunks = 0

        while cursor > 0:
            offsets = np.asarray(self._progress["potential_offsets"], dtype=np.int64)
            potential = int(np.searchsorted(offsets, cursor - 1, side="right") - 1)
            layer_start = int(offsets[potential])
            start = max(layer_start, cursor - self.checkpoint_states)
            if self._rust_kernel is not None:
                self._backup_rust_chunk(arrays, start, cursor)
            else:
                for row in range(cursor - 1, start - 1, -1):
                    index = int(arrays["state_index"][row])
                    equilibrium, cell_dropper_win, cell_checker_win = self._backup_state(
                        index,
                        arrays,
                    )
                    self._store_equilibrium(
                        arrays,
                        row,
                        equilibrium,
                        cell_dropper_win,
                        cell_checker_win,
                    )

            for array in arrays.values():
                array.flush()
            cursor = start
            chunks += 1
            self._progress["solve_cursor"] = cursor
            self._progress["states_solved"] = int(self._progress["reachable_state_count"]) - cursor
            if self._active_backend not in self._progress["execution_backends"]:
                self._progress["execution_backends"].append(self._active_backend)
            self._save_progress()
            if stop_after_chunks is not None and chunks >= stop_after_chunks:
                return False

        self._finalize(arrays)
        return True

    def _finalize(self, arrays: dict[str, np.memmap]) -> None:
        persisted_policy_max_saddle_gap = self._validate_hot_arrays(arrays)
        source_root = Path(__file__).resolve().parent
        digest_inputs = [
            source_root / name
            for name in (
                "state.py",
                "rules.py",
                "matrix.py",
                "packed.py",
                "packed_tablebase.py",
            )
        ]
        if "rust" in self._progress["execution_backends"]:
            repository_root = source_root.parent
            digest_inputs.extend(
                (
                    repository_root / "crates" / "abstract_solver" / "Cargo.toml",
                    repository_root / "crates" / "abstract_solver" / "src" / "lib.rs",
                    repository_root / "Cargo.lock",
                )
            )
        code_config_digest = digest_files(
            digest_inputs,
            config={
                "ruleset_id": self.rules.ruleset_id,
                "root": (0, 0, 0, 0),
                "packed_build_config_digest": self._config_digest,
            },
        )
        reachable = int(self._progress["reachable_state_count"])
        array_manifest: dict[str, dict[str, Any]] = {}
        for name, array in arrays.items():
            path = self.output_dir / f"{name}.npy"
            array_manifest[name] = {
                "file": path.name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": sha256_file(path),
            }
        manifest = {
            "schema_version": PACKED_TABLEBASE_SCHEMA,
            "metadata": {
                "ruleset_id": self.rules.ruleset_id,
                "state_schema": self.rules.schema_version,
                "state_encoding": "mixed_radix_packed_uint32_v1",
                "state_field_names": list(self.rules.state_field_names),
                "action_values": list(self.rules.action_values),
                "action_seconds": [
                    self.rules.action_seconds(action)
                    for action in self.rules.action_values
                ],
                "timing_convention_id": TIMING_CONVENTION_ID,
                "bucket_seconds": self.rules.bucket_seconds,
                "load_cap_units": self.rules.load_cap_units,
                "load_cap_seconds": self.rules.load_cap_seconds,
                "failed_check_penalty_units": self.rules.failed_check_penalty_units,
                "revival_model": self.rules.revival_model_metadata,
                "root_state_index": 0,
                "reachable_state_count": reachable,
                "physical_state_upper_bound": self.rules.physical_state_upper_bound,
                "maximum_potential": self.codec.maximum_potential,
                "potential_counts": self._progress["potential_counts"],
                "solver": "packed_reachable_acyclic_bottom_up_dynamic_programming",
                "matrix_solver": {
                    "pure_saddle_states": int(self._progress["pure_saddle_states"]),
                    "mixed_lp_states": int(self._progress["mixed_lp_states"]),
                    "lp_shape": [self.rules.action_size, self.rules.action_size],
                    "policy_saddle_gap": 2e-7,
                    "primal_feasibility": 1e-9,
                    "dual_feasibility": 1e-9,
                    "ipm_optimality": 1e-10,
                },
                "state_ids": "derived_on_lookup_or_export_sha256",
                "checkpoint_states": self.checkpoint_states,
                "execution_backends": self._progress["execution_backends"],
                "persisted_policy_max_saddle_gap": persisted_policy_max_saddle_gap,
                "code_config_digest": code_config_digest,
            },
            "arrays": array_manifest,
        }
        _atomic_json(self.output_dir / "tablebase.json", manifest)
        self._progress["phase"] = "complete"
        self._progress["solve_cursor"] = 0
        self._progress["manifest_sha256"] = sha256_file(self.output_dir / "tablebase.json")
        self._save_progress()

    def _validate_hot_arrays(self, arrays: dict[str, np.memmap]) -> float:
        reachable = int(self._progress["reachable_state_count"])
        chunk = self.ordering_chunk_states
        for start in range(0, reachable, chunk):
            end = min(start + chunk, reachable)
            value = np.asarray(arrays["value"][start:end])
            saddle = np.asarray(arrays["saddle_gap"][start:end])
            drop_policy = np.asarray(arrays["drop_policy"][start:end])
            check_policy = np.asarray(arrays["check_policy"][start:end])
            dropper_win = np.asarray(arrays["dropper_win_probability"][start:end])
            checker_win = np.asarray(arrays["checker_win_probability"][start:end])
            if not all(
                np.all(np.isfinite(array))
                for array in (
                    value,
                    saddle,
                    drop_policy,
                    check_policy,
                    dropper_win,
                    checker_win,
                )
            ):
                raise RuntimeError("cannot finalize a tablebase with non-finite rows")
            if np.any(saddle < 0.0) or np.any(saddle > 2e-7):
                raise RuntimeError("packed tablebase contains an invalid saddle gap")
            if np.any(drop_policy < 0.0) or np.any(check_policy < 0.0):
                raise RuntimeError("packed tablebase contains negative policy mass")
            if not np.allclose(
                drop_policy.sum(axis=1, dtype=np.float64),
                1.0,
                atol=2e-7,
                rtol=0.0,
            ):
                raise RuntimeError("packed drop policies are not normalized")
            if not np.allclose(
                check_policy.sum(axis=1, dtype=np.float64),
                1.0,
                atol=2e-7,
                rtol=0.0,
            ):
                raise RuntimeError("packed check policies are not normalized")
            if (
                np.any(dropper_win < -2e-10)
                or np.any(dropper_win > 1.0 + 2e-10)
                or np.any(checker_win < -2e-10)
                or np.any(checker_win > 1.0 + 2e-10)
            ):
                raise RuntimeError("packed tablebase contains invalid win probabilities")
        maximum_gap = self._audit_persisted_policy_gaps(arrays)
        if maximum_gap > 2e-7:
            raise RuntimeError(
                f"persisted packed policy saddle gap is too large: {maximum_gap}"
            )
        return maximum_gap

    def _audit_persisted_policy_gaps(
        self,
        arrays: dict[str, np.memmap],
    ) -> float:
        if self._rust_kernel is None:
            return self._audit_persisted_policy_gaps_python(arrays)

        offsets = np.asarray(self._progress["potential_offsets"], dtype=np.int64)
        size = self.rules.action_size
        maximum_gap = 0.0
        for potential in range(len(offsets) - 2, -1, -1):
            layer_end = int(offsets[potential + 1])
            for start in range(int(offsets[potential]), layer_end, self.checkpoint_states):
                end = min(start + self.checkpoint_states, layer_end)
                result = self._rust_kernel.backup_chunk_rs(
                    np.asarray(arrays["state_index"][start:end]),
                    arrays["ordinal_by_index"],
                    arrays["value"],
                    arrays["dropper_win_probability"],
                    arrays["checker_win_probability"],
                    self.rules.load_cap_units,
                    size,
                    self.rules.failed_check_penalty_units,
                    _rust_revival_model_code(self.rules),
                    self.rules.revival_baseline,
                    self.rules.dose_curve_exponent,
                    self.rules.ttd_half_life_units,
                    self.rules.ttd_curve_exponent,
                    self.rules.referee_decay_per_death_dose,
                    self.rules.referee_floor,
                )
                mixed_positions = np.asarray(result[6], dtype=np.int64)
                if mixed_positions.size == 0:
                    continue
                payoff = np.asarray(result[7]).reshape(-1, size, size)
                rows = start + mixed_positions
                drop_policy = np.asarray(
                    arrays["drop_policy"][rows],
                    dtype=np.float64,
                )
                check_policy = np.asarray(
                    arrays["check_policy"][rows],
                    dtype=np.float64,
                )
                expected = np.einsum(
                    "mi,mij,mj->m",
                    drop_policy,
                    payoff,
                    check_policy,
                    optimize=True,
                )
                row_gain = np.maximum(
                    0.0,
                    np.einsum(
                        "mij,mj->mi",
                        payoff,
                        check_policy,
                        optimize=True,
                    ).max(axis=1)
                    - expected,
                )
                column_gain = np.maximum(
                    0.0,
                    expected
                    - np.einsum(
                        "mi,mij->mj",
                        drop_policy,
                        payoff,
                        optimize=True,
                    ).min(axis=1),
                )
                maximum_gap = max(
                    maximum_gap,
                    float(np.max(row_gain + column_gain, initial=0.0)),
                )
        return maximum_gap

    def _audit_persisted_policy_gaps_python(
        self,
        arrays: dict[str, np.memmap],
    ) -> float:
        maximum_gap = 0.0
        reachable = int(self._progress["reachable_state_count"])
        for row in range(reachable - 1, -1, -1):
            payoff, _dropper_win, _checker_win = self._cell_matrices(
                int(arrays["state_index"][row]),
                arrays,
            )
            _expected, _row_gain, _column_gain, gap = matrix_saddle_gap(
                payoff,
                arrays["drop_policy"][row],
                arrays["check_policy"][row],
            )
            maximum_gap = max(maximum_gap, gap)
        return maximum_gap

    def run(self) -> Path:
        if not self.enumerate_reachable():
            raise RuntimeError("reachability stopped before completion")
        if not self.prepare_storage():
            raise RuntimeError("storage preparation stopped before completion")
        if not self.solve():
            raise RuntimeError("tablebase solve stopped before completion")
        return self.output_dir / "tablebase.json"

    def verify_and_refresh_manifest(self) -> Path:
        """Re-run artifact gates and refresh hashes/metadata without backups."""

        if self.phase != "complete":
            raise RuntimeError("only a complete packed tablebase can be reverified")
        self._finalize(self._open_hot_arrays(mode="r+"))
        return self.output_dir / "tablebase.json"


@dataclass
class PackedTablebase:
    """Memory-mapped lookup facade; IDs are deliberately absent from disk rows."""

    artifact_dir: Path
    verify_hashes: bool = True

    def __post_init__(self) -> None:
        self.artifact_dir = Path(self.artifact_dir)
        manifest_path = self.artifact_dir / "tablebase.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != PACKED_TABLEBASE_SCHEMA:
            raise ValueError("unsupported packed tablebase artifact schema")
        metadata = self.manifest["metadata"]
        self.rules = self._rules_for_manifest(metadata)
        self.codec = PackedStateCodec(self.rules.load_cap_units)
        self.arrays: dict[str, np.ndarray] = {}
        for name, spec in self.manifest["arrays"].items():
            path = self.artifact_dir / spec["file"]
            if self.verify_hashes and sha256_file(path) != spec["sha256"]:
                raise ValueError(f"packed tablebase digest mismatch for {name}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != spec["shape"] or str(array.dtype) != spec["dtype"]:
                raise ValueError(f"packed tablebase shape/dtype mismatch for {name}")
            self.arrays[name] = array
        if "state_id" in self.arrays:
            raise ValueError("packed hot artifacts must not contain per-state IDs")

    @staticmethod
    def _rules_for_manifest(metadata: dict[str, Any]) -> AbstractRuleset:
        revival = metadata["revival_model"]
        return AbstractRuleset(
            ruleset_id=str(metadata["ruleset_id"]),
            action_values=tuple(int(value) for value in metadata["action_values"]),
            bucket_seconds=int(metadata["bucket_seconds"]),
            load_cap_units=int(metadata["load_cap_units"]),
            failed_check_penalty_units=int(metadata["failed_check_penalty_units"]),
            revival_model_kind=str(revival["kind"]),
            revival_baseline=float(revival["baseline"]),
            dose_curve_exponent=float(revival.get("dose_curve_exponent", 3.0)),
            ttd_half_life_units=float(revival["ttd_half_life_units"]),
            ttd_curve_exponent=float(revival["ttd_curve_exponent"]),
            referee_decay_per_death_dose=float(
                revival.get("referee_decay_per_death_dose", 1.0)
            ),
            referee_floor=float(revival.get("referee_floor", 1.0)),
        )

    def lookup(self, state: AbstractState | tuple[int, int, int, int] | int) -> dict[str, Any]:
        if isinstance(state, AbstractState):
            fields = self.rules.state_fields(state)
            index = self.codec.encode(*fields)
            state_object = state
        elif isinstance(state, tuple):
            if len(state) != 4:
                raise ValueError("packed state tuple must have four fields")
            fields = tuple(int(value) for value in state)
            index = self.codec.encode(*fields)
            state_object = AbstractState(*fields)
        else:
            index = int(state)
            fields = self.codec.decode(index)
            state_object = AbstractState(*fields)
        row = int(self.arrays["ordinal_by_index"][index])
        if row == int(UNREACHABLE_ORDINAL):
            raise LookupError(f"state {fields} is outside the root's reachable closure")
        if int(self.arrays["state_index"][row]) != index:
            raise ValueError("packed ordinal index is internally inconsistent")
        return {
            "state": fields,
            "state_index": index,
            "state_id": state_id(state_object, self.rules),
            "value": float(self.arrays["value"][row]),
            "drop_policy": np.asarray(self.arrays["drop_policy"][row]).copy(),
            "check_policy": np.asarray(self.arrays["check_policy"][row]).copy(),
            "saddle_gap": float(self.arrays["saddle_gap"][row]),
            "dropper_win_probability": float(
                self.arrays["dropper_win_probability"][row]
            ),
            "checker_win_probability": float(
                self.arrays["checker_win_probability"][row]
            ),
        }

    def export_rows(self, indices: Iterator[int]) -> Iterator[dict[str, Any]]:
        for index in indices:
            yield self.lookup(index)


def build_packed_tablebase(
    rules: AbstractRuleset,
    output_dir: str | Path,
    *,
    checkpoint_states: int = 10_000,
    backend: str = "auto",
) -> Path:
    """Build or resume a complete packed tablebase and return its manifest."""

    return PackedTablebaseBuilder(
        rules=rules,
        output_dir=Path(output_dir),
        checkpoint_states=checkpoint_states,
        backend=backend,
    ).run()

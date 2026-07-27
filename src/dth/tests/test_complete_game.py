from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from dth.exact_agent import ExactDTHAgent, ExactAgentDeadlineError
from dth.reachability import (
    CensusCorruptionError,
    CensusError,
    RankLayerSolver,
    ReachabilityCensus,
    bellman_recertify,
    decode_state_id,
    encode_state_id,
    failure_dead_reachability_bitsets,
    propagate_bellman_interval,
    reconstruct_policy_from_certified_children,
)
from dth.solver import (
    CHECKER_ACTIONS,
    DROPPER_ACTIONS,
    SADDLE_GAP_TOLERANCE,
    ValueInterval,
    canonical_state_id,
    complete_game_dependencies,
    continuation_class_values,
    damage_rank,
    failure_dead_quotient,
    payoff,
    payoff_from_transition_classes,
    reconstruct_transition_class_matrix,
    solve_full_support_structured_matrix,
    solve_matrix,
    transition,
    validate_action,
    validate_live_state,
)
from dth.tablebase import (
    CertifiedTablebase,
    TablebaseCorruptionError,
    TablebaseSchemaError,
)


NEAR_TERMINAL = (239, 241, 299, 300)


def _build_census(
    tablebase: CertifiedTablebase,
    roots: list[tuple[int, int, int, int]],
) -> ReachabilityCensus:
    census = ReachabilityCensus(tablebase, roots)
    run = census.run(
        max_expansions=20_000,
        max_states=50_000,
        max_seconds=30.0,
    )
    assert run.stop_reason == "complete"
    return census


def test_damage_rank_strictly_increases_for_every_live_transition() -> None:
    state = (100, 100, 80, 80)
    for drop in DROPPER_ACTIONS:
        for check in CHECKER_ACTIONS:
            for _, child in transition(state, drop, check):
                if isinstance(child, tuple):
                    assert damage_rank(child) > damage_rank(state)


@pytest.mark.parametrize("checker_st", [0, 238, 239, 240, 298, 299])
@pytest.mark.parametrize("checker_ttd", [0, 239, 240, 299, 300])
def test_damage_rank_boundary_formulae_remain_strict(
    checker_st: int,
    checker_ttd: int,
) -> None:
    state = (checker_st, checker_ttd, 17, 23)
    for drop in DROPPER_ACTIONS:
        for check in CHECKER_ACTIONS:
            deltas = [
                damage_rank(child) - damage_rank(state)
                for _, child in transition(state, drop, check)
                if isinstance(child, tuple)
            ]
            assert all(delta > 0 for delta in deltas)
            if check >= drop and deltas:
                assert deltas == [check - drop + 1]
            if check < drop and deltas:
                assert deltas == [60]


@pytest.mark.parametrize(
    ("state", "horizon"),
    [
        ((0, 0, 0, 0), 1),
        ((179, 61, 0, 0), 2),
        ((239, 241, 299, 300), 3),
        ((200, 40, 240, 0), 2),
    ],
)
def test_transition_class_matrix_parity_is_within_1e12(
    state: tuple[int, int, int, int],
    horizon: int,
) -> None:
    np.testing.assert_allclose(
        payoff_from_transition_classes(state, horizon),
        payoff(state, horizon),
        rtol=0.0,
        atol=1e-12,
    )


def test_raw_state_ids_round_trip_without_quotient_loss() -> None:
    states = [
        (0, 0, 0, 0),
        (299, 300, 299, 300),
        (240, 0, 240, 0),
        (17, 299, 298, 1),
    ]
    for state in states:
        assert decode_state_id(encode_state_id(state)) == state


def test_failure_dead_quotient_erases_only_dead_ttd_and_preserves_classes() -> None:
    left = (240, 0, 241, 300)
    right = (240, 287, 241, 11)
    assert failure_dead_quotient(left) == (60, 59)
    assert failure_dead_quotient(right) == (60, 59)
    assert canonical_state_id(left) == canonical_state_id(right)

    def quotient_value(child: tuple[int, int, int, int]) -> float:
        checker_remaining, dropper_remaining = failure_dead_quotient(child)  # type: ignore[misc]
        return (checker_remaining - dropper_remaining) / 60.0

    left_success, left_failed = continuation_class_values(left, quotient_value)
    right_success, right_failed = continuation_class_values(right, quotient_value)
    np.testing.assert_allclose(left_success, right_success, rtol=0.0, atol=0.0)
    assert left_failed == right_failed


def test_checker_turn_bitsets_formalize_240_closure_exactly() -> None:
    proof = failure_dead_reachability_bitsets([(240, 0, 240, 0)])
    assert proof["exact_equivalence_classes"] == 3541
    assert len(proof["checker_turn_bitsets_hex"]) == 60
    assert sum(proof["classes_by_damage_rank"].values()) == 3541


def test_unified_census_is_policy_free_and_matches_bitset_count(tmp_path) -> None:
    with CertifiedTablebase(tmp_path / "unified.sqlite") as tablebase:
        census = _build_census(tablebase, [(240, 0, 240, 0)])
        report = census.report(
            census.run(max_expansions=0, max_states=50_000, max_seconds=1.0)
        )
        assert report["completion_status"] == "complete"
        assert report["unique_reachable_states"] == 3541
        assert report["failure_dead_bitset_proof"]["exact_equivalence_classes"] == 3541
        assert tablebase.verify()["cached_root_policies"] == 0
        tables = {
            row[0]
            for row in tablebase.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"values", "states", "rank_layers", "policy_cache"} <= tables
        indexes = {
            row[0]
            for row in tablebase.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
        assert "ix_frontier" in indexes


def test_rank_solver_commits_values_and_queue_atomically_then_caches_one_root_policy(
    tmp_path,
) -> None:
    with CertifiedTablebase(tmp_path / "exact.sqlite") as tablebase:
        census = _build_census(tablebase, [NEAR_TERMINAL])
        result = RankLayerSolver(census).run(
            max_new_solutions=100,
            max_seconds=30.0,
            batch_size=8,
            workers=2,
        )
        assert result["completion_status"] == "complete"
        assert result["queue"] == {"pending": 0, "in_progress": 0, "committed": 61}
        verification = tablebase.verify(full=True)
        assert verification["exact_values"] == 61
        assert verification["cached_root_policies"] == 0
        solution = reconstruct_policy_from_certified_children(
            NEAR_TERMINAL, tablebase, cache=True
        )
        assert solution.saddle_gap <= SADDLE_GAP_TOLERANCE
        assert tablebase.verify(full=True)["cached_root_policies"] == 1
        assert bellman_recertify(NEAR_TERMINAL, tablebase).value == pytest.approx(
            solution.value, abs=1e-12
        )


def test_interruption_resume_has_no_value_queue_crash_window(tmp_path) -> None:
    path = tmp_path / "resume.sqlite"
    with CertifiedTablebase(path) as tablebase:
        census = _build_census(tablebase, [NEAR_TERMINAL])
        partial = RankLayerSolver(census).run(
            max_new_solutions=1,
            max_seconds=30.0,
            batch_size=8,
        )
        assert partial["queue"] == {
            "pending": 60,
            "in_progress": 0,
            "committed": 1,
        }
        census.verify(full=True)

    with CertifiedTablebase(path) as tablebase:
        census = ReachabilityCensus(tablebase, [NEAR_TERMINAL])
        RankLayerSolver(census).run(
            max_new_solutions=100,
            max_seconds=30.0,
            batch_size=16,
        )
        resumed = tablebase.deterministic_snapshot()
        census.verify(full=True)

    with CertifiedTablebase(tmp_path / "fresh.sqlite") as tablebase:
        census = _build_census(tablebase, [NEAR_TERMINAL])
        RankLayerSolver(census).run(
            max_new_solutions=100,
            max_seconds=30.0,
            batch_size=16,
        )
        assert tablebase.deterministic_snapshot() == resumed


def test_unresolved_children_produce_global_interval_then_exact_value_refines_it(
    tmp_path,
) -> None:
    with CertifiedTablebase(tmp_path / "interval.sqlite") as tablebase:
        initial = propagate_bellman_interval(
            NEAR_TERMINAL, tablebase, persist=True
        )
        assert -1.0 <= initial.lower_bound <= initial.upper_bound <= 1.0
        census = _build_census(tablebase, [NEAR_TERMINAL])
        RankLayerSolver(census).run(
            max_new_solutions=100,
            max_seconds=30.0,
            batch_size=16,
        )
        exact = tablebase.get_complete_value(NEAR_TERMINAL)
        assert exact is not None and exact.exact and exact.value is not None
        assert initial.lower_bound <= exact.value <= initial.upper_bound
        assert exact.lower_bound >= initial.lower_bound - 1e-12
        assert exact.upper_bound <= initial.upper_bound + 1e-12


def test_structured_full_support_solve_matches_highs_or_fails_to_oracle() -> None:
    rng = np.random.default_rng(20260724)
    candidate = None
    structured = None
    for _ in range(500):
        success = rng.uniform(-0.9, 0.9, size=60)
        matrix = reconstruct_transition_class_matrix(success, float(rng.uniform(-0.9, 0.9)))
        try:
            structured = solve_full_support_structured_matrix(matrix)
        except RuntimeError:
            continue
        candidate = matrix
        break
    assert candidate is not None and structured is not None
    oracle = solve_matrix(candidate)
    assert structured[0] == pytest.approx(oracle[0], abs=1e-8)
    assert (
        np.max(candidate @ structured[2])
        - np.min(candidate.T @ structured[1])
        <= SADDLE_GAP_TOLERANCE
    )


def test_exact_agent_requires_a_bound_and_uses_only_certified_fallback(tmp_path) -> None:
    root = (299, 300, 299, 300)
    with CertifiedTablebase(tmp_path / "agent.sqlite") as tablebase:
        agent = ExactDTHAgent(tablebase)
        fallback = agent.prepare_finite_fallback(root, 1)
        assert fallback.scope == "finite-horizon-exact"
        with pytest.raises(ValueError):
            agent.evaluate(root)
        result = agent.evaluate(
            root,
            deadline_seconds=0.0,
            allow_expansion=False,
        )
        assert result.cache_provenance == "finite-cache"
    with CertifiedTablebase(tmp_path / "empty.sqlite") as tablebase:
        with pytest.raises(ExactAgentDeadlineError):
            ExactDTHAgent(tablebase).evaluate(
                root, deadline_seconds=0.0, allow_expansion=False
            )


def test_schema_and_corruption_firewalls_fail_closed(tmp_path) -> None:
    legacy = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(legacy)
    connection.execute("CREATE TABLE metadata(key TEXT, value TEXT)")
    connection.commit()
    connection.close()
    with pytest.raises(TablebaseSchemaError):
        CertifiedTablebase(legacy)

    path = tmp_path / "corrupt.sqlite"
    with CertifiedTablebase(path) as tablebase:
        census = _build_census(tablebase, [NEAR_TERMINAL])
        RankLayerSolver(census).run(
            max_new_solutions=1,
            max_seconds=30.0,
            batch_size=1,
        )
        tablebase.connection.execute(
            'UPDATE "values" SET certificate_sha256=zeroblob(32)'
        )
        tablebase.connection.commit()
        with pytest.raises(TablebaseCorruptionError):
            tablebase.verify(full=True)
    with pytest.raises(TablebaseCorruptionError):
        CertifiedTablebase(path)


def test_census_root_manifest_and_rank_corruption_fail_closed(tmp_path) -> None:
    path = tmp_path / "manifest.sqlite"
    with CertifiedTablebase(path) as tablebase:
        census = ReachabilityCensus(tablebase, [NEAR_TERMINAL])
        with pytest.raises(CensusError):
            census.claim_next_rank(limit=1)
        tablebase.connection.execute(
            "UPDATE states SET damage_rank=damage_rank+1"
        )
        with pytest.raises(CensusCorruptionError):
            census.verify(full=True)
    with CertifiedTablebase(path, verify_on_open=False) as tablebase:
        with pytest.raises(TablebaseSchemaError):
            ReachabilityCensus(tablebase, [(0, 0, 0, 0)])


def test_public_validation_and_dependency_contracts() -> None:
    with pytest.raises(ValueError):
        validate_live_state((0, 0, 0))
    with pytest.raises(ValueError):
        validate_action(0, role="dropper")
    with pytest.raises(ValueError):
        validate_action(61, role="checker")
    assert len(complete_game_dependencies(NEAR_TERMINAL)) == 60
    assert ValueInterval(-1.0, 1.0).midpoint == 0.0

from types import SimpleNamespace

import numpy as np
import pytest

from stl.commands.gen0_eval import (
    REQUIRED_RULER_SOURCES,
    _assert_isolated,
    _gate,
    _policy_report,
    _source_value_metrics,
)
from stl.learning.reachable import split_reachable_candidates
from stl.learning.certificates import ExactPolicyCertificate
from stl.learning.replay import TargetKind, exact_state_hash, reconstruct_game
from stl.solver.exact import ExactSolveResult, UtilityBreakdown


def _aggregate(*, maximum: float | None, median: float | None = None) -> dict:
    return {
        "count": 0 if maximum is None else 1,
        "mean": maximum,
        "median": median,
        "maximum": maximum,
    }


def _passing_policy_metrics() -> dict:
    return {
        "saddle_gap": _aggregate(maximum=0.04),
        "maximum_illegal_mass": 0.0,
        "label_value_recompute_error": _aggregate(maximum=0.0),
        "cutoff_recompute_error": _aggregate(maximum=0.0),
        "strict_unique_dropper_tv": _aggregate(maximum=None),
        "strict_unique_checker_tv": _aggregate(maximum=None),
    }


def _gate_result(value_sources: dict, policy: dict) -> dict:
    return _gate(
        value_sources=value_sources,
        policy=policy,
        reload_max_abs_difference=0.0,
        tablebase_mse_max=0.01,
        tablebase_interior_mse_max=0.05,
        exact_saddle_gap_max=0.05,
        unique_policy_tv_median_max=0.15,
        exact_cutoff_max=0.5,
    )


def test_value_metrics_are_reported_per_source() -> None:
    records = [
        SimpleNamespace(source="tablebase", value=1.0, cutoff_probability=0.0),
        SimpleNamespace(source="tablebase", value=-1.0, cutoff_probability=0.0),
        SimpleNamespace(source="exact_horizon_2", value=0.0, cutoff_probability=0.5),
    ]

    sources, mse, rmse, brier = _source_value_metrics(
        records, np.asarray([0.5, -0.5, 0.25])
    )

    assert sources["tablebase"]["mse"] == pytest.approx(0.25)
    assert sources["exact_horizon_2"]["max_cutoff_probability"] == 0.5
    assert mse == pytest.approx(0.1875)
    assert rmse == pytest.approx(np.sqrt(0.1875))
    assert brier == pytest.approx(0.0625)


def test_gate_passes_only_when_every_external_ruler_contract_passes() -> None:
    value_sources = {
        source: {
            "count": 1,
            "mse": 0.0,
            "max_cutoff_probability": 0.5 if source.startswith("exact_") else 0.0,
        }
        for source in REQUIRED_RULER_SOURCES
    }

    passing = _gate_result(value_sources, _passing_policy_metrics())
    assert passing["passed"] is True

    value_sources["tablebase"]["mse"] = 0.02
    policy = _passing_policy_metrics()
    policy["saddle_gap"] = _aggregate(maximum=0.06)
    failing = _gate_result(value_sources, policy)

    assert failing["passed"] is False
    assert "tablebase_mse" in failing["failures"]
    assert "exact_policy_saddle_gap" in failing["failures"]


def test_gate_permits_only_float32_roundoff_in_recomputed_exact_labels() -> None:
    value_sources = {
        source: {
            "count": 1,
            "mse": 0.0,
            "max_cutoff_probability": 0.5 if source.startswith("exact_") else 0.0,
        }
        for source in REQUIRED_RULER_SOURCES
    }
    policy = _passing_policy_metrics()
    policy["label_value_recompute_error"] = _aggregate(maximum=5e-7)
    policy["cutoff_recompute_error"] = _aggregate(maximum=5e-7)
    assert _gate_result(value_sources, policy)["passed"] is True

    policy["label_value_recompute_error"] = _aggregate(maximum=2e-6)
    failing = _gate_result(value_sources, policy)
    assert failing["passed"] is False
    assert "exact_label_recompute_error" in failing["failures"]


def test_train_ruler_isolation_rejects_an_exact_state_overlap() -> None:
    split = split_reachable_candidates(smoke=True, split_seed=20260714)
    train_record = SimpleNamespace(
        exact_state=split.train[0].exact_state,
        episode_id="train",
        features=np.asarray([1.0, 2.0]),
    )
    ruler_record = SimpleNamespace(
        exact_state=split.ruler[0].exact_state,
        episode_id="ruler",
        features=np.asarray([3.0, 4.0]),
    )
    assert _assert_isolated([train_record], [ruler_record]) == {
        "exact_states": 0,
        "episodes": 0,
        "features": 0,
    }

    ruler_record.exact_state = train_record.exact_state
    with pytest.raises(ValueError, match="exact_states"):
        _assert_isolated([train_record], [ruler_record])


def test_policy_report_re_solves_exact_matrix_and_measures_saddle_gap() -> None:
    split = split_reachable_candidates(smoke=True, split_seed=20260714)
    state = next(
        item.exact_state for item in split.train if not item.exact_state.game_over
    )
    game = reconstruct_game(state)
    policy = np.zeros(62, dtype=np.float64)
    policy[1:3] = 0.5
    legal = np.ones(62, dtype=np.bool_)
    record = SimpleNamespace(
        exact_state=state,
        target_kind=TargetKind.EXACT_VALUE,
        dropper_dist=policy,
        checker_dist=policy,
        dropper_legal_mask=legal,
        checker_legal_mask=legal,
        episode_id="exact",
        half_round_index=0,
        source="exact_horizon_2",
        value_horizon_half_rounds=2,
        value=0.0,
        cutoff_probability=0.0,
    )

    def predict_game(_game, *, horizon):
        assert horizon == 2
        return 0.0, policy.copy(), policy.copy()

    def exact_solver(_game, horizon, _config):
        return ExactSolveResult(
            dropper_strategy=np.asarray([0.5, 0.5]),
            checker_strategy=np.asarray([0.5, 0.5]),
            value_for_hal=0.0,
            breakdown=UtilityBreakdown(0.0, 0.5, 0.5, 0.0),
            unresolved_probability=0.0,
            half_round_horizon=horizon,
            drop_actions=(1, 2),
            check_actions=(1, 2),
            payoff_for_hal=np.asarray([[1.0, -1.0], [-1.0, 1.0]]),
        )

    report = _policy_report(
        [record], predict_game, exact_solver=exact_solver, uniqueness_tolerance=1e-9
    )

    assert report["active_rows"] == 1
    assert report["saddle_gap"]["maximum"] == pytest.approx(0.0)
    assert report["maximum_illegal_mass"] == 0.0
    assert report["label_value_recompute_error"]["maximum"] == 0.0
    assert report["cutoff_recompute_error"]["maximum"] == 0.0
    assert game.game_over is False


def test_policy_report_uses_matching_certificate_without_an_exact_resolve() -> None:
    split = split_reachable_candidates(smoke=True, split_seed=20260714)
    state = next(
        item.exact_state for item in split.train if not item.exact_state.game_over
    )
    policy = np.zeros(62, dtype=np.float64)
    policy[1:3] = 0.5
    record = SimpleNamespace(
        exact_state=state,
        target_kind=TargetKind.EXACT_VALUE,
        dropper_dist=policy,
        checker_dist=policy,
        dropper_legal_mask=np.ones(62, dtype=np.bool_),
        checker_legal_mask=np.ones(62, dtype=np.bool_),
        episode_id="certified",
        half_round_index=0,
        source="exact_horizon_2",
        value_horizon_half_rounds=2,
        value=0.0,
        cutoff_probability=0.0,
        search_config_digest="a" * 64,
    )
    certificate = ExactPolicyCertificate(
        state_hash=exact_state_hash(state),
        search_config_digest="a" * 64,
        horizon=2,
        drop_actions=(1, 2),
        check_actions=(1, 2),
        payoff_for_hal=np.asarray([[1.0, -1.0], [-1.0, 1.0]]),
        dropper_strategy=np.asarray([0.5, 0.5]),
        checker_strategy=np.asarray([0.5, 0.5]),
        value_for_hal=0.0,
        unresolved_probability=0.0,
    )

    def forbidden_solver(*_args, **_kwargs):
        raise AssertionError("certificate-backed report re-solved the state")

    report = _policy_report(
        [record],
        lambda _game, *, horizon: (0.0, policy.copy(), policy.copy()),
        exact_solver=forbidden_solver,
        certificates={certificate.state_hash: certificate},
        uniqueness_tolerance=1e-9,
    )
    assert report["saddle_gap"]["maximum"] == pytest.approx(0.0)
    assert report["certificate_spot_checks"]["count"] == 0

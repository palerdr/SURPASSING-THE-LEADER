import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.dth_env import DTHEnv
from environment.opponents.factory import create_scripted_opponent
from environment.route_stages import current_route_stage_flags
from scripts.search_opening_traces import (
    candidate_seconds_for_turn,
    compact_state_key,
    current_role,
    load_seed_prefixes,
    rank_branch,
    replay_prefix,
    search_many_seeds,
    search_seed,
)
from training.bridge_traces import BridgeTraceSpec, load_trace_file, save_trace_file


def test_candidate_seconds_respect_current_mask():
    env = DTHEnv(opponent=create_scripted_opponent("bridge_pressure"), agent_role="baku", seed=1990)
    env.reset()

    role = current_role(env)
    seconds = candidate_seconds_for_turn(env, role, "opening_small")
    mask = env.action_masks()

    assert seconds
    assert all(mask[second - 1] for second in seconds)


def test_replay_prefix_reaches_known_round7_seed_trace():
    env, _obs, terminated, truncated = replay_prefix(
        role="baku",
        opponent_name="bridge_pressure",
        opponent_model_path=None,
        seed=1990,
        prefix_actions=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3),
    )

    assert terminated is False
    assert truncated is False
    assert current_route_stage_flags(env.game)["round7_pressure"] is True


def test_search_seed_finds_known_opening_positive_trace():
    results, _near_misses = search_seed(
        role="baku",
        opponent_name="bridge_pressure",
        opponent_model_path=None,
        seed=1990,
        beam_width=64,
        max_depth=12,
        target_stage="round7_pressure",
        candidate_set_name="opening_small",
        max_results=2,
    )

    assert results
    env, _obs, terminated, truncated = replay_prefix(
        role="baku",
        opponent_name="bridge_pressure",
        opponent_model_path=None,
        seed=1990,
        prefix_actions=results[0].trace.actions,
    )

    assert terminated is False
    assert truncated is False
    assert current_route_stage_flags(env.game)["round7_pressure"] is True


def test_search_seed_is_deterministic_for_fixed_seed_and_beam():
    results_a, _ = search_seed(
        role="baku",
        opponent_name="bridge_pressure",
        opponent_model_path=None,
        seed=1990,
        beam_width=64,
        max_depth=12,
        target_stage="round7_pressure",
        candidate_set_name="opening_small",
        max_results=2,
    )
    results_b, _ = search_seed(
        role="baku",
        opponent_name="bridge_pressure",
        opponent_model_path=None,
        seed=1990,
        beam_width=64,
        max_depth=12,
        target_stage="round7_pressure",
        candidate_set_name="opening_small",
        max_results=2,
    )

    assert [result.trace for result in results_a] == [result.trace for result in results_b]


def test_compact_state_key_matches_for_same_replayed_prefix():
    env_a, _obs_a, _term_a, _trunc_a = replay_prefix(
        role="baku",
        opponent_name="bridge_pressure",
        opponent_model_path=None,
        seed=1990,
        prefix_actions=(1, 1, 1),
    )
    env_b, _obs_b, _term_b, _trunc_b = replay_prefix(
        role="baku",
        opponent_name="bridge_pressure",
        opponent_model_path=None,
        seed=1990,
        prefix_actions=(1, 1, 1),
    )

    assert compact_state_key(env_a) == compact_state_key(env_b)


def test_rank_branch_prefers_reached_target_state():
    success_env, _obs, _term, _trunc = replay_prefix(
        role="baku",
        opponent_name="bridge_pressure",
        opponent_model_path=None,
        seed=1990,
        prefix_actions=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3),
    )
    early_env, _obs2, _term2, _trunc2 = replay_prefix(
        role="baku",
        opponent_name="bridge_pressure",
        opponent_model_path=None,
        seed=1990,
        prefix_actions=(1, 1),
    )

    assert rank_branch(success_env, 12, "round7_pressure") > rank_branch(early_env, 2, "round7_pressure")


def test_searched_trace_file_round_trip(tmp_path):
    results, _near_misses = search_seed(
        role="baku",
        opponent_name="bridge_pressure",
        opponent_model_path=None,
        seed=1990,
        beam_width=64,
        max_depth=12,
        target_stage="round7_pressure",
        candidate_set_name="opening_small",
        max_results=1,
    )
    path = tmp_path / "search_opening.json"
    specs = tuple(result.trace for result in results)

    save_trace_file(str(path), specs)
    loaded = load_trace_file(str(path))

    assert loaded == specs


def test_load_seed_prefixes_uses_opening_traces_only(tmp_path):
    path = tmp_path / "seed_traces.json"
    save_trace_file(
        str(path),
        (
            BridgeTraceSpec(
                name="opening_seed",
                agent_role="baku",
                opponent_name="bridge_pressure",
                opponent_model_path=None,
                scenario_name="opening",
                actions=(1, 1, 1, 1, 1, 1, 1, 3, 1),
                seed=1990,
            ),
            BridgeTraceSpec(
                name="round7_seed",
                agent_role="baku",
                opponent_name="bridge_pressure",
                opponent_model_path=None,
                scenario_name="round7_pressure",
                actions=(25, 1),
                seed=42,
            ),
        ),
    )

    prefixes = load_seed_prefixes([str(path)], prefix_turns=8)

    assert prefixes
    assert all(len(prefix) == 8 for prefix in prefixes)


def test_search_many_seeds_respects_per_seed_cap():
    traces, _near_misses = search_many_seeds(
        role="baku",
        opponent_name="bridge_pressure",
        opponent_model_path=None,
        seed_start=0,
        seed_count=64,
        beam_width=64,
        max_depth=12,
        target_stage="round7_pressure",
        candidate_set_name="opening_small",
        max_traces=8,
        max_traces_per_seed=2,
        max_traces_per_family=8,
        family_prefix_turns=8,
        seed_prefixes=(),
        verbose=False,
    )

    from collections import Counter

    counts = Counter(trace.seed for trace in traces)
    assert all(count <= 2 for count in counts.values())

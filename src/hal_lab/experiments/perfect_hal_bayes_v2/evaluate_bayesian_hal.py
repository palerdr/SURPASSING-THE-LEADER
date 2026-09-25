"""Compare Bayesian Perfect Hal with v1 and equilibrium without replay bias.

Human logs support causal prediction and one-step deviation values at logged
states. We measure full-game wins in fresh paired simulations. Human-fitted
simulations remain model-dependent estimates, not human treatment effects.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from arena.dth_adapter import project_to_dth_state
from arena.match import play_match_game
from arena.policies import bayesian_hal, perfect_hal
from arena.policies.bayesian_hal import BayesianHalConfig, BayesianHalOpponentModel
from arena.policies.perfect_hal import PerfectHalOpponentModel, PerfectHalPolicyProvider
from dth.agent import CompleteDTHAgent
from hal_lab.harness.baselines import ExactProvider, FittedHumanOpponent, ResetMemoryProvider, UniformProvider
from hal_lab.harness.human_data import fit_priors, human_observation, split_games
from hal_lab.harness.opponent_league import ReactiveDTHOpponent, SUPPORTED_FAMILIES
from hal_lab.harness.stats import cluster_interval, hard_response

CONTROLLERS = ("exact", "legacy", "bayes", "bayes_population", "bayes_reset")


def evaluate_logs(partitions, agent, bayes_config, protocol, split):
    player_results = []
    for player, parts in partitions.items():
        if not parts[split]:
            continue
        priors = fit_priors(partitions, protocol["prior_pseudocount"], excluded_player=player)
        models = {"legacy": PerfectHalOpponentModel(), "bayes": BayesianHalOpponentModel(bayes=bayes_config),
                  "bayes_population": BayesianHalOpponentModel(bayes=bayes_config, role_priors=priors),
                  "bayes_reset": BayesianHalOpponentModel(bayes=bayes_config)}
        scores = {name: [] for name in CONTROLLERS}
        discarded = 0
        phases = ("train", "validation") if split == "validation" else ("train", "validation", "test")
        for phase in phases:
            for game in parts[phase]:
                for move_index, move in enumerate(game["public_history"]):
                    observation = human_observation(move)
                    if observation is None:
                        discarded += 1
                        # A skipped reveal breaks the pattern sequence. Resume with
                        # fresh evidence instead of inventing an adjacent action.
                        for model in models.values():
                            model.reset()
                        continue
                    decision, role, action, own = observation
                    models["bayes_reset"].reset()
                    forecasts = {name: model.predict(role, state_regime=model.state_regime(decision),
                        game_index=game["ordinal"], game_decision_index=move_index) for name, model in models.items()}
                    if phase == split:
                        stage = agent.stage_game(project_to_dth_state(decision))
                        payoff = stage.matrix if decision.role == "dropper" else -stage.matrix.T
                        equilibrium = stage.drop_policy if decision.role == "dropper" else stage.check_policy
                        opponent_equilibrium = stage.check_policy if decision.role == "dropper" else stage.drop_policy
                        for name in CONTROLLERS:
                            prediction = opponent_equilibrium if name == "exact" else forecasts[name].policy
                            policy = equilibrium if name == "exact" else hard_response(payoff @ prediction)
                            scores[name].append({
                                "nll": -float(np.log(max(prediction[action - 1], 1e-15))),
                                "brier": float(prediction @ prediction - 2 * prediction[action - 1] + 1),
                                "one_step_win_probability": float((policy @ payoff[:, action - 1] + 1) / 2),
                                "gain_over_exact": float((policy - equilibrium) @ payoff[:, action - 1] / 2),
                            })
                    for name, model in models.items():
                        model.observe(forecasts[name], opponent_action=action, self_action=own)
        summaries = {name: {metric: float(np.mean([s[metric] for s in samples])) for metric in samples[0]} for name, samples in scores.items() if samples}
        player_results.append({"player": player, "named_sample": any(g["named_sample"] for g in parts["train"]),
            "scored_moves": len(scores["exact"]), "excluded_leap_moves_in_prefix": discarded, "controllers": summaries})
    return {
        "interpretation": "One-step deviations at logged states projected into pure DTH, followed by pure-DTH equilibrium. These are not full-game human win rates.",
        "players": player_results,
        "paired_gain_over_exact": {name: cluster_interval([p["controllers"][name]["gain_over_exact"] for p in player_results if p["controllers"]], protocol) for name in CONTROLLERS if name != "exact"},
        "paired_nll_vs_legacy": {name: cluster_interval([p["controllers"][name]["nll"] - p["controllers"]["legacy"]["nll"] for p in player_results if p["controllers"]], protocol) for name in ("bayes", "bayes_population")},
    }


def make_provider(name, artifact, agent, bayes_config, priors):
    if name == "exact":
        return ExactProvider(agent)
    model = PerfectHalOpponentModel() if name == "legacy" else BayesianHalOpponentModel(
        bayes=bayes_config, role_priors=priors if name == "bayes_population" else None)
    provider = PerfectHalPolicyProvider(artifact, agent=agent, opponent_model=model)
    return ResetMemoryProvider(provider) if name == "bayes_reset" else provider


def evaluate_games(partitions, artifact, agent, bayes_config, protocol, split):
    base = protocol[f"{split}_seed_base"]
    count = protocol[f"{split}_identities_per_family"]
    priors = fit_priors(partitions, protocol["prior_pseudocount"])
    scenarios = []
    for family_index, family in enumerate(SUPPORTED_FAMILIES):
        for index in range(count):
            seed = base + family_index * 100 + index
            scenarios.append((family, seed, lambda f=family, s=seed: ReactiveDTHOpponent(f, seed=s), priors, "synthetic", seed))
    for family, make_opponent in (("uniform", UniformProvider), ("equilibrium", lambda: ExactProvider(agent))):
        for index in range(count):
            seed = base + (2000 if family == "uniform" else 3000) + index
            scenarios.append((family, seed, make_opponent, priors, "synthetic", seed))
    for player, parts in partitions.items():
        if not parts[split]:
            continue
        excluded_priors = fit_priors(partitions, protocol["prior_pseudocount"], excluded_player=player)
        for reactive in (False, True):
            seed = base + 10000 + player * 10 + int(reactive)
            scenarios.append((f"human_fit_{player}_{'response' if reactive else 'categorical'}", seed,
                lambda games=parts["train"], r=reactive: FittedHumanOpponent(games, reactive=r), excluded_priors, "human_fitted", player))
    sessions = []
    for index, (family, seed, make_opponent, population, source, cluster) in enumerate(scenarios):
        result = {"family": family, "identity": seed, "source": source, "cluster": cluster, "controllers": {}}
        for name in CONTROLLERS:
            provider = make_provider(name, artifact, agent, bayes_config, population)
            opponent = make_opponent()
            games = []
            for game_index in range(protocol["games_per_identity"]):
                first = game_index % 2 == 0
                candidate_seat = "Hal" if first else "Baku"
                winner, moves = play_match_game(provider if first else opponent, opponent if first else provider,
                    seed=seed * 100 + game_index // 2, start_clock=720,
                    max_half_rounds=protocol["max_half_rounds"], game_index=game_index, pure_dth=True)
                games.append({"won": None if winner is None else winner == candidate_seat, "moves": moves, "seat": candidate_seat})
            result["controllers"][name] = games
        sessions.append(result)
        print(f"{split}: {index + 1}/{len(scenarios)} identities", flush=True)
    return {"sessions": sessions, "summaries": summarize_sessions(sessions, protocol)}


def summarize_sessions(sessions, protocol):
    def score(g):
        return 0.5 if g["won"] is None else float(g["won"])
    result = {}
    for source in ("synthetic", "human_fitted"):
        subset = [s for s in sessions if s["source"] == source]
        clusters = defaultdict(list)
        for session in subset:
            clusters[session["cluster"]].append(session)
        def paired(name, baseline):
            return cluster_interval([
                np.mean([score(g) for s in group for g in s["controllers"][name]])
                - np.mean([score(g) for s in group for g in s["controllers"][baseline]])
                for group in clusters.values()
            ], protocol)
        comparisons = {}
        for name in CONTROLLERS:
            games = [g for s in subset for g in s["controllers"][name]]
            comparisons[name] = {"games": len(games), "wins": sum(g["won"] is True for g in games),
                "losses": sum(g["won"] is False for g in games), "stopped": sum(g["won"] is None for g in games),
                "win_rate": sum(g["won"] is True for g in games) / len(games) if games else None,
                "mean_score": float(np.mean([score(g) for g in games])) if games else None,
                "paired_vs_exact": paired(name, "exact"),
                "paired_vs_legacy": paired(name, "legacy"),
                "paired_vs_reset": paired(name, "bayes_reset"),
                "by_seat": {seat: float(np.mean([score(g) for g in games if g["seat"] == seat])) if games else None for seat in ("Hal", "Baku")},
                "by_family": {family: float(np.mean([score(g) for s in subset if s["family"] == family for g in s["controllers"][name]])) for family in sorted({s["family"] for s in subset})},
                "by_game_index": [float(np.mean([score(s["controllers"][name][i]) for s in subset])) for i in range(protocol["games_per_identity"])] if subset else [],
            }
        result[source] = comparisons
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("src/hal_lab/experiments/perfect_hal_bayes_v2/perfect_hal_bayes_v2.json"))
    parser.add_argument("--artifact", type=Path, default=Path("src/dth/artifacts/complete_full_v1"))
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=Path("src/hal_lab/experiments/perfect_hal_bayes_v2/perfect_hal_bayes_v2_selection.json"))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("choose a fresh output path; evaluation reports are immutable")
    protocol = json.loads(args.config.read_text())
    if protocol["schema"] != "arena-perfect-hal-bayes-evaluation-v2":
        raise ValueError("unknown evaluation protocol")
    if not (0 < protocol["human_train_fraction"] < 1 and 0 < protocol["human_validation_fraction"] < 1
            and protocol["human_train_fraction"] + protocol["human_validation_fraction"] < 1):
        raise ValueError("human split fractions must leave a test partition")
    if protocol["games_per_identity"] < 2 or protocol["games_per_identity"] % 2:
        raise ValueError("paired evaluation requires a positive even game count")
    source_files = [Path(__file__), Path(bayesian_hal.__file__), Path(perfect_hal.__file__)]
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    selection = None
    if args.split == "test":
        selection = json.loads(args.selection.read_text())
        if selection["config_sha256"] != hashlib.sha256(args.config.read_bytes()).hexdigest() or selection["source_sha256"] != source_hashes:
            raise ValueError("test source or configuration differs from the validation selection")
        if selection["input_sha256"] != hashlib.sha256(args.human_data.read_bytes()).hexdigest():
            raise ValueError("test data differs from the validation selection")
    raw = json.loads(args.human_data.read_text())
    if raw["schema"] != "arena-anonymous-ledger-export-v1":
        raise ValueError("unknown human data schema")
    partitions = split_games(raw["games"], protocol)
    config_values = dict(protocol["bayes"])
    config_values["hazards"] = tuple(config_values["hazards"])
    bayes = BayesianHalConfig(**config_values)
    started = time.monotonic()
    agent = CompleteDTHAgent(args.artifact)
    report = {
        "schema": "arena-perfect-hal-bayes-results-v2", "split": args.split, "protocol": protocol,
        "input_sha256": hashlib.sha256(args.human_data.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "selection": selection,
        "table_manifest_sha256": hashlib.sha256((args.artifact / "tablebase.json").read_bytes()).hexdigest(),
        "partitions": {str(p): {s: [g["ordinal"] for g in gs] for s, gs in parts.items()} for p, parts in partitions.items()},
        "observed_baseline": {"games": len(raw["games"]), "finished": sum(g["status"] == "finished" for g in raw["games"]),
            "human_wins": sum(g["human_won"] is True for g in raw["games"]), "scope": "historical canonical STL; includes leap opportunities"},
        "claims": ["Logged-state gains project into pure DTH and assume equilibrium after one deviation.", "Simulator win rates do not establish human treatment effects.",
                   "Human-fitted opponents omit unobserved human responses.", "Browser identities may belong to the same person; identity intervals assume independence.",
                   "No whole-game optimality or safety claim for the exploit policy."],
    }
    report["human"] = evaluate_logs(partitions, agent, bayes, protocol, args.split)
    report["simulation"] = evaluate_games(partitions, args.artifact, agent, bayes, protocol, args.split)
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["simulation"]["summaries"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

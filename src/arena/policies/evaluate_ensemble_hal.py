"""Compare the policy ensemble with both constituents and an equal mixture."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from arena.match import play_match_game
from arena.policies.bayesian_hal import BayesianHalOpponentModel
from arena.policies.ensemble_hal import EnsembleHalConfig, EnsembleHalPolicyProvider
from arena.policies.evaluate_bayesian_hal import (
    ExactProvider, FittedHumanOpponent, UniformProvider, cluster_interval, split_games,
)
from arena.policies.opponent_league import ReactiveDTHOpponent, SUPPORTED_FAMILIES
from arena.policies.perfect_hal import PerfectHalOpponentModel, PerfectHalPolicyProvider
from dth.agent import CompleteDTHAgent

CONTROLLERS = ("exact", "old", "bayesian", "fixed_mix", "ensemble")


def make_provider(name, artifact, agent, config):
    if name == "exact":
        return ExactProvider(agent)
    if name in ("fixed_mix", "ensemble"):
        ensemble = EnsembleHalConfig(config["learning_rate"] if name == "ensemble" else 0, config["share"])
        return EnsembleHalPolicyProvider(artifact, agent=agent, ensemble=ensemble)
    model = PerfectHalOpponentModel() if name == "old" else BayesianHalOpponentModel()
    return PerfectHalPolicyProvider(artifact, agent=agent, opponent_model=model)


def summarize(sessions, config):
    result = {}
    for source in ("synthetic", "human_fitted"):
        subset = [s for s in sessions if s["source"] == source]
        if not subset:
            continue
        clusters = defaultdict(list)
        for session in subset:
            clusters[session["cluster"]].append(session)
        def wins(group, name):
            return [g["won"] is True for s in group for g in s["controllers"][name]]
        comparisons = {}
        for name in CONTROLLERS:
            games = [g for s in subset for g in s["controllers"][name]]
            comparisons[name] = {
                "games": len(games), "wins": sum(g["won"] is True for g in games),
                "losses": sum(g["won"] is False for g in games),
                "stopped": sum(g["won"] is None for g in games),
                "win_rate": float(np.mean(wins(subset, name))),
                "by_family": {family: float(np.mean(wins([s for s in subset if s["family"] == family], name)))
                              for family in sorted({s["family"] for s in subset})},
                "by_seat": {seat: float(np.mean([g["won"] is True for g in games if g["seat"] == seat]))
                            for seat in ("Hal", "Baku")},
                "paired_win_rate_vs": {
                    baseline: cluster_interval([float(np.mean(wins(group, name)) - np.mean(wins(group, baseline)))
                                                for group in clusters.values()], config)
                    for baseline in CONTROLLERS if baseline != name},
            }
        result[source] = comparisons
    return result


def evaluate(partitions, artifact, agent, config):
    base = config["test_seed_base"]
    scenarios = []
    for family_index, family in enumerate((*SUPPORTED_FAMILIES, "uniform", "equilibrium")):
        for index in range(config["test_identities_per_family"]):
            seed = base + family_index * 1000 + index
            factory = (UniformProvider if family == "uniform" else
                       (lambda: ExactProvider(agent)) if family == "equilibrium" else
                       (lambda f=family, s=seed: ReactiveDTHOpponent(f, seed=s)))
            scenarios.append((family, seed, factory, "synthetic", seed))
    for player, parts in partitions.items():
        if not parts["test"]:
            continue
        for reactive in (False, True):
            for replicate in range(config["human_emulator_replicates"]):
                seed = base + 100000 + player * 100 + int(reactive) * 10 + replicate
                scenarios.append(("human_response" if reactive else "human_categorical", seed,
                    lambda games=parts["train"], r=reactive: FittedHumanOpponent(games, reactive=r),
                    "human_fitted", player))
    sessions = []
    for index, (family, seed, factory, source, cluster) in enumerate(scenarios):
        session = {"family": family, "seed": seed, "source": source, "cluster": cluster,
                   "controllers": {}, "ensemble_diagnostics": {}}
        for name in CONTROLLERS:
            provider = make_provider(name, artifact, agent, config)
            opponent = factory()
            games = []
            for game_index in range(config["games_per_identity"]):
                first = game_index % 2 == 0
                seat = "Hal" if first else "Baku"
                winner, moves = play_match_game(provider if first else opponent, opponent if first else provider,
                    seed=seed * 100 + game_index // 2, start_clock=720,
                    max_half_rounds=config["max_half_rounds"], game_index=game_index, pure_dth=True)
                games.append({"won": None if winner is None else winner == seat, "moves": moves, "seat": seat})
            session["controllers"][name] = games
            if name in ("fixed_mix", "ensemble"):
                diagnostics = provider.experiment_diagnostics()
                diagnostics.pop("latest")
                session["ensemble_diagnostics"][name] = diagnostics
        sessions.append(session)
        print(f"{index + 1}/{len(scenarios)} sessions: {source} {family}", flush=True)
    return sessions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--human-data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("src/arena/config/perfect_hal_ensemble_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("choose a new output path; experiment reports are immutable")
    config = json.loads(args.config.read_text())
    EnsembleHalConfig(config["learning_rate"], config["share"])
    if config["games_per_identity"] < 2 or config["games_per_identity"] % 2:
        raise ValueError("paired seats require a positive even number of games")
    inputs = [args.config, args.human_data, args.artifact / "tablebase.json"]
    inputs += [Path(__file__).with_name(name + ".py") for name in
               ("evaluate_ensemble_hal", "ensemble_hal", "perfect_hal", "bayesian_hal", "evaluate_bayesian_hal", "opponent_league")]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    data = json.loads(args.human_data.read_text())
    partitions = split_games(data["games"], config)
    agent = CompleteDTHAgent(args.artifact)
    start = time.monotonic()
    sessions = evaluate(partitions, args.artifact, agent, config)
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)] for p in inputs):
        raise RuntimeError("experiment inputs changed during evaluation")
    report = {"schema_version": "arena-perfect-hal-policy-ensemble-evaluation-v1",
              "protocol": config, "input_sha256": hashes,
              "elapsed_seconds": time.monotonic() - start,
              "sessions": sessions, "summaries": summarize(sessions, config)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report["summaries"], indent=2))


if __name__ == "__main__":
    main()

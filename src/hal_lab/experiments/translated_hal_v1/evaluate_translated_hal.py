"""Select translated responses on validation seeds; bind fresh holdouts."""
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
from arena.policies.perfect_hal import PerfectHalPolicyProvider
from arena.policies.translated_hal import TranslatedHalConfig, TranslatedHalOpponentModel
from dth.agent import CompleteDTHAgent
from hal_lab.experiments.perfect_hal_bayes_v2.evaluate_bayesian_hal import (
    ExactProvider, FittedHumanOpponent, UniformProvider, cluster_interval, split_games,
)
from hal_lab.experiments.perfect_hal_ensemble_v1.evaluate_ensemble_hal import make_provider as baseline_provider
from hal_lab.harness.opponent_league import ReactiveDTHOpponent, SUPPORTED_FAMILIES

PROTOCOL = {
    "schema_version": "arena-translated-hal-evaluation-v1",
    "validation_seed_base": 34100000, "test_seed_base": 45100000,
    "validation_identities": 4, "test_identities": 16,
    "games_per_identity": 8, "max_half_rounds": 240,
    "human_emulator_replicates": 4, "bootstrap_seed": 45199000,
    "bootstrap_replicates": 5000, "human_train_fraction": 0.6,
    "human_validation_fraction": 0.2, "minimum_player_games": 5,
    "learning_rate": 2.0, "share": 0.04,
    "selection": "Maximize mean of synthetic and human-fitted validation scores; use stress score to break ties.",
    "promotion": "Holdout candidate-minus-Old clustered score lower bound >= -0.01 in both primary sources; positive pooled primary gain; no increased stop rate. Inspect seat and family regressions. Operational checks must pass.",
    "interpretation": "Human-fitted simulations use existing training prefixes and do not establish human win-rate gains.",
}
GRID = {
    "offset_default": asdict(TranslatedHalConfig()),
    "offset_fast": asdict(TranslatedHalConfig(expert_weight_retention=0.8, offset_retention=0.7)),
    "offset_slow": asdict(TranslatedHalConfig(expert_weight_retention=0.97, offset_retention=0.97)),
    "offset_moderate": asdict(TranslatedHalConfig(expert_weight_retention=0.88, offset_retention=0.85)),
}


class OffsetOpponent:
    """A held-out translated-response probe with a seeded offset and switch."""

    def __init__(self, family, seed):
        rng = np.random.default_rng(seed)
        self.family = family
        self.offset = int(rng.choice([-7, -3, -2, 2, 3, 7]))
        self.last = int(rng.integers(10, 50))
        self.count = 0

    def policy(self, decision):
        self.name = decision.actor_name
        anchor = 61 - self.last if self.family == "mirror_offset" else self.last
        offset = -self.offset if self.family == "switch_offset" and self.count >= 16 else self.offset
        return {int(np.clip(anchor + offset, 1, 60)): 1.0}

    def observe(self, record):
        self.last = record.check_time if record.dropper_name == self.name else record.drop_time
        self.count += 1


def provider(name, artifact, agent, configs):
    if name in configs:
        config = TranslatedHalConfig(**configs[name])
        return PerfectHalPolicyProvider(artifact, config, agent=agent,
            opponent_model=TranslatedHalOpponentModel(config))
    return baseline_provider(name, artifact, agent, PROTOCOL)


def summarize(sessions, names):
    output = {}
    for source in sorted({s["source"] for s in sessions}):
        subset = [s for s in sessions if s["source"] == source]
        clusters = defaultdict(list)
        for session in subset:
            clusters[session["cluster"]].append(session)
        def score(group, name):
            return np.mean([g["score"] for s in group for g in s["controllers"][name]])
        result = {}
        for name in names:
            games = [g for s in subset for g in s["controllers"][name]]
            result[name] = {
                "games": len(games), "wins": sum(g["won"] is True for g in games),
                "stopped": sum(g["won"] is None for g in games),
                "score": float(score(subset, name)),
                "win_rate": float(np.mean([g["won"] is True for g in games])),
                "by_seat": {seat: float(np.mean([g["score"] for g in games if g["seat"] == seat])) for seat in ("Hal", "Baku")},
                "by_family": {f: float(score([s for s in subset if s["family"] == f], name)) for f in sorted({s["family"] for s in subset})},
                "paired_score_vs_old": cluster_interval([float(score(group, name) - score(group, "old")) for group in clusters.values()], PROTOCOL),
            }
        output[source] = result
    return output


def evaluate(partitions, artifact, agent, split, configs, names):
    base = PROTOCOL[f"{split}_seed_base"]
    scenarios = []
    families = (*SUPPORTED_FAMILIES, "uniform", "equilibrium", "translated_offset", "mirror_offset", "switch_offset")
    for fi, family in enumerate(families):
        for index in range(PROTOCOL[f"{split}_identities"]):
            seed = base + fi * 1000 + index
            factory = (UniformProvider if family == "uniform" else
                (lambda: ExactProvider(agent)) if family == "equilibrium" else
                (lambda f=family, s=seed: OffsetOpponent(f, s)) if family.endswith("offset") else
                (lambda f=family, s=seed: ReactiveDTHOpponent(f, seed=s)))
            scenarios.append((family, seed, factory, "stress" if family.endswith("offset") else "synthetic", seed))
    for player, parts in partitions.items():
        if not parts[split]:
            continue
        for reactive in (False, True):
            for replicate in range(PROTOCOL["human_emulator_replicates"]):
                seed = base + 100000 + player * 100 + int(reactive) * 10 + replicate
                scenarios.append(("human_response" if reactive else "human_categorical", seed,
                    lambda games=parts["train"], r=reactive: FittedHumanOpponent(games, reactive=r), "human_fitted", player))
    sessions = []
    for index, (family, seed, factory, source, cluster) in enumerate(scenarios):
        session = {"family": family, "seed": seed, "source": source, "cluster": cluster, "controllers": {}}
        for name in names:
            candidate = provider(name, artifact, agent, configs)
            opponent = factory()
            games = []
            for gi in range(PROTOCOL["games_per_identity"]):
                first = gi % 2 == 0
                seat = "Hal" if first else "Baku"
                winner, moves = play_match_game(candidate if first else opponent, opponent if first else candidate,
                    seed=seed * 100 + gi // 2, start_clock=720,
                    max_half_rounds=PROTOCOL["max_half_rounds"], game_index=gi, pure_dth=True)
                won = None if winner is None else winner == seat
                games.append({"won": won, "score": 0.5 if won is None else float(won), "moves": moves, "seat": seat})
            session["controllers"][name] = games
        sessions.append(session)
        if (index + 1) % 8 == 0:
            print(f"{split}: {index + 1}/{len(scenarios)} identities", flush=True)
    return sessions


def hashes(artifact, human):
    paths = list(Path("src/arena/policies").glob("*.py"))
    paths += [Path("src/arena/match.py"), Path("src/arena/contracts.py"), human, artifact / "tablebase.json"]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--human-data", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or (args.split == "validation" and args.selection.exists()):
        raise ValueError("choose new paths; evidence is immutable")
    inputs = hashes(args.artifact, args.human_data)
    if args.split == "validation":
        configs, names = GRID, ("old", *GRID)
    else:
        selection = json.loads(args.selection.read_text())
        if selection["input_sha256"] != inputs or selection["protocol"] != PROTOCOL:
            raise ValueError("frozen inputs changed; use a new protocol and holdout")
        configs = {"candidate": selection["config"], "no_offsets": {**selection["config"], "translated": False}}
        names = ("old", "exact", "bayesian", "fixed_mix", "ensemble", *configs)
    partitions = split_games(json.loads(args.human_data.read_text())["games"], PROTOCOL)
    agent = CompleteDTHAgent(args.artifact)
    start = time.monotonic()
    sessions = evaluate(partitions, args.artifact, agent, args.split, configs, names)
    if hashes(args.artifact, args.human_data) != inputs:
        raise RuntimeError("inputs changed during evaluation")
    summaries = summarize(sessions, names)
    report = {"protocol": PROTOCOL, "split": args.split, "configs": configs, "input_sha256": inputs,
        "elapsed_seconds": time.monotonic() - start, "sessions": sessions, "summaries": summaries}
    write(args.output, report)
    if args.split == "validation":
        selected = max(configs, key=lambda n: (
            np.mean([summaries[s][n]["score"] for s in ("synthetic", "human_fitted")]), summaries["stress"][n]["score"]))
        write(args.selection, {"name": selected, "config": configs[selected], "protocol": PROTOCOL,
            "input_sha256": inputs, "validation_report": str(args.output),
            "validation_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()})
    print(json.dumps({s: {n: {k: v for k, v in r.items() if k not in ("by_family", "by_seat")} for n, r in values.items()} for s, values in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()

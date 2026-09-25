"""Run one larger confirmation of the frozen candidate without retuning."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from arena.match import play_match_game
from dth.agent import CompleteDTHAgent
from hal_lab.experiments.perfect_hal_bayes_v2.evaluate_bayesian_hal import ExactProvider, FittedHumanOpponent, UniformProvider, split_games
from hal_lab.experiments.translated_hal_v1.evaluate_translated_hal import (
    PROTOCOL, OffsetOpponent, hashes, provider, summarize, write,
)
from hal_lab.harness.opponent_league import ReactiveDTHOpponent, SUPPORTED_FAMILIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--human-data", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--first-test", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.protocol.exists() or args.output.exists():
        raise ValueError("confirmation paths must be new")
    selection = json.loads(args.selection.read_text())
    inputs = hashes(args.artifact, args.human_data)
    # The first evaluation's source files stay frozen. This new runner adds no model changes.
    if any(inputs[p] != sha for p, sha in selection["input_sha256"].items()):
        raise ValueError("candidate or first evaluation inputs changed")
    protocol = {**PROTOCOL, "schema_version": "arena-translated-hal-confirmation-v1",
        "seed_base": 67100000, "synthetic_identities_per_family": 64,
        "stress_identities_per_family": 32, "human_emulator_replicates": 40,
        "controllers": ["old", "candidate"], "config": selection["config"],
        "first_test_sha256": hashlib.sha256(args.first_test.read_bytes()).hexdigest(),
        "input_sha256": inputs,
        "decision": "Apply the original promotion gate once to this larger independent confirmation. Keep the first failed gate in the evidence. No model or threshold changes and no further confirmation attempts under this protocol."}
    write(args.protocol, protocol)
    partitions = split_games(json.loads(args.human_data.read_text())["games"], PROTOCOL)
    agent = CompleteDTHAgent(args.artifact)
    scenarios = []
    base = protocol["seed_base"]
    for fi, family in enumerate((*SUPPORTED_FAMILIES, "uniform", "equilibrium", "translated_offset", "mirror_offset", "switch_offset")):
        stress = family.endswith("offset")
        for index in range(protocol["stress_identities_per_family" if stress else "synthetic_identities_per_family"]):
            seed = base + fi * 1000 + index
            factory = (UniformProvider if family == "uniform" else
                (lambda: ExactProvider(agent)) if family == "equilibrium" else
                (lambda f=family, s=seed: OffsetOpponent(f, s)) if stress else
                (lambda f=family, s=seed: ReactiveDTHOpponent(f, seed=s)))
            scenarios.append((family, seed, factory, "stress" if stress else "synthetic", seed))
    for player, parts in partitions.items():
        if not parts["test"]:
            continue
        for reactive in (False, True):
            for replicate in range(protocol["human_emulator_replicates"]):
                seed = base + 1000000 + player * 10000 + int(reactive) * 1000 + replicate
                scenarios.append(("human_response" if reactive else "human_categorical", seed,
                    lambda games=parts["train"], r=reactive: FittedHumanOpponent(games, reactive=r), "human_fitted", player))
    if len({s[1] for s in scenarios}) != len(scenarios):
        raise ValueError("confirmation seed collision")
    sessions = []
    start = time.monotonic()
    configs = {"candidate": selection["config"]}
    for index, (family, seed, factory, source, cluster) in enumerate(scenarios):
        session = {"family": family, "seed": seed, "source": source, "cluster": cluster, "controllers": {}}
        for name in protocol["controllers"]:
            candidate, opponent = provider(name, args.artifact, agent, configs), factory()
            games = []
            for gi in range(protocol["games_per_identity"]):
                first = gi % 2 == 0
                seat = "Hal" if first else "Baku"
                winner, moves = play_match_game(candidate if first else opponent, opponent if first else candidate,
                    seed=seed * 100 + gi // 2, start_clock=720, max_half_rounds=protocol["max_half_rounds"],
                    game_index=gi, pure_dth=True)
                won = None if winner is None else winner == seat
                games.append({"won": won, "score": 0.5 if won is None else float(won), "moves": moves, "seat": seat})
            session["controllers"][name] = games
        sessions.append(session)
        if (index + 1) % 32 == 0:
            print(f"confirmation: {index + 1}/{len(scenarios)} identities", flush=True)
    if hashes(args.artifact, args.human_data) != inputs:
        raise RuntimeError("confirmation inputs changed")
    summaries = summarize(sessions, protocol["controllers"])
    primary = ("synthetic", "human_fitted")
    passed = all(summaries[s]["candidate"]["paired_score_vs_old"]["interval_95"][0] >= -0.01
        and summaries[s]["candidate"]["stopped"] <= summaries[s]["old"]["stopped"] for s in primary)
    passed = passed and sum(summaries[s]["candidate"]["wins"] - summaries[s]["old"]["wins"] for s in primary) > 0
    write(args.output, {"protocol": protocol, "sessions": sessions, "summaries": summaries,
        "elapsed_seconds": time.monotonic() - start, "statistical_gate_passed": passed})
    print(json.dumps({"statistical_gate_passed": passed, "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()

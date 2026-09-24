"""Fit an external human prior, freeze it, then test transfer without retuning."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import time
from urllib.request import urlopen

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln

from arena.dth_adapter import project_to_dth_state
from arena.match import play_match_game
from arena.policies.evaluate_bayesian_hal import ExactProvider, UniformProvider, cluster_interval, hard_response, human_observation
from arena.policies.evaluate_translated_hal import OffsetOpponent
from arena.policies.opponent_league import ReactiveDTHOpponent, SUPPORTED_FAMILIES
from arena.policies.perfect_hal import PerfectHalConfig, PerfectHalOpponentModel, PerfectHalPolicyProvider
from arena.policies.reward_prior_hal import RewardPriorConfig, RewardPriorHal
from arena.policies.translated_hal import TranslatedHalConfig, TranslatedHalOpponentModel
from dth.agent import CompleteDTHAgent

SOURCE_ROOT = "https://raw.githubusercontent.com/eliaka/repeatedgames/224a605a21127cac69763f990e077f7b05abd422/"
SOURCE_PATH = "human_experiment/analysis/repgames.csv"
SOURCE_SHA256 = "f3c3d46a2e34f85f950bfdffd0d04d6c6fc748268c14e5700646e44b2d42d11a"
NAMES = ("old", "translated", "neutral", "matched_neutral", "external")
PROTOCOL = {
    "schema": "arena-external-reward-prior-v1",
    "source_url": SOURCE_ROOT + SOURCE_PATH,
    "source_sha256": SOURCE_SHA256,
    "source_population": "Human Prolific participants playing two ten-round games against GPT-4; exclude LLM simulation folders.",
    "split": "SHA256('external-hal-prior-v1:' + participant) modulo 10: 0..5 train, 6..7 validation, 8..9 test. Both games stay with their participant.",
    "fit": "Two beta-binomial population priors for repeat after previous score zero/positive. Pool counts within each training participant. Fit alpha and beta in [0.05,100].",
    "selection": "Choose concentration multiplier from [0.25,1,4] by equal-participant validation prequential NLL. Do not refit on validation.",
    "transfer": "Preserve repeat odds relative to uniform chance: p60=a/(a+59*b), preserve concentration. A successful check or an evaded check is favorable. This mapping is a transfer hypothesis.",
    "controls": "Frozen translated and Old; adapter with binary Beta(1,1); adapter with external concentration and neutral binary mean; external adapter.",
    "adapter": "One additional expert with separate role/outcome counts; global distribution over alternatives. Use the frozen translated expert scoring rates. No target-domain selection.",
    "external_online": "Reset repeat counts at each new game; score trials 2..10 before updating.",
    "retrospective": "Score all recorded ordinary moves from a cold start per browser identity; reset all models after excluded leap turns. These identities appeared in prior experiments. No fresh STL human test exists in this corpus.",
    "claim_boundary": "Logged-state one-step DTH projections and fresh synthetic games do not establish live human win-rate gains. This protocol has no deployment authority.",
    "bootstrap_seed": 78199000, "bootstrap_replicates": 5000,
    "simulation_seed_base": 78100000, "identities_per_family": 8,
    "games_per_identity": 8, "max_half_rounds": 240,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def participant_split(participant):
    bucket = int(hashlib.sha256(f"external-hal-prior-v1:{participant}".encode()).hexdigest(), 16) % 10
    return "train" if bucket < 6 else "validation" if bucket < 8 else "test"


def load_external(path):
    if digest(path) != SOURCE_SHA256:
        raise ValueError("external human CSV checksum mismatch")
    participants = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader(io.StringIO(Path(path).read_text())):
        action, reward = int(row["action"]), int(row["score"])
        game = row["game"]
        if action not in (0, 1) or game not in ("PD", "BoS"):
            raise ValueError("unexpected human trial")
        if reward not in ({0, 5, 8, 10} if game == "PD" else {0, 7, 10}):
            raise ValueError("unexpected human reward")
        participants[row["id"]][game].append((action, int(reward > 0)))
    if len(participants) != 195 or any(set(games) != {"PD", "BoS"} or any(len(g) != 10 for g in games.values()) for games in participants.values()):
        raise ValueError("human participant/session cardinality mismatch")
    return dict(participants)


def transitions(games):
    for trials in games.values():
        for previous, current in zip(trials, trials[1:]):
            yield previous[1], int(previous[0] == current[0])


def fit_population(participants):
    counts = []
    for games in participants.values():
        person = np.zeros((2, 2))
        for outcome, repeat in transitions(games):
            person[outcome, 1 - repeat] += 1
        counts.append(person)
    counts = np.asarray(counts)
    priors, diagnostics = [], []
    for outcome in range(2):
        repeat, switch = counts[:, outcome, :].T
        def loss(log_ab):
            a, b = np.exp(log_ab)
            return -float(np.sum(betaln(repeat + a, switch + b) - betaln(a, b)))
        fits = [minimize(loss, np.log(start), method="L-BFGS-B", bounds=[(np.log(.05), np.log(100))] * 2)
                for start in ((1, 1), (4, 1), (1, 4))]
        valid = [f for f in fits if f.success and np.isfinite(f.fun) and np.isfinite(f.x).all()]
        if not valid:
            raise RuntimeError("population fit failed")
        fit = min(valid, key=lambda f: f.fun)
        priors.append(np.exp(fit.x).tolist())
        diagnostics.append({"negative_log_marginal_without_constant": float(fit.fun),
            "participants_with_evidence": int(np.sum(repeat + switch > 0)),
            "repeats": int(repeat.sum()), "switches": int(switch.sum()),
            "at_bound": bool(np.any(np.isclose(np.exp(fit.x), .05)) or np.any(np.isclose(np.exp(fit.x), 100)))})
    return np.asarray(priors), diagnostics


def external_scores(participants, prior):
    rows = []
    for participant, games in participants.items():
        losses = []
        for trials in games.values():
            counts = np.array(prior, dtype=float, copy=True)
            for previous, current in zip(trials, trials[1:]):
                outcome, repeat = previous[1], int(previous[0] == current[0])
                probability = counts[outcome, 1 - repeat] / counts[outcome].sum()
                losses.append(-float(np.log(probability)))
                counts[outcome, 1 - repeat] += 1
        rows.append({"participant": participant, "nll": float(np.mean(losses)), "transitions": len(losses)})
    return rows


def inputs(args):
    paths = [Path(__file__), Path(__file__).with_name("reward_prior_hal.py"),
        Path(__file__).with_name("translated_hal.py"), Path(__file__).with_name("perfect_hal.py"),
        Path(__file__).with_name("evaluate_bayesian_hal.py"), Path(__file__).with_name("evaluate_translated_hal.py"),
        Path(__file__).with_name("opponent_league.py"), Path("src/arena/match.py"), Path("src/arena/contracts.py"),
        args.selection, args.human_data, args.artifact / "tablebase.json", args.output / "human.csv"]
    return {str(path.resolve()): digest(path) for path in paths}


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "protocol.json").exists() or (args.output / "freeze.json").exists():
        raise ValueError("use a new output directory for a new fit")
    source = args.output / "human.csv"
    if not source.exists():
        with urlopen(SOURCE_ROOT + SOURCE_PATH, timeout=30) as stream:
            source.write_bytes(stream.read())
    for name in ("LICENSE.md", "README.md"):
        with urlopen(SOURCE_ROOT + name, timeout=30) as stream:
            (args.output / f"source-{name}").write_bytes(stream.read())
    selection = json.loads(args.selection.read_text())
    for name in ("perfect_hal.py", "translated_hal.py"):
        path = f"src/arena/policies/{name}"
        if digest(path) != selection["input_sha256"][path]:
            raise ValueError("frozen baseline changed")
    write_new(args.output / "protocol.json", {**PROTOCOL, "input_sha256": inputs(args)})
    participants = load_external(source)
    splits = {split: {p: games for p, games in participants.items() if participant_split(p) == split} for split in ("train", "validation", "test")}
    prior, diagnostics = fit_population(splits["train"])
    validation = {str(scale): float(np.mean([r["nll"] for r in external_scores(splits["validation"], prior * scale)])) for scale in (.25, 1., 4.)}
    scale = float(min(validation, key=validation.get))
    freeze = {"protocol_sha256": digest(args.output / "protocol.json"), "input_sha256": inputs(args),
        "baseline_config": selection["config"], "binary_prior": (prior * scale).tolist(),
        "fit_diagnostics": diagnostics, "concentration_scale": scale, "validation_nll": validation,
        "split_participants": {s: sorted(group) for s, group in splits.items()},
        "interpretation": PROTOCOL["claim_boundary"]}
    write_new(args.output / "freeze.json", freeze)
    print(json.dumps({k: freeze[k] for k in ("binary_prior", "fit_diagnostics", "validation_nll")}, indent=2))
    print({s: len(group) for s, group in splits.items()})


def make_models(freeze):
    prior = np.asarray(freeze["binary_prior"])
    config = RewardPriorConfig(**freeze["baseline_config"])
    return {"old": PerfectHalOpponentModel(),
        "translated": TranslatedHalOpponentModel(TranslatedHalConfig(**freeze["baseline_config"])),
        "neutral": RewardPriorHal(config, np.ones((2, 2))),
        "matched_neutral": RewardPriorHal(config, np.repeat(prior.sum(axis=1)[:, None] / 2, 2, axis=1)),
        "external": RewardPriorHal(config, prior)}


def paired_summary(rows, metrics):
    result = {}
    for name in NAMES:
        result[name] = {metric: cluster_interval([r["controllers"][name][metric] for r in rows], PROTOCOL) for metric in metrics}
        result[name]["paired_vs_translated"] = {metric: cluster_interval([r["controllers"][name][metric] - r["controllers"]["translated"][metric] for r in rows], PROTOCOL) for metric in metrics}
    result["external_vs_matched_neutral"] = {metric: cluster_interval([r["controllers"]["external"][metric] - r["controllers"]["matched_neutral"][metric] for r in rows], PROTOCOL) for metric in metrics}
    result["external_vs_neutral"] = {metric: cluster_interval([r["controllers"]["external"][metric] - r["controllers"]["neutral"][metric] for r in rows], PROTOCOL) for metric in metrics}
    return result


def retrospective(args, freeze, agent):
    groups = defaultdict(list)
    for game in sorted(json.loads(args.human_data.read_text())["games"], key=lambda g: g["ordinal"]):
        groups[game["player"]].append(game)
    rows = []
    for player, games in groups.items():
        models = make_models(freeze)
        scores = {name: [] for name in NAMES}
        excluded = 0
        for game in games:
            for index, move in enumerate(game["public_history"]):
                observation = human_observation(move)
                if observation is None:
                    excluded += 1
                    for model in models.values():
                        model.reset()
                    continue
                decision, role, action, own = observation
                stage = agent.stage_game(project_to_dth_state(decision))
                payoff = stage.matrix if decision.role == "dropper" else -stage.matrix.T
                for name, model in models.items():
                    forecast = model.predict(role, state_regime=model.state_regime(decision), game_index=game["ordinal"], game_decision_index=index)
                    response = hard_response(payoff @ forecast.policy)
                    scores[name].append({"nll": -float(np.log(max(forecast.policy[action - 1], 1e-15))),
                        "one_step_value": float((response @ payoff[:, action - 1] + 1) / 2),
                        "early": model.observations(role) < 5})
                    model.observe(forecast, opponent_action=action, self_action=own)
        rows.append({"player": player, "moves": len(scores["old"]), "excluded_leaps": excluded,
            "controllers": {name: {metric: float(np.mean([r[metric.removeprefix("early_")] for r in records if not metric.startswith("early_") or r["early"]]))
                for metric in ("nll", "one_step_value", "early_nll", "early_one_step_value")} for name, records in scores.items()}})
    return {"interpretation": PROTOCOL["retrospective"], "players": rows,
        "summary": paired_summary(rows, ("nll", "one_step_value", "early_nll", "early_one_step_value"))}


def simulations(args, freeze, agent):
    rows = []
    families = (*SUPPORTED_FAMILIES, "uniform", "equilibrium", "translated_offset", "mirror_offset", "switch_offset")
    for fi, family in enumerate(families):
        for identity in range(PROTOCOL["identities_per_family"]):
            seed = PROTOCOL["simulation_seed_base"] + fi * 1000 + identity
            row = {"family": family, "seed": seed, "controllers": {}}
            for name, model in make_models(freeze).items():
                provider = PerfectHalPolicyProvider(args.artifact, model.config, agent=agent, opponent_model=model)
                opponent = (UniformProvider() if family == "uniform" else ExactProvider(agent) if family == "equilibrium"
                    else OffsetOpponent(family, seed) if family.endswith("offset") else ReactiveDTHOpponent(family, seed=seed))
                games = []
                for gi in range(PROTOCOL["games_per_identity"]):
                    first = gi % 2 == 0
                    seat = "Hal" if first else "Baku"
                    winner, moves = play_match_game(provider if first else opponent, opponent if first else provider,
                        seed=seed * 100 + gi // 2, start_clock=720, max_half_rounds=PROTOCOL["max_half_rounds"], game_index=gi, pure_dth=True)
                    games.append({"score": .5 if winner is None else float(winner == seat), "won": winner == seat,
                        "stopped": winner is None, "moves": moves, "seat": seat})
                row["controllers"][name] = {"score": float(np.mean([g["score"] for g in games])), "games": games}
            rows.append(row)
        print(f"simulated {family}: {len(rows)} identities", flush=True)
    sections = {}
    for source, subset in (("synthetic", [r for r in rows if not r["family"].endswith("offset")]), ("offset_stress", [r for r in rows if r["family"].endswith("offset")])):
        summary = paired_summary(subset, ("score",))
        for name in NAMES:
            games = [g for r in subset for g in r["controllers"][name]["games"]]
            summary[name].update({"games": len(games), "wins": sum(g["won"] for g in games), "stopped": sum(g["stopped"] for g in games),
                "by_seat": {seat: float(np.mean([g["score"] for g in games if g["seat"] == seat])) for seat in ("Hal", "Baku")},
                "by_family": {f: float(np.mean([r["controllers"][name]["score"] for r in subset if r["family"] == f])) for f in sorted({r["family"] for r in subset})}})
        sections[source] = summary
    return {"identities": rows, "summary": sections}


def evaluate(args):
    freeze = json.loads((args.output / "freeze.json").read_text())
    if (args.output / "results.json").exists() or (args.output / "evaluation-start.json").exists():
        raise ValueError("this frozen evaluation has already started")
    if inputs(args) != freeze["input_sha256"] or digest(args.output / "protocol.json") != freeze["protocol_sha256"]:
        raise ValueError("frozen inputs changed")
    write_new(args.output / "evaluation-start.json", {"freeze_sha256": digest(args.output / "freeze.json")})
    start = time.monotonic()
    participants = load_external(args.output / "human.csv")
    test = {p: g for p, g in participants.items() if participant_split(p) == "test"}
    prior = np.asarray(freeze["binary_prior"])
    priors = {"neutral": np.ones((2, 2)), "matched_neutral": np.repeat(prior.sum(axis=1)[:, None] / 2, 2, axis=1), "external": prior}
    external = {name: external_scores(test, p) for name, p in priors.items()}
    external_summary = {name: cluster_interval([r["nll"] for r in rows], PROTOCOL) for name, rows in external.items()}
    external_summary["paired_external_minus_neutral"] = cluster_interval([a["nll"] - b["nll"] for a, b in zip(external["external"], external["neutral"], strict=True)], PROTOCOL)
    external_summary["paired_external_minus_matched_neutral"] = cluster_interval([a["nll"] - b["nll"] for a, b in zip(external["external"], external["matched_neutral"], strict=True)], PROTOCOL)
    agent = CompleteDTHAgent(args.artifact)
    logs = retrospective(args, freeze, agent)
    simulated = simulations(args, freeze, agent)
    if inputs(args) != freeze["input_sha256"]:
        raise RuntimeError("evaluation inputs changed during execution")
    result = {"freeze_sha256": digest(args.output / "freeze.json"), "external_holdout": {"participants": external, "summary": external_summary},
        "retrospective_stl": logs, "simulations": simulated, "elapsed_seconds": time.monotonic() - start,
        "deployment_promoted": False, "fresh_stl_human_test": "unavailable", "interpretation": PROTOCOL["claim_boundary"]}
    write_new(args.output / "results.json", result)
    print(json.dumps({"external_holdout": external_summary, "retrospective_external": logs["summary"]["external"],
        "simulation_external": {s: v["external"] for s, v in simulated["summary"].items()}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=Path("outputs/perfect-hal-bayes-v2/tablebase"))
    parser.add_argument("--human-data", type=Path, default=Path("outputs/perfect-hal-bayes-v2/human-games.json"))
    parser.add_argument("--selection", type=Path, default=Path("src/arena/config/translated_hal_v1_selection.json"))
    args = parser.parse_args()
    (prepare if args.mode == "fit" else evaluate)(args)


if __name__ == "__main__":
    main()

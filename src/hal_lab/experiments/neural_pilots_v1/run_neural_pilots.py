"""Run four bounded neural experiments with frozen models and fresh holdouts."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn

from arena.policies import aggro_hal, perfect_hal, translated_hal
from arena.policies.perfect_hal import PerfectHalOpponentModel, PerfectHalPolicyProvider
from arena.policies.translated_hal import TranslatedHalConfig, TranslatedHalOpponentModel
from dth.agent import CompleteDTHAgent
from hal_lab.experiments.neural_pilots_v1.neural_pilots import PublicMemory, make_network, tensor
from hal_lab.experiments.perfect_hal_bayes_v2 import evaluate_bayesian_hal
from hal_lab.experiments.perfect_hal_bayes_v2.evaluate_bayesian_hal import ExactProvider, UniformProvider, cluster_interval, hard_response
from hal_lab.experiments.translated_hal_v1 import evaluate_translated_hal
from hal_lab.experiments.translated_hal_v1.evaluate_translated_hal import OffsetOpponent
from hal_lab.harness import opponent_league
from hal_lab.harness.opponent_league import SUPPORTED_FAMILIES, ReactiveDTHOpponent
from hal_lab.training import aggro_env
from hal_lab.training.aggro_env import AggroSessionEnv

TRAIN_FAMILIES = (*SUPPORTED_FAMILIES[:10], "translated_offset")
VALIDATION_FAMILIES = (*TRAIN_FAMILIES, "counter_recent", "switch")
TEST_FAMILIES = (*SUPPORTED_FAMILIES, "uniform", "equilibrium", "translated_offset", "mirror_offset", "switch_offset")
UNSEEN_FAMILIES = ("retreat_after_detected_exploitation", "bait_then_reverse", "mirror_offset", "switch_offset")
PROTOCOL = {
    "schema": "arena-neural-pilots-v1", "seed": 88100000,
    "train_families": TRAIN_FAMILIES, "validation_families": VALIDATION_FAMILIES,
    "test_families": TEST_FAMILIES, "unseen_families": UNSEEN_FAMILIES,
    "train_seed": 88100000, "validation_seed": 89100000, "test_seed": 90100000,
    "rl_train_seed": 91100000, "rl_validation_seed": 92100000, "adversary_test_seed": 93100000,
    "train_identities": 48, "validation_identities": 12, "test_identities": 8,
    "games_per_session": 4, "max_half_rounds": 240, "epochs": 12,
    "batch_size": 128, "supervised_lr": .0008,
    "rl_updates": 160, "rl_sessions_per_update": 4, "rl_lr": .0003,
    "rl_checkpoint_interval": 40, "rl_validation_sessions": 24,
    "gamma": .995, "entropy_coef": .005, "bptt_steps": 32,
    "adversary_test_identities": 64,
    "bootstrap_seed": 90199000, "bootstrap_replicates": 5000,
    "predictor": "Two-layer 48-wide transformer or GRU; window 16; next-action cross entropy. Online score weighting adds the neural prediction to frozen translated Hal, starting at 1/25.",
    "selector": "64-wide MLP adjusts the 24 existing expert log weights; train by mixture next-action log loss.",
    "adversary": "64-wide GRU actor trains against frozen translated Hal with discounted session win/loss rewards and an exact-matrix tactical input.",
    "probe": "Same actor trains across reactive opponent sessions; a myopic control maximizes revealed one-step exact payoff. Compare late-game wins and reset-GRU control before claiming useful session memory. This pilot cannot establish active information seeking from wins alone.",
    "data": "Generated public histories. Current opponent actions and latent opponent parameters never enter inputs. Train behavior mixes translated responses with uniform exploration or exact policy by identity.",
    "selection": "Select supervised epochs by validation prediction NLL; select RL checkpoints by validation session score. Freeze before test and do not retry a test after tuning.",
    "memory_ablation": "Reset or shuffle sequence history while retaining the current public token and statistical forecaster. Actor reset clears its GRU at each decision. These isolate neural memory, not all memory.",
    "scope": "Research-only pure DTH. No production changes, leap-action optimization, or human win-rate claims. Preserve prior Aggro checkpoints and all frozen alternatives.",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def softmax(values):
    result = np.exp(values - np.max(values))
    return result / result.sum()


class Experiment:
    def __init__(self, args):
        self.args = args
        selection = json.loads(args.selection.read_text())
        self.config = TranslatedHalConfig(**selection["config"])
        for name in ("perfect_hal.py", "translated_hal.py"):
            path = Path("src/arena/policies") / name
            if sha(path) != selection["input_sha256"][str(path)]:
                raise ValueError("frozen baseline source changed")
        self.agent = CompleteDTHAgent(args.artifact)

    def opponent(self, family, seed):
        if family in ("translated", "old"):
            model = TranslatedHalOpponentModel(self.config) if family == "translated" else PerfectHalOpponentModel()
            return PerfectHalPolicyProvider(self.args.artifact, model.config, agent=self.agent, opponent_model=model)
        if family == "equilibrium":
            return ExactProvider(self.agent)
        if family == "uniform":
            return UniformProvider()
        if family.endswith("offset"):
            return OffsetOpponent(family, seed)
        return ReactiveDTHOpponent(family, seed=seed)

    def environment(self, family, seed, first=True):
        return AggroSessionEnv(self.opponent(family, seed), self.agent,
            games_per_session=PROTOCOL["games_per_session"], seed=seed * 100,
            max_half_rounds=PROTOCOL["max_half_rounds"], learner_starts_in_hal_seat=first)

    def input_hashes(self):
        paths = [Path(__file__), Path(__file__).with_name("neural_pilots.py"), self.args.selection,
            self.args.artifact / "tablebase.json", Path("src/arena/dth_adapter.py"), Path("src/arena/contracts.py")]
        paths += [Path(module.__file__) for module in (aggro_env, aggro_hal, opponent_league,
            perfect_hal, translated_hal, evaluate_bayesian_hal, evaluate_translated_hal)]
        return {str(p.resolve()): sha(p) for p in paths}

    def collect(self, split):
        families = TRAIN_FAMILIES if split == "train" else VALIDATION_FAMILIES if split == "validation" else TEST_FAMILIES
        count = PROTOCOL[f"{split}_identities"]
        data = defaultdict(list)
        for fi, family in enumerate(families):
            for identity in range(count):
                seed = PROTOCOL[f"{split}_seed"] + fi * 1000 + identity
                rng = np.random.default_rng(seed)
                env, memory = self.environment(family, seed, identity % 2 == 0), PublicMemory(self.config)
                decision = env.reset()
                while decision is not None:
                    forecast, window, gate = memory.prepare(decision)
                    if identity % 3 == 0:
                        policy = decision.exact_policy
                    else:
                        policy = .8 * hard_response(decision.role_oriented_matrix @ forecast.policy) + .2 / 60
                    action = int(rng.choice(60, p=policy)) + 1
                    step = env.step(action)
                    data["windows"].append(window)
                    data["gate"].append(gate)
                    data["experts"].append(forecast.expert_policies.astype(np.float32))
                    data["baseline"].append(forecast.policy.astype(np.float32))
                    data["target"].append(step.record.opponent_action - 1)
                    data["identity"].append(seed)
                    data["family"].append(fi)
                    memory.observe(step.record.opponent_action, action)
                    decision = step.next_decision
            print(f"collect {split} {family}: {len(data['target'])} decisions", flush=True)
        data = {k: np.asarray(v) for k, v in data.items()}
        np.savez_compressed(self.args.output / f"{split}.npz", **data)
        return data

    def session(self, family, seed, kind, network=None, *, training=False, ablation=None):
        rng = np.random.default_rng(seed + 173)
        ablation_rng = np.random.default_rng(seed + 349)
        env, memory = self.environment(family, seed, seed % 2 == 0), PublicMemory(self.config)
        old = PerfectHalOpponentModel() if kind == "old" else None
        decision = env.reset()
        games, nlls, weights, latencies = [], [], [], []
        logs, critics, entropies, rewards, local_losses = [], [], [], [], []
        hidden = None
        steps = 0
        while decision is not None:
            started = time.perf_counter()
            forecast, window, gate = memory.prepare(decision)
            prediction = forecast.policy
            if kind in ("transformer", "gru", "selector"):
                prediction, weight = memory.predict(network, kind, forecast, window, gate, ablation=ablation, rng=ablation_rng)
                weights.append(float(weight))
            elif old is not None:
                old_forecast = old.predict(decision.opponent_role, state_regime=forecast.context.state_regime,
                    game_index=decision.game_index, game_decision_index=decision.half_round_index)
                prediction = old_forecast.policy
            if kind in ("adversary", "probe", "myopic", "initial"):
                if ablation == "reset":
                    hidden = None
                with nullcontext() if training else torch.no_grad():
                    values = decision.role_oriented_matrix @ prediction
                    logits, critic, hidden = network(tensor(window[-1]), tensor(prediction), tensor(values), hidden)
                    distribution = torch.distributions.Categorical(logits=logits[0])
                    policy = distribution.probs.detach().numpy().astype(float)
                    policy /= policy.sum()
                    action = int(rng.choice(60, p=policy)) + 1
                    if training:
                        logs.append(distribution.log_prob(torch.tensor(action - 1)))
                        critics.append(critic[0])
                        entropies.append(distribution.entropy())
            else:
                policy = decision.exact_policy if kind == "exact" else hard_response(decision.role_oriented_matrix @ prediction)
                action = int(rng.choice(60, p=policy)) + 1
            latencies.append((time.perf_counter() - started) * 1000)
            step = env.step(action)
            record = step.record
            nlls.append(-float(np.log(max(prediction[record.opponent_action - 1], 1e-12))))
            if training:
                rewards.append(record.terminal_game_reward)
                column = tensor(decision.role_oriented_matrix[:, record.opponent_action - 1])[0]
                local_losses.append(-(distribution.probs * column).sum())
            memory.observe(record.opponent_action, action)
            if old is not None:
                old.observe(old_forecast, opponent_action=record.opponent_action, self_action=action)
            if record.game_boundary:
                games.append({"score": (record.terminal_game_reward + 1) / 2,
                    "stopped": record.game_truncated, "seat": decision.learner_seat,
                    "moves": record.half_round_index + 1})
            steps += 1
            if training and hidden is not None and steps % PROTOCOL["bptt_steps"] == 0:
                hidden = hidden.detach()
            decision = step.next_decision
        result = {"games": games, "score": float(np.mean([g["score"] for g in games])),
            "late_score": float(np.mean([g["score"] for g in games[2:]])),
            "nll": float(np.mean(nlls)), "neural_weight": float(np.mean(weights)) if weights else None,
            "decision_ms_p50": float(np.median(latencies)), "decision_ms_p95": float(np.quantile(latencies, .95))}
        if training:
            if kind == "myopic":
                loss = torch.stack(local_losses).mean() - PROTOCOL["entropy_coef"] * torch.stack(entropies).mean()
            else:
                returns, future = [], 0.
                for reward in reversed(rewards):
                    future = reward + PROTOCOL["gamma"] * future
                    returns.append(future)
                target = torch.tensor(list(reversed(returns)), dtype=torch.float32)
                values = torch.stack(critics)
                advantage = target - values.detach()
                loss = -(torch.stack(logs) * advantage).mean() + .25 * (values - target).square().mean()
                loss -= PROTOCOL["entropy_coef"] * torch.stack(entropies).mean()
            return result, loss
        return result


def supervised_loss(network, kind, data, indices):
    targets = torch.tensor(data["target"][indices], dtype=torch.long)
    if kind == "selector":
        prediction = network(torch.tensor(data["gate"][indices]), torch.tensor(data["experts"][indices]))
        return -torch.log(prediction.gather(1, targets[:, None]).clamp_min(1e-9)).mean()
    return nn.functional.cross_entropy(network(torch.tensor(data["windows"][indices])), targets)


def validation_loss(network, kind, data):
    with torch.no_grad():
        parts = [(float(supervised_loss(network, kind, data, slice(i, i + 512))), min(512, len(data["target"]) - i))
            for i in range(0, len(data["target"]), 512)]
    return sum(loss * count for loss, count in parts) / sum(count for _, count in parts)


def train_supervised(experiment, train, validation, kind):
    torch.manual_seed(PROTOCOL["seed"])
    network = make_network(kind)
    optimizer = torch.optim.AdamW(network.parameters(), lr=PROTOCOL["supervised_lr"], weight_decay=.0001)
    rng, history = np.random.default_rng(PROTOCOL["seed"]), []
    best = float("inf")
    for epoch in range(1, PROTOCOL["epochs"] + 1):
        network.train()
        indices = rng.permutation(len(train["target"]))
        losses = []
        for start in range(0, len(indices), PROTOCOL["batch_size"]):
            optimizer.zero_grad()
            loss = supervised_loss(network, kind, train, indices[start:start + PROTOCOL["batch_size"]])
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite supervised loss")
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        network.eval()
        score = validation_loss(network, kind, validation)
        history.append({"epoch": epoch, "train_nll": float(np.mean(losses)), "validation_nll": score})
        if score < best:
            best = score
            torch.save({"kind": kind, "state": network.state_dict(), "epoch": epoch}, experiment.args.output / f"{kind}.pt")
        print(kind, history[-1], flush=True)
    return history


def train_actor(experiment, kind):
    torch.manual_seed(PROTOCOL["seed"])
    network = make_network(kind)
    optimizer = torch.optim.AdamW(network.parameters(), lr=PROTOCOL["rl_lr"], weight_decay=.0001)
    history, best = [], -float("inf")
    for update in range(1, PROTOCOL["rl_updates"] + 1):
        network.train()
        optimizer.zero_grad()
        scores, losses = [], []
        for j in range(PROTOCOL["rl_sessions_per_update"]):
            index = (update - 1) * PROTOCOL["rl_sessions_per_update"] + j
            family = "translated" if kind == "adversary" else TRAIN_FAMILIES[index % len(TRAIN_FAMILIES)]
            result, loss = experiment.session(family, PROTOCOL["rl_train_seed"] + index, kind, network, training=True)
            (loss / PROTOCOL["rl_sessions_per_update"]).backward()
            scores.append(result["score"])
            losses.append(float(loss.detach()))
        nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if update % PROTOCOL["rl_checkpoint_interval"] == 0:
            network.eval()
            validation = [experiment.session("translated" if kind == "adversary" else VALIDATION_FAMILIES[j % len(VALIDATION_FAMILIES)],
                PROTOCOL["rl_validation_seed"] + j, kind, network)["score"] for j in range(PROTOCOL["rl_validation_sessions"])]
            score = float(np.mean(validation))
            history.append({"update": update, "train_score_last_batch": float(np.mean(scores)), "loss_last_batch": float(np.mean(losses)), "validation_score": score})
            if score > best:
                best = score
                torch.save({"kind": kind, "state": network.state_dict(), "update": update}, experiment.args.output / f"{kind}.pt")
            print(kind, history[-1], flush=True)
    return history


def train(experiment):
    output = experiment.args.output
    if (output / "protocol.json").exists():
        raise ValueError("use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    inputs = experiment.input_hashes()
    write(output / "protocol.json", {**PROTOCOL, "input_sha256": inputs,
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()}})
    started = time.monotonic()
    training, validation = experiment.collect("train"), experiment.collect("validation")
    history = {kind: train_supervised(experiment, training, validation, kind) for kind in ("transformer", "gru", "selector")}
    for kind in ("adversary", "probe", "myopic"):
        history[kind] = train_actor(experiment, kind)
    if inputs != experiment.input_hashes():
        raise RuntimeError("training inputs changed")
    write(output / "training.json", {"history": history, "train_decisions": len(training["target"]),
        "validation_decisions": len(validation["target"]), "elapsed_seconds": time.monotonic() - started})
    write(output / "freeze.json", {"input_sha256": inputs,
        "artifacts": {str(p.resolve()): sha(p) for p in [*output.glob("*.pt"), output / "protocol.json", output / "training.json", output / "train.npz", output / "validation.npz"]}})


def load_network(output, kind):
    network = make_network(kind)
    if kind != "initial":
        payload = torch.load(output / f"{kind}.pt", map_location="cpu", weights_only=True)
        if payload["kind"] != kind:
            raise ValueError("checkpoint architecture mismatch")
        network.load_state_dict(payload["state"], strict=True)
    return network.eval()


def summarize(rows, baseline):
    names = list(rows[0]["controllers"])
    summary = {}
    for name in names:
        games = [g for r in rows for g in r["controllers"][name]["games"]]
        summary[name] = {"games": len(games), "wins": sum(g["score"] == 1 for g in games),
            "stopped": sum(g["stopped"] for g in games),
            "score": cluster_interval([r["controllers"][name]["score"] for r in rows], PROTOCOL),
            "paired_vs_baseline": cluster_interval([r["controllers"][name]["score"] - r["controllers"][baseline]["score"] for r in rows], PROTOCOL),
            "late_paired_vs_baseline": cluster_interval([r["controllers"][name]["late_score"] - r["controllers"][baseline]["late_score"] for r in rows], PROTOCOL),
            "mean_session_p95_ms": float(np.mean([r["controllers"][name]["decision_ms_p95"] for r in rows])),
            "by_seat": {s: float(np.mean([g["score"] for g in games if g["seat"] == s])) for s in ("Hal", "Baku")}}
    return summary


def evaluate(experiment):
    output = experiment.args.output
    freeze = json.loads((output / "freeze.json").read_text())
    if (output / "evaluation-start.json").exists():
        raise ValueError("this frozen test already started")
    if freeze["input_sha256"] != experiment.input_hashes() or any(sha(p) != h for p, h in freeze["artifacts"].items()):
        raise ValueError("frozen inputs changed")
    write(output / "evaluation-start.json", {"freeze_sha256": sha(output / "freeze.json")})
    networks = {k: load_network(output, k) for k in ("transformer", "gru", "selector", "probe", "myopic", "adversary", "initial")}
    test_data = experiment.collect("test")
    prediction = {kind: validation_loss(networks[kind], kind, test_data) for kind in ("transformer", "gru", "selector")}
    prediction["translated"] = float(-np.log(np.maximum(test_data["baseline"][np.arange(len(test_data["target"])), test_data["target"]], 1e-9)).mean())
    rows = []
    variants = ("old", "translated", "transformer", "gru", "selector", "probe", "myopic", "probe_reset", "transformer_reset", "transformer_shuffle")
    for fi, family in enumerate(TEST_FAMILIES):
        for identity in range(PROTOCOL["test_identities"]):
            seed = PROTOCOL["test_seed"] + fi * 1000 + identity + 500
            row = {"family": family, "seed": seed, "controllers": {}}
            for name in variants:
                kind, _, ablation = name.partition("_")
                row["controllers"][name] = experiment.session(family, seed, kind, networks.get(kind), ablation=ablation or None)
            rows.append(row)
        print(f"test {family}: {len(rows)} identities", flush=True)
    adversary = []
    for fi, family in enumerate(("translated", "old", "equilibrium")):
        for identity in range(PROTOCOL["adversary_test_identities"]):
            seed = PROTOCOL["adversary_test_seed"] + fi * 1000 + identity
            adversary.append({"family": family, "seed": seed, "controllers": {
                name: experiment.session(family, seed, name, networks.get(name)) for name in ("adversary", "initial", "translated")}})
        print(f"adversary test {family}", flush=True)
    summary = {"all": summarize(rows, "translated"),
        "unseen_families": summarize([r for r in rows if r["family"] in UNSEEN_FAMILIES], "translated"),
        "by_family": {f: summarize([r for r in rows if r["family"] == f], "translated") for f in TEST_FAMILIES}}
    probe_controls = {control: {metric: cluster_interval([r["controllers"]["probe"][metric] - r["controllers"][control][metric] for r in rows], PROTOCOL)
        for metric in ("score", "late_score")} for control in ("myopic", "probe_reset")}
    if freeze["input_sha256"] != experiment.input_hashes() or any(sha(p) != h for p, h in freeze["artifacts"].items()):
        raise RuntimeError("inputs changed during test")
    result = {"freeze_sha256": sha(output / "freeze.json"), "raw_prediction_nll": prediction,
        "prediction_note": "Raw neural forecasts on a fixed behavior-policy corpus. Full-game neural predictors use online score weighting with translated Hal.",
        "sessions": rows, "summary": summary, "probe_controls": probe_controls,
        "adversary_sessions": adversary, "adversary_summary": {f: summarize([r for r in adversary if r["family"] == f], "initial") for f in ("translated", "old", "equilibrium")},
        "parameter_counts": {k: sum(p.numel() for p in n.parameters()) for k, n in networks.items()},
        "promoted": False, "interpretation": PROTOCOL["scope"]}
    write(output / "results.json", result)
    print(json.dumps({"prediction_nll": prediction, "wins": {n: s["wins"] for n, s in summary["all"].items()}, "probe_controls": probe_controls}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=Path("outputs/perfect-hal-bayes-v2/tablebase"))
    parser.add_argument("--selection", type=Path, default=Path("src/arena/config/translated_hal_v1_selection.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    np.random.seed(PROTOCOL["seed"])
    torch.manual_seed(PROTOCOL["seed"])
    experiment = Experiment(args)
    (train if args.mode == "train" else evaluate)(experiment)


if __name__ == "__main__":
    main()

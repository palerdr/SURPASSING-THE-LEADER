"""Compare selector capacity and retrained ablations on fresh identity splits."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn

from arena.policies.evaluate_bayesian_hal import cluster_interval, hard_response
from arena.policies.neural_pilots import PublicMemory
from arena.policies.run_neural_pilots import (
    Experiment, TRAIN_FAMILIES, VALIDATION_FAMILIES, TEST_FAMILIES, UNSEEN_FAMILIES,
    load_network, sha, write,
)
from arena.policies.selector_study import Selector, SelectorEnsemble

PROTOCOL = {
    "schema": "arena-selector-architecture-study-v1",
    "train_seed": 111100000, "validation_seed": 112100000,
    "validation_play_seed": 113100000, "test_play_seed": 114100000,
    "test_seed": 115100000, "bootstrap_seed": 114199000, "bootstrap_replicates": 10000,
    "train_identities": 128, "validation_identities": 24, "test_identities": 32,
    "validation_play_identities": 12, "test_play_identities": 32,
    "training_seeds": [210, 211, 212], "epochs": 12, "batch_size": 256,
    "learning_rate": .0008, "weight_decay": .0001,
    "games_per_session": 4, "max_half_rounds": 240,
    "architectures": ["small", "wide", "attention"],
    "train_families": TRAIN_FAMILIES, "validation_families": VALIDATION_FAMILIES,
    "test_families": TEST_FAMILIES, "excluded_from_train_and_validation": UNSEEN_FAMILIES,
    "data": "Fresh public histories from exact, translated-with-exploration, or prior-selector-with-exploration behavior. Score before reveal; sample actions outside networks. No human data.",
    "epoch_selection": "Minimum equal-identity validation next-action NLL, each training seed selected separately. Equal-weight forecast ensemble of all three seeds; no best-seed selection.",
    "architecture_selection": "Maximum validation whole-session score across the three ensembles. Break ties by lower validation NLL, then parameter count. Freeze before fresh test.",
    "ablations": "Retrain selected architecture without error inputs, without public context, and without inherited log weights. Retrain a static 24-parameter correction. Attention also gets a retrained no-shapes control. Same data, seeds, epochs, optimizer and checkpoint criterion.",
    "controls": "Old, frozen translated, prior pilot checkpoint, zero-correction untrained selector, all three architecture ensembles, selected seed 210 alone, and retrained ablations. No-errors and no-context retain indirect information through existing predictors.",
    "test": "One frozen evaluation. Primary contrast is the validation-selected ensemble versus translated Hal. Architecture and ablation contrasts are exploratory. Report paired identity-bootstrap intervals and family/seat slices.",
    "scope": "Research-only pure DTH. No human win-rate claim, deployment change, or action-61 optimization.",
}


class Study(Experiment):
    def __init__(self, args):
        super().__init__(args)
        evidence = json.loads(Path("src/arena/config/neural_pilots_v1_results.json").read_text())
        prior = args.prior / "selector.pt"
        if sha(prior) != evidence["artifacts"]["outputs/neural-pilots-v1/selector.pt"]:
            raise ValueError("prior selector checkpoint changed")
        self.prior = load_network(args.prior, "selector")

    def input_hashes(self):
        result = super().input_hashes()
        paths = [Path(__file__), Path(__file__).with_name("selector_study.py"), self.args.prior / "selector.pt",
            Path("src/arena/config/neural_pilots_v1_results.json")]
        result.update({str(p.resolve()): sha(p) for p in paths})
        return result

    def collect_selectors(self, split):
        families = TRAIN_FAMILIES if split == "train" else VALIDATION_FAMILIES if split == "validation" else TEST_FAMILIES
        data = defaultdict(list)
        for fi, family in enumerate(families):
            for identity in range(PROTOCOL[f"{split}_identities"]):
                seed = PROTOCOL[f"{split}_seed"] + fi * 1000 + identity
                rng = np.random.default_rng(seed)
                env, memory = self.environment(family, seed, identity % 2 == 0), PublicMemory(self.config)
                decision = env.reset()
                while decision is not None:
                    forecast, window, features = memory.prepare(decision)
                    prediction = forecast.policy
                    if identity % 3 == 2:
                        prediction, _ = memory.predict(self.prior, "selector", forecast, window, features)
                    policy = decision.exact_policy if identity % 3 == 0 else .8 * hard_response(decision.role_oriented_matrix @ prediction) + .2 / 60
                    action = int(rng.choice(60, p=policy)) + 1
                    step = env.step(action)
                    data["gate"].append(features)
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


def identity_means(losses, identities):
    return np.asarray([np.mean(losses[identities == i]) for i in np.unique(identities)])


def row_losses(network, data):
    rows = []
    with torch.no_grad():
        for start in range(0, len(data["target"]), 512):
            batch = slice(start, start + 512)
            prediction = network(torch.tensor(data["gate"][batch]), torch.tensor(data["experts"][batch]))
            targets = torch.tensor(data["target"][batch], dtype=torch.long)
            rows.extend(-prediction.gather(1, targets[:, None]).clamp_min(1e-9).log().numpy()[:, 0])
    return np.asarray(rows)


def fit(study, train, validation, architecture, ablation="none"):
    members, histories = [], []
    for seed in PROTOCOL["training_seeds"]:
        torch.manual_seed(seed)
        network = Selector(architecture, ablation)
        optimizer = torch.optim.AdamW(network.parameters(), lr=PROTOCOL["learning_rate"], weight_decay=PROTOCOL["weight_decay"])
        rng, best, history = np.random.default_rng(seed), float("inf"), []
        path = study.args.output / f"{architecture}-{ablation}-{seed}.pt"
        for epoch in range(1, PROTOCOL["epochs"] + 1):
            network.train()
            indices = rng.permutation(len(train["target"]))
            for start in range(0, len(indices), PROTOCOL["batch_size"]):
                batch = indices[start:start + PROTOCOL["batch_size"]]
                prediction = network(torch.tensor(train["gate"][batch]), torch.tensor(train["experts"][batch]))
                targets = torch.tensor(train["target"][batch], dtype=torch.long)
                loss = -prediction.gather(1, targets[:, None]).clamp_min(1e-9).log().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            network.eval()
            score = float(identity_means(row_losses(network, validation), validation["identity"]).mean())
            history.append({"epoch": epoch, "validation_nll": score})
            if score < best:
                best = score
                torch.save({"architecture": architecture, "ablation": ablation, "seed": seed,
                    "epoch": epoch, "validation_nll": score, "state": network.state_dict()}, path)
            if epoch % 4 == 0:
                print(f"fit {architecture}/{ablation} seed {seed} epoch {epoch}: {score:.5f}", flush=True)
        payload = torch.load(path, weights_only=True)
        network.load_state_dict(payload["state"])
        members.append(network.eval())
        histories.append({"seed": seed, "selected_epoch": payload["epoch"], "best_validation_nll": best, "history": history})
    return SelectorEnsemble(members).eval(), histories


def load_ensemble(output, architecture, ablation="none"):
    members = []
    for seed in PROTOCOL["training_seeds"]:
        payload = torch.load(output / f"{architecture}-{ablation}-{seed}.pt", weights_only=True)
        if (payload["architecture"], payload["ablation"], payload["seed"]) != (architecture, ablation, seed):
            raise ValueError("checkpoint contract mismatch")
        network = Selector(architecture, ablation)
        network.load_state_dict(payload["state"], strict=True)
        members.append(network.eval())
    return SelectorEnsemble(members).eval()


def play(study, families, seed_base, count, networks, baselines=()):
    rows = []
    for fi, family in enumerate(families):
        for identity in range(count):
            seed = seed_base + fi * 1000 + identity
            row = {"family": family, "seed": seed, "controllers": {}}
            for name in baselines:
                row["controllers"][name] = study.session(family, seed, name)
            for name, network in networks.items():
                row["controllers"][name] = study.session(family, seed, "selector", network)
            rows.append(row)
        print(f"play {seed_base} {family}: {len(rows)} identities", flush=True)
    return rows


def summarize(rows, baseline):
    summary = {}
    for name in rows[0]["controllers"]:
        games = [g for row in rows for g in row["controllers"][name]["games"]]
        summary[name] = {
            "games": len(games), "wins": sum(g["score"] == 1 for g in games),
            "stopped": sum(g["stopped"] for g in games),
            "score": cluster_interval([r["controllers"][name]["score"] for r in rows], PROTOCOL),
            "paired_vs_baseline": cluster_interval([r["controllers"][name]["score"] - r["controllers"][baseline]["score"] for r in rows], PROTOCOL),
            "late_paired_vs_baseline": cluster_interval([r["controllers"][name]["late_score"] - r["controllers"][baseline]["late_score"] for r in rows], PROTOCOL),
            "mean_session_p95_ms": float(np.mean([r["controllers"][name]["decision_ms_p95"] for r in rows])),
            "by_seat": {s: float(np.mean([g["score"] for g in games if g["seat"] == s])) for s in ("Hal", "Baku")},
        }
    return summary


def train(study):
    output = study.args.output
    if (output / "protocol.json").exists():
        raise ValueError("choose a new study output directory")
    output.mkdir(parents=True, exist_ok=True)
    inputs = study.input_hashes()
    write(output / "protocol.json", {**PROTOCOL, "input_sha256": inputs,
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()}})
    started = time.monotonic()
    training, validation = study.collect_selectors("train"), study.collect_selectors("validation")
    networks, histories = {}, {}
    for architecture in PROTOCOL["architectures"]:
        networks[architecture], histories[architecture] = fit(study, training, validation, architecture)
    validation_games = play(study, VALIDATION_FAMILIES, PROTOCOL["validation_play_seed"], PROTOCOL["validation_play_identities"], networks)
    scores = {name: {"score": float(np.mean([r["controllers"][name]["score"] for r in validation_games])),
        "nll": float(identity_means(row_losses(net, validation), validation["identity"]).mean()),
        "parameters_per_member": sum(p.numel() for p in net.members[0].parameters())} for name, net in networks.items()}
    selected = min(scores, key=lambda name: (-scores[name]["score"], scores[name]["nll"], scores[name]["parameters_per_member"]))
    write(output / "architecture-selection.json", {"selected": selected, "scores": scores, "sessions": validation_games})
    print(f"selected architecture: {selected}; {scores}", flush=True)
    ablations = ["no_errors", "no_context", "no_prior"] + (["no_shapes"] if selected == "attention" else [])
    for ablation in ablations:
        _, histories[ablation] = fit(study, training, validation, selected, ablation)
    _, histories["static"] = fit(study, training, validation, "static")
    if study.input_hashes() != inputs:
        raise RuntimeError("training inputs changed")
    write(output / "training.json", {"history": histories, "train_decisions": len(training["target"]),
        "validation_decisions": len(validation["target"]), "elapsed_seconds": time.monotonic() - started})
    artifacts = [*output.glob("*.pt"), output / "protocol.json", output / "training.json", output / "architecture-selection.json", output / "train.npz", output / "validation.npz"]
    write(output / "freeze.json", {"selected": selected, "ablations": ablations, "input_sha256": inputs,
        "artifacts": {str(p.resolve()): sha(p) for p in artifacts}})


def evaluate(study):
    output = study.args.output
    freeze = json.loads((output / "freeze.json").read_text())
    if (output / "evaluation-start.json").exists():
        raise ValueError("this frozen evaluation already started")
    def verify():
        if study.input_hashes() != freeze["input_sha256"] or any(sha(p) != h for p, h in freeze["artifacts"].items()):
            raise ValueError("frozen inputs changed")
    verify()
    write(output / "evaluation-start.json", {"freeze_sha256": sha(output / "freeze.json")})
    selected = freeze["selected"]
    networks = {name: load_ensemble(output, name) for name in PROTOCOL["architectures"]}
    networks.update({a: load_ensemble(output, selected, a) for a in freeze["ablations"]})
    networks.update({"static": load_ensemble(output, "static"), "pilot": study.prior,
        "untrained": Selector("small").eval(), "single_seed": networks[selected].members[0]})
    test_data = study.collect_selectors("test")
    losses = {name: identity_means(row_losses(net, test_data), test_data["identity"]) for name, net in networks.items()}
    actual = test_data["target"]
    base_losses = -np.log(np.maximum(test_data["baseline"][np.arange(len(actual)), actual], 1e-9))
    losses["translated"] = identity_means(base_losses, test_data["identity"])
    prediction = {name: {"nll": cluster_interval(values, PROTOCOL),
        "paired_vs_translated": cluster_interval(values - losses["translated"], PROTOCOL)} for name, values in losses.items()}
    rows = play(study, TEST_FAMILIES, PROTOCOL["test_play_seed"], PROTOCOL["test_play_identities"], networks, baselines=("translated", "old"))
    summaries = {"all": summarize(rows, "translated"), "excluded_families": summarize([r for r in rows if r["family"] in UNSEEN_FAMILIES], "translated"),
        "by_family": {f: summarize([r for r in rows if r["family"] == f], "translated") for f in TEST_FAMILIES}}
    controls = {name: {metric: cluster_interval([r["controllers"][selected][metric] - r["controllers"][name][metric] for r in rows], PROTOCOL)
        for metric in ("score", "late_score")} for name in (*freeze["ablations"], "static", "untrained", "pilot", "single_seed")}
    weight_by_family = {}
    with torch.no_grad():
        for fi, family in enumerate(TEST_FAMILIES):
            weights = []
            indices = np.flatnonzero(test_data["family"] == fi)
            for start in range(0, len(indices), 256):
                batch = indices[start:start + 256]
                weights.extend(networks[selected].weights(torch.tensor(test_data["gate"][batch]), torch.tensor(test_data["experts"][batch])).numpy())
            weight_by_family[family] = np.mean(weights, axis=0).tolist()
    verify()
    result = {"freeze_sha256": sha(output / "freeze.json"), "selected": selected,
        "prediction": prediction, "sessions": rows, "summary": summaries, "selected_minus_controls": controls,
        "expert_names": list(study.config.expert_names), "mean_selected_weights_on_test_corpus": weight_by_family,
        "parameter_counts": {name: sum(p.numel() for p in net.parameters()) for name, net in networks.items()},
        "primary_contrast": summaries["all"][selected]["paired_vs_baseline"],
        "scope": PROTOCOL["scope"], "promoted": False}
    write(output / "results.json", result)
    print(json.dumps({"selected": selected, "wins": {k: v["wins"] for k, v in summaries["all"].items()}, "primary_contrast": result["primary_contrast"], "controls": controls}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=Path("outputs/perfect-hal-bayes-v2/tablebase"))
    parser.add_argument("--selection", type=Path, default=Path("src/arena/config/translated_hal_v1_selection.json"))
    parser.add_argument("--prior", type=Path, default=Path("outputs/neural-pilots-v1"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(PROTOCOL["training_seeds"][0])
    study = Study(args)
    (train if args.mode == "train" else evaluate)(study)


if __name__ == "__main__":
    main()

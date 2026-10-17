"""Leap support prediction v1: the study runner of docs/PREDICTION.md.

Phases: ``keys`` prints the protocol's training keys, ``export`` runs the stl
exporter for every study key as a subprocess, ``study`` trains Net A on
residue rows and Net B on residue plus easy rows, scores both on the hidden
keys, and writes the report, and ``record`` copies the report into the
tracked results file that ``hal_lab.provenance`` verifies. This module
imports no ``stl`` module; it reads the exported ``.npz`` files alone.

Revision 2 of the protocol (2026-10-16) added the 59 success differences to
the inputs, widened the body to 256-256-256, and raised the step budget to
1,000,000. Revision 1 (61 inputs, 64-64, 20,000 steps) reached 3.0 % and
0.05 % exact match on the hidden keys; docs/PREDICTION.md records both.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
EXPORT = ROOT / "src" / "stl" / "outputs" / "leap-residue"
OUTPUT = ROOT / "src" / "hal_lab" / "outputs" / "leap_support_v1"
INPUTS = 120
RESULTS = Path(__file__).with_name("leap_support_v1_results.json")
BITS = 120

PROTOCOL = {
    "version": "leap_support_v1",
    "revision": 2,
    "seed": 20261016,
    "hidden_keys": ["H2_39", "H1_44"],
    # The H keys whose artifact record counts at least rows_per_key residue
    # classes (src/stl/outputs/leap-full-native/checkpoint.json). The other 63
    # H keys hold under a million residue classes each, 27 of them none.
    "candidate_keys": ["H1_36", "H1_38", "H1_40", "H1_44", "H1_45", "H1_48",
                       "H2_30", "H2_31", "H2_34", "H2_35", "H2_37", "H2_38",
                       "H2_39", "H2_41", "H2_42", "H2_43", "H2_46", "H2_47"],
    "training_keys_per_half": 4,
    "rows_per_key": 1_000_000,
    "validation_fraction": 0.05,
    "features": "the 60 success payoffs, the failure payoff, and the 59 success differences",
    "hidden_layers": [256, 256, 256],
    "batch_size": 4096,
    "steps": 1_000_000,
    "validation_every": 50_000,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "gates": {"hidden_key_exact_match": 0.90, "parameter_cap": 1_000_000, "training_hours_cap": 2.0},
    "experiment2_margin_points": 2.0,
}


def protocol_hash() -> str:
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_keys(protocol: dict = PROTOCOL) -> list[str]:
    """We draw the training keys per half from the non-hidden candidates with the seed."""
    rng = np.random.default_rng([protocol["seed"], 1])
    hidden = set(protocol["hidden_keys"])
    chosen = []
    for half in ("H1", "H2"):
        pool = sorted(k for k in protocol["candidate_keys"] if k.startswith(half) and k not in hidden)
        chosen += sorted(rng.choice(pool, protocol["training_keys_per_half"], replace=False).tolist())
    assert not set(chosen) & hidden
    return chosen


def load_rows(path: Path):
    """We return the 120 inputs, the 120 label bits, and the residue flag of one export.

    The inputs are the 60 success payoffs, the failure payoff, and the 59
    success differences. The hole of a residue support sits at the kink, the
    most negative difference, which a network reads from the differences far
    more easily than from the payoffs.
    """
    with np.load(path) as d:
        success = d["success"]
        x = np.concatenate([success, d["failure"][:, None], np.diff(success, axis=1)], axis=1).astype(np.float32)
        shifts = np.arange(60, dtype=np.uint64)
        bits = [(d[m][:, None] >> shifts) & np.uint64(1) for m in ("p_mask", "q_mask")]
        y = np.concatenate(bits, axis=1).astype(np.float32)
        return x, y, d["residue"].astype(bool)


def build_model(torch, mean, std, hidden):
    layers, width = [], INPUTS
    for units in hidden:
        layers += [torch.nn.Linear(width, units), torch.nn.ReLU()]
        width = units
    layers.append(torch.nn.Linear(width, BITS))
    model = torch.nn.Sequential(*layers)
    model.mean = torch.as_tensor(mean)
    model.std = torch.as_tensor(std)
    return model


def exact_match(torch, model, x, y, device, batch=65_536):
    """We return the exact-match rate, the per-bit accuracy, and the wrong-bit histogram."""
    model.eval()
    wrong = []
    with torch.no_grad():
        for lo in range(0, len(x), batch):
            xb = (torch.as_tensor(x[lo:lo + batch], device=device) - model.mean.to(device)) / model.std.to(device)
            predicted = (model(xb) > 0).float()
            wrong.append((predicted != torch.as_tensor(y[lo:lo + batch], device=device)).sum(dim=1).cpu().numpy())
    wrong = np.concatenate(wrong)
    histogram = np.bincount(wrong.astype(np.int64), minlength=8)[:8].tolist()
    return {"rows": int(len(wrong)), "exact_match": float(np.mean(wrong == 0)),
            "bit_accuracy": float(1.0 - wrong.mean() / BITS), "wrong_bits_histogram_0_to_7": histogram}


def train(torch, x, y, protocol, seed, device, log):
    """We train one net for the protocol's step budget and keep the best validation weights."""
    rng = np.random.default_rng([protocol["seed"], seed])
    torch.manual_seed(int(rng.integers(2**31)))
    order = rng.permutation(len(x)); split = int(len(x) * protocol["validation_fraction"])
    valid, fit = order[:split], order[split:]
    mean, std = x[fit].mean(axis=0), x[fit].std(axis=0) + 1e-6
    model = build_model(torch, mean, std, protocol["hidden_layers"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    xf = torch.as_tensor((x[fit] - mean) / std, device=device)
    # Labels stay uint8 on the device; Net B holds 16 million rows of 120 bits.
    yf = torch.as_tensor(y[fit].astype(np.uint8), device=device)
    best, best_state, started = -1.0, None, time.perf_counter()
    batch = protocol["batch_size"]
    for step in range(1, protocol["steps"] + 1):
        model.train()
        idx = torch.as_tensor(rng.integers(len(fit), size=batch), device=device)
        optimizer.zero_grad()
        loss = loss_fn(model(xf[idx]), yf[idx].float())
        loss.backward(); optimizer.step()
        if step % protocol["validation_every"] == 0 or step == protocol["steps"]:
            score = exact_match(torch, model, x[valid], y[valid], device)["exact_match"]
            log(f"step {step} loss {loss.item():.4f} validation exact match {score:.4f}")
            if score > best:
                best = score; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    seconds = time.perf_counter() - started
    return model, {"validation_exact_match": best, "training_seconds": seconds,
                   "parameters": int(sum(p.numel() for p in model.parameters()))}


def save_weights(torch, model, path: Path):
    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(path, mean=model.mean.numpy(), std=model.std.numpy(), **arrays)


def repo_path(path: Path) -> str:
    """We key a bound file by repository path; a file outside the repository keeps its absolute path."""
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def study(export: Path, output: Path, protocol: dict = PROTOCOL, log=print) -> dict:
    """We run both experiments of docs/PREDICTION.md and write the report.

    The report's ``sha256`` table keys every bound file by repository path, so
    ``hal_lab.provenance`` verifies each one as bytes.
    """
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    keys = training_keys(protocol); hidden = list(protocol["hidden_keys"])
    files = {k: export / f"{k}.npz" for k in keys + hidden}
    missing = [k for k, p in files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"export first: {missing}")
    rows = {k: load_rows(p) for k, p in files.items()}
    assert not any(k in hidden for k in keys)
    x_res = np.concatenate([rows[k][0][rows[k][2]] for k in keys]); y_res = np.concatenate([rows[k][1][rows[k][2]] for k in keys])
    x_easy = np.concatenate([rows[k][0][~rows[k][2]] for k in keys]); y_easy = np.concatenate([rows[k][1][~rows[k][2]] for k in keys])
    output.mkdir(parents=True, exist_ok=True)
    report = {"protocol": protocol, "protocol_sha256": protocol_hash(), "device": device,
              "training_keys": keys, "hidden_keys": hidden,
              "sha256": {repo_path(p): sha256(p) for p in files.values()},
              "rows": {"residue_training": int(len(x_res)), "easy_training": int(len(x_easy)),
                       **{f"hidden_{k}": int(rows[k][2].sum()) for k in hidden}},
              "nets": {}}
    for name, (x, y) in (("A", (x_res, y_res)),
                         ("B", (np.concatenate([x_res, x_easy]), np.concatenate([y_res, y_easy])))):
        log(f"net {name}: {len(x):,} training rows on {device}")
        model, summary = train(torch, x, y, protocol, 1, device, log)
        weights = output / f"net_{name}.npz"; save_weights(torch, model, weights)
        summary["weights"] = repo_path(weights); report["sha256"][repo_path(weights)] = sha256(weights)
        summary["hidden"] = {k: exact_match(torch, model, rows[k][0][rows[k][2]], rows[k][1][rows[k][2]], device) for k in hidden}
        report["nets"][name] = summary
    gates = protocol["gates"]; a = report["nets"]["A"]
    report["gate_results"] = {
        "hidden_key_exact_match": {k: a["hidden"][k]["exact_match"] >= gates["hidden_key_exact_match"] for k in hidden},
        "parameter_cap": a["parameters"] <= gates["parameter_cap"],
        "training_hours_cap": a["training_seconds"] <= gates["training_hours_cap"] * 3600,
    }
    report["experiment1_passed"] = all(report["gate_results"]["hidden_key_exact_match"].values()) and \
        report["gate_results"]["parameter_cap"] and report["gate_results"]["training_hours_cap"]
    margin = protocol["experiment2_margin_points"] / 100
    diff = {k: report["nets"]["B"]["hidden"][k]["exact_match"] - a["hidden"][k]["exact_match"] for k in hidden}
    verdict = "easy stages hurt" if all(d < -margin for d in diff.values()) else \
        "easy stages help" if all(d > margin for d in diff.values()) else "no measurable effect at this budget"
    report["experiment2"] = {"b_minus_a_exact_match": diff, "verdict": verdict}
    for source in (Path(__file__), ROOT / "src" / "stl" / "experiments" / "export_leap_residue.py"):
        report["sha256"][repo_path(source)] = sha256(source)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def record(output: Path, results: Path = RESULTS) -> dict:
    """We copy the report into the tracked results file, binding the report by hash."""
    path = output / "report.json"
    frozen = {**json.loads(path.read_text()), "report_sha256": {repo_path(path): sha256(path)}}
    results.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["keys", "export", "study", "record"])
    parser.add_argument("--export", type=Path, default=EXPORT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--rows", type=int, default=PROTOCOL["rows_per_key"])
    args = parser.parse_args(argv)
    keys = training_keys()
    if args.phase == "keys":
        print(json.dumps({"training_keys": keys, "hidden_keys": PROTOCOL["hidden_keys"]}))
    elif args.phase == "export":
        pending = [k for k in keys + PROTOCOL["hidden_keys"] if not (args.export / f"{k}.npz").exists()]
        if pending:
            subprocess.run([sys.executable, "-m", "stl.experiments.export_leap_residue", "--keys", *pending,
                            "--rows", str(args.rows), "--seed", str(PROTOCOL["seed"]), "--output", str(args.export)],
                           check=True, cwd=ROOT)
    elif args.phase == "record":
        record(args.output)
    else:
        report = study(args.export, args.output)
        print(json.dumps({k: report[k] for k in ("gate_results", "experiment1_passed", "experiment2")}, indent=2))


if __name__ == "__main__":
    main()

"""Correct the shuffle control on fresh seeds while retaining the first report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from arena.policies.run_neural_pilots import Experiment, PROTOCOL, TEST_FAMILIES, load_network, sha, summarize, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Parent experiment directory")
    parser.add_argument("--artifact", type=Path, default=Path("outputs/perfect-hal-bayes-v2/tablebase"))
    parser.add_argument("--selection", type=Path, default=Path("src/arena/config/translated_hal_v1_selection.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    experiment = Experiment(args)
    freeze = json.loads((args.output / "freeze.json").read_text())
    runner = Path(__file__).with_name("run_neural_pilots.py").resolve()
    original = runner.read_text().replace("        ablation_rng = np.random.default_rng(seed + 349)\n", "").replace("ablation=ablation, rng=ablation_rng)", "ablation=ablation, rng=rng)")
    import hashlib
    for path, expected in freeze["input_sha256"].items():
        actual = sha(path)
        if actual != expected and not (Path(path) == runner and hashlib.sha256(original.encode()).hexdigest() == expected):
            raise ValueError("changes exceed the isolated shuffle RNG repair")
    if any(sha(p) != h for p, h in freeze["artifacts"].items()):
        raise ValueError("selected artifacts changed")
    output = args.output / "memory-audit"
    output.mkdir(exist_ok=False)
    inputs = {**experiment.input_hashes(), str(Path(__file__).resolve()): sha(__file__)}
    write(output / "protocol.json", {"schema": "arena-neural-memory-audit-v1", "seed_base": 101100000,
        "identities_per_family": 8, "families": TEST_FAMILIES, "games_per_identity": PROTOCOL["games_per_session"],
        "parent_freeze_sha256": sha(args.output / "freeze.json"), "parent_results_sha256": sha(args.output / "results.json"),
        "input_sha256": inputs, "reason": "The first shuffle control consumed the action RNG. Separate RNG streams. Preserve original results; hold model weights fixed; score fresh identities once."})
    network = load_network(args.output, "transformer")
    rows = []
    for fi, family in enumerate(TEST_FAMILIES):
        for identity in range(8):
            seed = 101100000 + fi * 1000 + identity
            rows.append({"family": family, "seed": seed, "controllers": {
                name: experiment.session(family, seed, "transformer", network, ablation=ablation)
                for name, ablation in (("transformer", None), ("reset", "reset"), ("shuffle", "shuffle"))}})
        print(f"memory audit {family}", flush=True)
    if any(sha(p) != h for p, h in inputs.items()) or any(sha(p) != h for p, h in freeze["artifacts"].items()):
        raise RuntimeError("audit inputs changed")
    result = {"protocol_sha256": sha(output / "protocol.json"), "sessions": rows, "summary": summarize(rows, "transformer"),
        "interpretation": "Compare neural window order and retention; retain the current token and statistical memory. No model selection or human claim."}
    write(output / "results.json", result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

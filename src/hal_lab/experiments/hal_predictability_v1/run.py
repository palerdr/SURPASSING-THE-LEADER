"""Hal predictability v1: how much public information predicts Hal's second.

Phases: ``generate`` records Hal's decisions per (policy, opponent) pair, ``fit``
scores departure and three predictors and applies the control, ``exploit`` plays
a counter-player against translated-v1, and ``report`` writes the tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

import arena.dth_adapter, arena.match, arena.policies.adaptive, arena.policies.perfect_hal  # noqa: E401
import arena.policies.translated_hal, arena.translated_hal_adapter, dth.agent  # noqa: E401
from arena.contracts import observe_provider, reset_provider_game
from arena.dth_adapter import project_to_dth_state
from arena.match import play_match_game
from arena.policies import registry
from hal_lab.cli import build_parser
from hal_lab.harness.baselines import ExactProvider, UniformProvider

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "src" / "hal_lab" / "outputs" / "hal_predictability_v1"
ARTIFACT = ROOT / "src" / "dth" / "artifacts" / "complete_full_v1"

PROTOCOL = {
    "version": "hal_predictability_v1", "revision": 2, "seed": 20261019,
    # Registry names and flags, parsed by the `python -m hal_lab match` parser.
    "policies": {
        "dth": ["--candidate", "dth"],
        "translated-v1": ["--candidate", "perfect-hal", "--perfect-hal-model", "translated-v1"],
        "adaptive-uniform": ["--candidate", "adaptive-dth"],
    },
    "opponents": ["uniform", "exact"], "games": 1000, "start_clock": 720, "max_half_rounds": 200,
    # The floor mixes in uniform mass, so a second off the equilibrium support stays finite.
    "k": 4, "test_fraction": 0.2, "validation_fraction": 0.1, "floor_uniform_mix": 1e-3,
    "laplace": 1.0, "l2_grid": [1e-4, 1e-3, 1e-2], "lbfgs_max_iter": 5000, "resamples": 1000,
    # The counter-player reads the state-only table of this pair's fit games.
    "exploit": {"games": 500, "table": "translated-v1__exact"},
}


def protocol_hash(protocol: dict) -> str:
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def game_seeds(protocol: dict, stream: int = 0) -> list[int]:
    """Derive one seed per game; stream 0 feeds generate and stream 1 feeds exploit."""
    games = protocol["games"] if stream == 0 else protocol["exploit"]["games"]
    return [int(s) for s in np.random.SeedSequence([protocol["seed"], stream]).generate_state(games)]


def test_games(protocol: dict) -> list[int]:
    """Draw the held-out game indices from the protocol seed."""
    return sorted(int(g) for g in np.random.default_rng(protocol["seed"]).permutation(protocol["games"])
                  [: round(protocol["games"] * protocol["test_fraction"])])


def _bucket(state, checker) -> tuple:
    return (*(min(v // w, top) for v, w, top in zip(state, (30, 60, 30, 60), (9, 5, 9, 5))), int(checker))


class Recorder:
    """Forward a provider and record each decision with its public context."""

    def __init__(self, inner, floor_agent, k, base):
        self.inner, self.floor_agent, self.k, self.rows, self.base = inner, floor_agent, k, [], base
        self.counts, self.opp_counts = [0] * 60, {"dropper": [0] * 60, "checker": [0] * 60}
        self.own, self.opp, self.pending, self.game = [], [], None, -1

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def reset_game(self):
        self.own, self.opp, self.game = [], [], self.game + 1
        reset_provider_game(self.inner)

    def policy(self, decision):
        state = project_to_dth_state(decision)
        move = self.floor_agent.decide(state)  # decide raises on a state that it rejects
        floor = move.drop_policy if decision.role == "dropper" else move.check_policy
        self.pending = (decision.role, decision.actor_name, state, floor)
        return self.inner.policy(decision)

    def observe(self, record):
        role, seat, state, floor = self.pending
        own, opp = (record.drop_time, record.check_time) if role == "dropper" else (record.check_time, record.drop_time)
        opp_role = "checker" if role == "dropper" else "dropper"
        pad = lambda xs: ([0] * self.k + xs)[-self.k :][::-1]  # noqa: E731
        self.rows.append({**self.base, "game": self.game, "move": record.half_round_index,
                          "seat": seat, "role": role, "state": list(state), "own": pad(self.own),
                          "opp": pad(self.opp), "counts": list(self.counts),
                          "opp_counts": list(self.opp_counts[opp_role]),
                          "second": int(own), "floor": [float(p) for p in floor]})
        self.own, self.opp, self.pending = [*self.own, int(own)], [*self.opp, int(opp)], None
        self.counts[int(own) - 1] += 1
        self.opp_counts[opp_role][int(opp) - 1] += 1
        observe_provider(self.inner, record)


def _make_hal(protocol: dict, flags: list[str]):
    args = build_parser().parse_args(["match", *flags, "--opponent", "dth", "--pure-dth", "--output", "-",
                                      "--seed", str(protocol["seed"]), "--dth-complete-tablebase", str(ARTIFACT)])
    return registry.make_provider(args.candidate, args)


def _play(hal, rival, seeds, protocol) -> list[bool]:
    """Seat Hal as Hal in even games and as Baku in odd games; return the rival's wins."""
    wins = []
    for index, seed in enumerate(seeds):
        one, two = (hal, rival) if index % 2 == 0 else (rival, hal)
        winner, _ = play_match_game(one, two, seed=seed, start_clock=protocol["start_clock"],
                                    max_half_rounds=protocol["max_half_rounds"], game_index=index, pure_dth=True)
        wins.append(winner == ("Baku" if index % 2 == 0 else "Hal"))
    return wins


def generate(protocol: dict, out: Path = OUTPUT) -> None:
    """Play every series and write its decisions and manifest."""
    floor_agent = dth.agent.CompleteDTHAgent(ARTIFACT)
    seeds, moves_dir = game_seeds(protocol), out / "moves"
    moves_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol_hash": protocol_hash(protocol), "seeds": seeds,
                "test_games": test_games(protocol), "pairs": {}}
    for policy, flags in protocol["policies"].items():
        for opponent in protocol["opponents"]:
            hal = Recorder(_make_hal(protocol, flags), floor_agent, protocol["k"],
                           {"policy": policy, "opponent": opponent})
            _play(hal, UniformProvider() if opponent == "uniform" else ExactProvider(floor_agent), seeds, protocol)
            path, rows = moves_dir / f"{policy}__{opponent}.jsonl", hal.rows
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            manifest["pairs"][f"{policy}__{opponent}"] = {"moves": len(rows), "games": len(seeds),
                "hal_seat": "even game index", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            print(f"{policy} vs {opponent}: {len(rows)} moves", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


def _load(out: Path, pair: str) -> tuple[list[dict], list[int]]:
    """Return a pair's recorded moves and the manifest's test games."""
    rows = [json.loads(line) for line in (out / "moves" / f"{pair}.jsonl").read_text().splitlines()]
    return rows, json.loads((out / "manifest.json").read_text())["test_games"]


def _split(rows, protocol, test):
    games = np.array([r["game"] for r in rows])
    is_test = np.isin(games, sorted(test))
    order = np.random.default_rng(protocol["seed"] + 1).permutation(np.unique(games[~is_test]))
    is_val = ~is_test & np.isin(games, order[: max(1, round(len(order) * protocol["validation_fraction"]))])
    return games, is_test, is_val, ~is_test & ~is_val


def _state_table(rows, mask) -> dict[tuple, np.ndarray]:
    table: dict[tuple, np.ndarray] = {}
    for r in (r for r, m in zip(rows, mask) if m):
        table.setdefault(_bucket(r["state"], r["role"] == "checker"), np.zeros(60))[r["second"] - 1] += 1
    return table


def _features(rows, protocol):
    k, mix = protocol["k"], protocol["floor_uniform_mix"]
    floor = (1 - mix) * np.array([r["floor"] for r in rows]) + mix / 60
    buckets = np.array([_bucket(r["state"], r["role"] == "checker") for r in rows])
    hist = np.array([r["own"][:k] + r["opp"][:k] for r in rows])
    counts = [np.array([r[key] for r in rows], dtype=float) for key in ("counts", "opp_counts")]
    counts = [c / np.maximum(c.sum(1, keepdims=True), 1.0) for c in counts]
    one_hot = [np.eye(61)[hist[:, j]] for j in range(2 * k)]
    one_hot += [np.eye(n)[buckets[:, j]] for j, n in enumerate((10, 6, 10, 6, 2))]
    x = np.concatenate([np.log(floor), *one_hot, *counts], axis=1)
    return np.log(floor), x, buckets, np.array([r["second"] - 1 for r in rows])


def _history_model(logf, x, y, fit_rows, val, protocol):
    """Fit logits = log floor + z W + b by L-BFGS per L2 weight; keep the best on validation."""
    std = x[fit_rows].std(0)
    z = torch.as_tensor((x - x[fit_rows].mean(0)) / np.where(std > 0, std, 1.0))
    lf, yy, fr, vr = (torch.as_tensor(a) for a in (logf, y, fit_rows, val))
    grid, best = [], None
    for l2 in protocol["l2_grid"]:
        w, b = (torch.zeros(*shape, dtype=torch.float64, requires_grad=True) for shape in ((z.shape[1], 60), (60,)))
        opt = torch.optim.LBFGS([w, b], max_iter=protocol["lbfgs_max_iter"], tolerance_grad=1e-9,
                                tolerance_change=1e-12, history_size=50, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(lf[fr] + z[fr] @ w + b, yy[fr]) + l2 * w.pow(2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            logp = torch.log_softmax(lf + z @ w + b, dim=1).numpy()
        grid.append({"l2": l2, "val_loss": float(-logp[val, y[val]].mean()),
                     "iterations": int(opt.state[w]["n_iter"])})
        if best is None or grid[-1]["val_loss"] < best[0]:
            best = (grid[-1]["val_loss"], l2, logp)
    return best[2], {"l2": best[1], "grid": grid}


def _score_pair(rows, test, protocol):
    logf, x, buckets, y = _features(rows, protocol)
    games, is_test, is_val, is_fit = _split(rows, protocol, test)
    table = _state_table(rows, is_fit)
    state_p = np.array([table.get(tuple(b), np.zeros(60)) + protocol["laplace"] for b in buckets])
    state_p /= state_p.sum(1, keepdims=True)
    hist_logp, fit_info = _history_model(logf, x, y, np.flatnonzero(is_fit), np.flatnonzero(is_val), protocol)
    idx = np.flatnonzero(is_test)
    preds = {"floor": logf, "state": np.log(state_p), "history": hist_logp}
    nll = {name: -lp[idx, y[idx]] for name, lp in preds.items()}
    top1 = {name: lp[idx].argmax(1) == y[idx] for name, lp in preds.items()}
    off = int(sum(r["floor"][r["second"] - 1] == 0.0 for r in rows))
    return {"nll": nll, "top1": top1, "departure": nll["floor"] + (np.exp(logf) * logf).sum(1)[idx], "games": games[idx],
            "moves": len(rows), "off_support": off, "history_fit": fit_info}


def _summary(parts, protocol):
    """Pool scored pairs and bootstrap departure and leak over held-out games."""
    nll = {n: np.concatenate([p["nll"][n] for p in parts]) for n in ("floor", "state", "history")}
    top1 = {n: np.concatenate([p["top1"][n] for p in parts]) for n in nll}
    games = np.concatenate([p["games"] + 10**7 * i for i, p in enumerate(parts)])
    count = np.bincount(gid := np.unique(games, return_inverse=True)[1])
    draw = np.random.default_rng(protocol["seed"]).integers(0, len(count), (protocol["resamples"], len(count)))
    def boot(values):
        sums = np.bincount(gid, values)
        lo, hi = np.percentile(sums[draw].sum(1) / count[draw].sum(1), [2.5, 97.5])
        return float(sums.sum() / count.sum()), [float(lo), float(hi)]
    departure, departure_ci = boot(np.concatenate([p["departure"] for p in parts]))
    leak, leak_ci = boot(nll["floor"] - nll["history"])
    return {"moves": int(sum(p["moves"] for p in parts)), "test_moves": int(len(gid)),
            "test_games": int(len(count)), "off_support": int(sum(p["off_support"] for p in parts)),
            "history_fits": [p["history_fit"] for p in parts], "departure": departure, "departure_ci": departure_ci,
            **{f"{n}_loss": float(v.mean()) for n, v in nll.items()},
            **{f"{n}_top1": float(v.mean()) for n, v in top1.items()}, "leak": leak, "leak_ci": leak_ci}


def check_control(results: dict) -> None:
    """Fail when the exact policy's departure interval excludes zero."""
    for name, row in results["rows"].items():
        lo, hi = row["departure_ci"]
        if row["policy"] == "dth" and not lo <= 0.0 <= hi:
            raise RuntimeError(f"control failed: exact policy {name} departure {row['departure']:.4f} nats, "
                               f"95% interval [{lo:.4f}, {hi:.4f}] excludes 0")


def fit(protocol: dict, out: Path = OUTPUT) -> dict:
    """Score the predictors per pair and pooled per policy."""
    if json.loads((out / "manifest.json").read_text())["protocol_hash"] != protocol_hash(protocol):
        raise RuntimeError("moves were generated under another protocol; rerun generate")
    results: dict = {"rows": {}}
    for policy in protocol["policies"]:
        parts = []
        for opponent in protocol["opponents"]:
            parts.append(_score_pair(*_load(out, f"{policy}__{opponent}"), protocol))
            results["rows"][f"{policy}__{opponent}"] = {"policy": policy, "opponent": opponent, **_summary(parts[-1:], protocol)}
        results["rows"][f"{policy}__pooled"] = {"policy": policy, "opponent": "pooled", **_summary(parts, protocol)}
    (out / "fit.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    check_control(results)
    return results


class Counter:
    """Best-respond to the state-only forecast of Hal's second in the exact stage game."""

    def __init__(self, agent, table, laplace):
        self.agent, self.table, self.laplace = agent, table, laplace

    def policy(self, decision):
        state = project_to_dth_state(decision)
        stage = self.agent.stage_game(state)
        hal_checker = decision.role == "dropper"
        counts = self.table.get(_bucket(state, hal_checker))
        forecast = ((stage.check_policy if hal_checker else stage.drop_policy) if counts is None
                    else (counts + self.laplace) / (counts.sum() + 60 * self.laplace))
        # Dropper rows maximize the matrix; Checker columns minimize it.
        action = np.argmax(stage.matrix @ forecast) if hal_checker else np.argmin(forecast @ stage.matrix)
        return {int(action) + 1: 1.0}


def _wilson(wins: int, n: int, z: float = 1.959964) -> list[float]:
    p, scale = wins / n, 1 + z * z / n
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return [(p + z * z / (2 * n)) / scale - half, (p + z * z / (2 * n)) / scale + half]


def exploit(protocol: dict, out: Path = OUTPUT) -> dict:
    """Play the counter-player and the exact opponent against fresh translated-v1 series."""
    agent, pair = dth.agent.CompleteDTHAgent(ARTIFACT), protocol["exploit"]["table"]
    rows, test = _load(out, pair)
    table = _state_table(rows, _split(rows, protocol, test)[3])
    seeds, result = game_seeds(protocol, stream=1), {"table": pair, "rows": {}}
    for name, rival in (("counter", Counter(agent, table, protocol["laplace"])), ("exact", ExactProvider(agent))):
        wins = _play(_make_hal(protocol, protocol["policies"]["translated-v1"]), rival, seeds, protocol)
        result["rows"][name] = {"games": len(wins), "wins": sum(wins), "win_rate": sum(wins) / len(wins),
                                "wilson_95": _wilson(sum(wins), len(wins))}
    (out / "exploit.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


def _sources() -> dict:
    def digest(*paths):
        return {str(Path(p).resolve().relative_to(ROOT)): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    shared = (arena.match.__file__, registry.__file__)
    return {"dth": digest(*shared, arena.dth_adapter.__file__, dth.agent.__file__),
            "translated-v1": digest(*shared, arena.translated_hal_adapter.__file__,
                                    arena.policies.perfect_hal.__file__, arena.policies.translated_hal.__file__,
                                    arena.translated_hal_adapter.SELECTION),
            "adaptive-uniform": digest(*shared, arena.policies.adaptive.__file__)}


def report(protocol: dict, out: Path = OUTPUT) -> str:
    """Write results.json and results.md and return the markdown tables."""
    results = json.loads((out / "fit.json").read_text())
    check_control(results)
    played = json.loads((out / "exploit.json").read_text())
    ci = lambda v, c: f"{v:+.4f} [{c[0]:+.4f}, {c[1]:+.4f}]"  # noqa: E731
    lines = ["| Policy | Opponent | Moves (test) | Off-support | Departure [95% CI] | Floor | State-only | History | L2 | Top-1 floor / state / history |",
             "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |"]
    for r in results["rows"].values():
        lines.append(f"| {r['policy']} | {r['opponent']} | {r['moves']} ({r['test_moves']}) | {r['off_support']} | "
                     f"{ci(r['departure'], r['departure_ci'])} | {r['floor_loss']:.4f} | {r['state_loss']:.4f} | "
                     f"{r['history_loss']:.4f} | {', '.join(f"{f['l2']:g}" for f in r['history_fits'])} | {r['floor_top1']:.3f} / "
                     f"{r['state_top1']:.3f} / {r['history_top1']:.3f} |")
    lines += ["", "| Policy | Opponent | Leak, floor minus history [95% CI] |", "| --- | --- | --- |"]
    lines += [f"| {r['policy']} | {r['opponent']} | {ci(r['leak'], r['leak_ci'])} |" for r in results["rows"].values()]
    lines += ["", f"| Player vs translated-v1 (table {played['table']}) | Wins | Win rate [Wilson 95%] |", "| --- | ---: | --- |"]
    lines += [f"| {name} | {r['wins']} of {r['games']} | {r['win_rate']:.3f} [{r['wilson_95'][0]:.3f}, "
              f"{r['wilson_95'][1]:.3f}] |" for name, r in played["rows"].items()]
    table = "\n".join(lines) + "\n"
    record = {"protocol": protocol, "protocol_hash": protocol_hash(protocol),
              "dth_manifest_sha256": hashlib.sha256((ARTIFACT / "tablebase.json").read_bytes()).hexdigest(),
              "sources": _sources(), "control": "passed", "exploit": played, **results}
    (out / "results.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    (out / "results.md").write_text(f"Losses and departure in nats per move on held-out games.\n\n{table}", encoding="utf-8")
    print(table, end="")
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hal_lab.experiments.hal_predictability_v1.run")
    parser.add_argument("phase", choices=("generate", "fit", "exploit", "report"))
    {"generate": generate, "fit": fit, "exploit": exploit, "report": report}[parser.parse_args(argv).phase](PROTOCOL)
    return 0


if __name__ == "__main__":
    sys.exit(main())

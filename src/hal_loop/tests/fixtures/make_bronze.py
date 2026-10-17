"""Write a deterministic bronze fixture in the hosted ledger's row shape.

Twelve finished games from five players, one stopped game, one active game,
and one finished leap-window game, as `GET /rest/v1/games` returns them. In the
leap-window game's 61-second turn, Baku drops at 61 and Hal checks at 60.
Run it from the repository root to regenerate the fixture:

    uv run python src/hal_loop/tests/fixtures/make_bronze.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("bronze") / "games" / "2026-10-13.jsonl"
SEED = 20261017


def hal_second(rng):
    return int(rng.integers(1, 61))


def game(rng, player, series, index, finished=True, stopped=False):
    moves, half, clock = [], 1, 720
    hal = {"name": "Hal", "deaths": 0, "ttd_seconds": 0, "cylinder_seconds": 0}
    baku = {"name": "Baku", "deaths": 0, "ttd_seconds": 0, "cylinder_seconds": 0}
    rounds = int(rng.integers(2, 9))
    for move_index in range(rounds):
        dropper, checker = ("Hal", "Baku") if half == 1 else ("Baku", "Hal")
        drop, check = hal_second(rng) if dropper == "Hal" else int(rng.choice([1, 5, 30, 44, 60])), hal_second(rng) if checker == "Hal" else int(rng.choice([1, 5, 44, 60]))
        before = {"half": half, "round": move_index // 2 + 1, "players": [dict(hal), dict(baku)], "clock_display": "8:12:00 AM", "clock_seconds": clock, "turn_duration": 60}
        if check >= drop:
            result, survived, squandered = "check_success", None, check - drop + 1
            (hal if checker == "Hal" else baku)["cylinder_seconds"] += squandered
        else:
            result, survived, squandered = "check_fail_survived", True, 0
            victim = baku if checker == "Baku" else hal
            victim["deaths"] += 1; victim["ttd_seconds"] += 60
        moves.append({"result": result, "checker": checker, "dropper": dropper, "survived": survived, "drop_second": drop, "check_second": check,
                      "squandered_seconds": squandered, "public_state_before": before, "survival_probability": 0.95 if survived else None,
                      "death_duration_seconds": 60 if survived else 0})
        half = 2 if half == 1 else 1; clock += 300
    status = "finished" if finished else ("stopped" if stopped else "active")
    human_won = bool(rng.integers(0, 2)) if finished else None
    return {"series_id": hashlib.md5(f"{series}".encode()).hexdigest(), "game_index": index, "player_id": hashlib.sha256(f"player-{player}".encode()).hexdigest(),
            "status": status, "human_won": human_won, "score": (float(rng.integers(1, 300)) if human_won else None), "half_rounds": len(moves),
            "human_cylinder_seconds": str(float(baku["cylinder_seconds"])), "human_ttd_seconds": str(float(baku["ttd_seconds"])), "human_deaths": baku["deaths"],
            "hal_cylinder_seconds": str(float(hal["cylinder_seconds"])), "hal_ttd_seconds": str(float(hal["ttd_seconds"])), "hal_deaths": hal["deaths"],
            "start_clock": 720, "pure_dth": False, "hal_agent": "translated-v1", "code_version": "ad1880bfeec91a5c8c150db8c6cd26a92497bedaed109f75a410827b7e968ffb",
            "game_seed": str(int(rng.integers(1, 2**63))), "policy_seed": str(int(rng.integers(1, 2**63))),
            "started_at": f"2026-10-13 {10 + series:02d}:{index * 5:02d}:00.000000+00", "updated_at": f"2026-10-13 {10 + series:02d}:{index * 5 + 4:02d}:00.000000+00",
            "public_history": moves}


def leap_game(rng, player, series, index):
    """Return a finished game whose second half-round falls in the leap window."""
    hal = {"name": "Hal", "deaths": 0, "ttd_seconds": 0, "cylinder_seconds": 0}
    baku = {"name": "Baku", "deaths": 0, "ttd_seconds": 0, "cylinder_seconds": 0}
    # (half, dropper, checker, drop, check, clock, turn); the leap window opens at clock 3540.
    plan = [(1, "Hal", "Baku", 20, 30, 3240, 60), (2, "Baku", "Hal", 61, 60, 3540, 61), (1, "Hal", "Baku", 10, 44, 3841, 60)]
    moves = []
    for move_index, (half, dropper, checker, drop, check, clock, turn) in enumerate(plan):
        before = {"half": half, "round": move_index // 2 + 1, "players": [dict(hal), dict(baku)], "clock_display": "8:59:00 AM",
                  "clock_seconds": clock, "turn_duration": turn}
        victim = hal if checker == "Hal" else baku
        if check >= drop:
            result, survived, squandered = "check_success", None, check - drop + 1
            victim["cylinder_seconds"] += squandered
        else:
            result, survived, squandered = "check_fail_survived", True, 0
            victim["deaths"] += 1; victim["ttd_seconds"] += 60
        moves.append({"result": result, "checker": checker, "dropper": dropper, "survived": survived, "drop_second": drop, "check_second": check,
                      "squandered_seconds": squandered, "public_state_before": before, "survival_probability": 0.95 if survived else None,
                      "death_duration_seconds": 60 if survived else 0})
    return {"series_id": hashlib.md5(f"{series}".encode()).hexdigest(), "game_index": index, "player_id": hashlib.sha256(f"player-{player}".encode()).hexdigest(),
            "status": "finished", "human_won": False, "score": None, "half_rounds": len(moves),
            "human_cylinder_seconds": str(float(baku["cylinder_seconds"])), "human_ttd_seconds": str(float(baku["ttd_seconds"])), "human_deaths": baku["deaths"],
            "hal_cylinder_seconds": str(float(hal["cylinder_seconds"])), "hal_ttd_seconds": str(float(hal["ttd_seconds"])), "hal_deaths": hal["deaths"],
            "start_clock": 3240, "pure_dth": False, "hal_agent": "translated-v1", "code_version": "ad1880bfeec91a5c8c150db8c6cd26a92497bedaed109f75a410827b7e968ffb",
            "game_seed": str(int(rng.integers(1, 2**63))), "policy_seed": str(int(rng.integers(1, 2**63))),
            "started_at": f"2026-10-13 {10 + series:02d}:{index * 5:02d}:00.000000+00", "updated_at": f"2026-10-13 {10 + series:02d}:{index * 5 + 4:02d}:00.000000+00",
            "public_history": moves}


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    series = 0
    for player in range(5):
        for index in range(3 if player < 2 else 2):
            rows.append(game(rng, player, series, index)); 
        series += 1
    rows.append(game(rng, 1, 9, 0, finished=False, stopped=True))
    rows.append(game(rng, 4, 10, 0, finished=False))
    rows.append(leap_game(rng, 0, 11, 0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()

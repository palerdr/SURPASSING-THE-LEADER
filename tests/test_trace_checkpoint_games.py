import os
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.trace_checkpoint_games as trace_module
from training.strength.match_gate import _opponent_seed


class _FakeOpponent:
    def choose_action(self, game, role: str, turn_duration: int) -> int:
        return 1

    def reset(self) -> None:
        pass


def test_game_seed_for_index_matches_tournament_sequence():
    seed = 17
    rng = random.Random(seed)
    expected = [rng.randrange(1 << 31) for _ in range(4)]

    assert [trace_module._game_seed_for_index(seed, idx) for idx in range(4)] == expected


def test_make_ladder_opponent_uses_ladder_opponent_seed(monkeypatch):
    calls = []

    def fake_create_scripted_opponent(name, seed=None):
        calls.append((name, seed))
        return _FakeOpponent()

    monkeypatch.setattr(
        trace_module,
        "create_scripted_opponent",
        fake_create_scripted_opponent,
    )

    opponent = trace_module._make_ladder_opponent("random", 7)

    assert isinstance(opponent, _FakeOpponent)
    assert calls == [("random", _opponent_seed(7, "random"))]


def test_play_games_reuses_one_ladder_opponent_per_seed(monkeypatch):
    args = SimpleNamespace()
    agent = object()
    opponent = _FakeOpponent()
    made_opponents = []
    played_opponents = []

    monkeypatch.setattr(trace_module, "_make_agent", lambda args, checkpoint, seed: agent)

    def fake_make_ladder_opponent(name, seed):
        made_opponents.append((name, seed))
        return opponent

    def fake_play_one(args, *, agent, baku, opponent, seed, game_index):
        played_opponents.append(baku)
        return {"game_index": game_index}

    monkeypatch.setattr(trace_module, "_make_ladder_opponent", fake_make_ladder_opponent)
    monkeypatch.setattr(trace_module, "_play_one", fake_play_one)

    rows = trace_module._play_games(
        args,
        checkpoint="checkpoint.pt",
        opponent="random",
        seed=11,
        games=3,
        start_game_index=4,
    )

    assert made_opponents == [("random", 11)]
    assert played_opponents == [opponent] * 7
    assert rows == [
        {"game_index": 4, "checkpoint": "checkpoint.pt"},
        {"game_index": 5, "checkpoint": "checkpoint.pt"},
        {"game_index": 6, "checkpoint": "checkpoint.pt"},
    ]

"""Read the dbt marts from the DuckDB warehouse into per-player histograms.

The readers find each table by name in any schema, so a dbt schema setting moves nothing here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np

from hal_loop.prior import ACTIONS, ROLES, PlayerCounts

HISTOGRAM_COLUMNS = 'player_id, human_role, split, second, "count"'


def _qualified(connection: duckdb.DuckDBPyConnection, table: str) -> str:
    found = connection.execute(
        "SELECT table_schema FROM information_schema.tables WHERE table_name = ? ORDER BY table_schema",
        [table],
    ).fetchall()
    if not found:
        raise LookupError(f"the warehouse has no table named {table}; run the dbt models first")
    return f'"{found[0][0]}"."{table}"'


def query(warehouse: Path, table: str, columns: str, where: str = "TRUE") -> list[tuple]:
    """Return rows of one mart from a read-only connection."""
    with duckdb.connect(str(warehouse), read_only=True) as connection:
        name = _qualified(connection, table)
        return connection.execute(f"SELECT {columns} FROM {name} WHERE {where}").fetchall()


@dataclass(frozen=True)
class Histograms:
    """Per-player move counts keyed by (player_id, role, split)."""

    counts: dict[tuple[str, str, str], np.ndarray]

    def players(self, split: str) -> list[str]:
        return sorted({player for player, _, s in self.counts if s == split})

    def get(self, player: str, role: str, split: str) -> np.ndarray:
        return self.counts.get((player, role, split), np.zeros(ACTIONS))

    def population(self, split: str) -> dict[str, np.ndarray]:
        """Return each role's summed counts over every player in one split."""
        totals = {role: np.zeros(ACTIONS) for role in ROLES}
        for (_, role, s), counts in self.counts.items():
            if s == split:
                totals[role] = totals[role] + counts
        return totals

    def player_counts(self, role: str, train: str = "train", held_out: str = "validation") -> list[tuple[str, PlayerCounts]]:
        """Return (player_id, PlayerCounts) for each player with held-out moves in the role."""
        players = sorted({player for player, r, s in self.counts if r == role and s == held_out})
        return [
            (player, PlayerCounts(self.get(player, role, train), self.get(player, role, held_out)))
            for player in players
        ]


def load_histograms(warehouse: Path) -> Histograms:
    """Read role_histograms into 60-second count vectors."""
    counts: dict[tuple[str, str, str], np.ndarray] = defaultdict(lambda: np.zeros(ACTIONS))
    for player, role, split, second, count in query(warehouse, "role_histograms", HISTOGRAM_COLUMNS):
        counts[(player, role, split)][int(second) - 1] += float(count)
    return Histograms(dict(counts))


def histogram_rows(warehouse: Path, splits: tuple[str, ...]) -> list[tuple]:
    """Return the sorted (player_id, human_role, split, second, count) rows of role_histograms in these splits."""
    names = ", ".join(f"'{split}'" for split in splits)
    rows = query(warehouse, "role_histograms", HISTOGRAM_COLUMNS, f"split IN ({names})")
    return sorted((str(p), str(r), str(s), int(second), int(count)) for p, r, s, second, count in rows)


@dataclass(frozen=True)
class TrainingGames:
    """The training split as the fit sees it.

    `game_keys` holds every finished, stopped, or abandoned game with a training move,
    because the fit uses the moves of unfinished games. `finished` counts the finished games alone.
    """

    game_keys: list[str]
    finished: int
    moves: int
    players: int


def training_games(warehouse: Path) -> TrainingGames:
    """Read the training split's game keys, finished-game count, move count, and player count."""
    rows = query(warehouse, "move_context", "game_key, player_id", "split = 'train'")
    finished = {key for (key,) in query(warehouse, "stg_games", "game_key", "status = 'finished'")}
    keys = sorted({row[0] for row in rows})
    return TrainingGames(keys, sum(key in finished for key in keys), len(rows), len({row[1] for row in rows}))

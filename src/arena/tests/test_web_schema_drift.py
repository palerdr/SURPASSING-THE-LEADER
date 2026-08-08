"""Keep the hand-written TypeScript types honest.

``web/src/types.ts`` mirrors :mod:`arena.web.schema` by hand. A hand-kept mirror
is the right trade for a surface this small, but only if drift fails the suite
rather than surfacing as ``undefined`` in a browser.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from arena.web.schema import OutcomeView, PlayerView, Snapshot

_TYPES = Path(__file__).resolve().parents[1] / "webclient" / "src" / "types.ts"


def _interface_fields(source: str, name: str) -> set[str]:
    match = re.search(rf"export interface {name} \{{(.*?)\n\}}", source, re.DOTALL)
    if match is None:
        raise AssertionError(f"web/src/types.ts has no interface {name}")
    return set(re.findall(r"^\s*(\w+)\??:", match.group(1), re.MULTILINE))


@pytest.mark.parametrize(
    ("model", "interface"),
    [(Snapshot, "Snapshot"), (PlayerView, "PlayerView"), (OutcomeView, "OutcomeView")],
)
def test_typescript_mirrors_the_python_schema(model, interface: str) -> None:
    source = _TYPES.read_text(encoding="utf-8")
    assert set(model.model_fields) == _interface_fields(source, interface)

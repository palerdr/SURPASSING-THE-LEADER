"""Keep the hand-written TypeScript types honest.

``webclient/src/types.ts`` mirrors :mod:`arena.web.schema` by hand. A hand-kept mirror
is the right trade for a surface this small, but only if drift fails the suite
rather than surfacing as ``undefined`` in a browser.
"""

from __future__ import annotations

import json
import re
import types
from enum import Enum
from pathlib import Path
from typing import Literal, Union, get_args, get_origin

import pytest
from pydantic import BaseModel

from arena.web.schema import OutcomeView, PlayerView, Snapshot
from arena.session import Phase
from stl.engine.game import HalfRoundResult

_TYPES = Path(__file__).resolve().parents[1] / "webclient" / "src" / "types.ts"


def _interface_fields(source: str, name: str) -> dict[str, tuple[str, bool]]:
    match = re.search(rf"export interface {name} \{{(.*?)\n\}}", source, re.DOTALL)
    if match is None:
        raise AssertionError(f"webclient/src/types.ts has no interface {name}")
    fields: dict[str, tuple[str, bool]] = {}
    for field, optional, annotation in re.findall(
        r"^\s*(\w+)(\?)?:\s*([^;]+);", match.group(1), re.MULTILINE
    ):
        fields[field] = (" ".join(annotation.split()), not bool(optional))
    return fields


def _typescript_type(annotation: object) -> str:
    origin = get_origin(annotation)
    if origin is list:
        return f"{_typescript_type(get_args(annotation)[0])}[]"
    if origin in (types.UnionType, Union):
        return " | ".join(_typescript_type(item) for item in get_args(annotation))
    if origin is Literal:
        return " | ".join(json.dumps(item) for item in get_args(annotation))
    if annotation is type(None):
        return "null"
    if annotation in (int, float):
        return "number"
    if annotation is str:
        return "string"
    if annotation is bool:
        return "boolean"
    if annotation is Phase:
        return "Phase"
    if annotation is HalfRoundResult:
        return "OutcomeResult"
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.__name__
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        raise AssertionError(f"TypeScript alias missing for enum {annotation.__name__}")
    raise AssertionError(f"unsupported Python wire annotation {annotation!r}")


def _model_fields(model: type[BaseModel]) -> dict[str, tuple[str, bool]]:
    return {
        name: (_typescript_type(field.annotation), field.is_required())
        for name, field in model.model_fields.items()
    }


def _string_union(source: str, name: str) -> set[str]:
    match = re.search(rf"export type {name}\s*=\s*(.*?);", source, re.DOTALL)
    if match is None:
        raise AssertionError(f"webclient/src/types.ts has no type alias {name}")
    return set(re.findall(r'"([^"]+)"', match.group(1)))


@pytest.mark.parametrize(
    ("model", "interface"),
    [(Snapshot, "Snapshot"), (PlayerView, "PlayerView"), (OutcomeView, "OutcomeView")],
)
def test_typescript_mirrors_the_python_schema(model, interface: str) -> None:
    source = _TYPES.read_text(encoding="utf-8")
    assert _model_fields(model) == _interface_fields(source, interface)


def test_typescript_enum_aliases_match_python_values() -> None:
    source = _TYPES.read_text(encoding="utf-8")
    assert _string_union(source, "Phase") == {item.value for item in Phase}
    assert _string_union(source, "OutcomeResult") == {
        item.value for item in HalfRoundResult
    }

"""Static import firewall for rigorous CFR modules."""

from __future__ import annotations

import ast
import pathlib


FORBIDDEN_IMPORTS = {
    "environment.awareness",
    "environment.observation",
    "environment.reward",
    "environment.route_math",
    "environment.route_stages",
}


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_environment_cfr_has_no_shaping_or_observation_imports():
    root = pathlib.Path(__file__).resolve().parents[1]
    cfr_files = sorted((root / "environment" / "cfr").glob("**/*.py"))
    assert cfr_files

    violations: list[tuple[str, str]] = []
    for path in cfr_files:
        for module in _imported_modules(path):
            for forbidden in FORBIDDEN_IMPORTS:
                if module == forbidden or module.startswith(f"{forbidden}."):
                    violations.append((str(path.relative_to(root)), module))

    assert violations == []


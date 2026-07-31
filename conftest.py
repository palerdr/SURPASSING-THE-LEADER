"""Repository-wide pytest options.

The suite carries a handful of tests that build real backup-tablebase
artifacts end to end. They are the ones that would catch a resume or a
backend-parity regression, so they must run in CI on every change -- but they
cost more than the other ~1,100 tests combined, which makes the default local
loop slow enough to discourage running it. They are marked ``slow`` and
deselected by default; ``--slow`` puts them back.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="also run tests marked slow (full artifact builds, backend parity)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="needs --slow (or CI, which always passes it)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)

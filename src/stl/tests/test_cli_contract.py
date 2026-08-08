"""Contract tests for the neutral Hydra experiment harness."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from stl.cli import _argv_from_command, dispatch


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_default_config_is_neutral() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config")

    assert cfg.command.name == "none"
    assert cfg.command.module is None


def test_command_values_translate_to_argparse_tokens() -> None:
    command = OmegaConf.create(
        {
            "name": "example",
            "module": "example.command",
            "count": 3,
            "enabled": True,
            "disabled": False,
            "items": ["a", "b"],
            "optional": None,
        }
    )

    assert _argv_from_command(command) == [
        "--count",
        "3",
        "--enabled",
        "--no-disabled",
        "--items",
        "a",
        "--items",
        "b",
    ]


def test_dispatch_imports_configured_module(monkeypatch) -> None:
    observed: list[list[str]] = []

    def command_main() -> None:
        observed.append(sys.argv[:])

    monkeypatch.setattr(
        "stl.cli.importlib.import_module",
        lambda name: SimpleNamespace(main=command_main),
    )
    cfg = OmegaConf.create(
        {"command": {"name": "example", "module": "example.command", "seed": 7}}
    )

    dispatch(cfg)

    assert observed == [["example.command", "--seed", "7"]]

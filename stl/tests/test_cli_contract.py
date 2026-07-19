"""Static contract checks for the Hydra-to-argparse command surface."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


ROOT = Path(__file__).resolve().parents[1]
COMMAND_DIR = ROOT / "config" / "command"


class _ParserCaptured(Exception):
    def __init__(self, parser: argparse.ArgumentParser) -> None:
        self.parser = parser


def _capture_parse_args(parser: argparse.ArgumentParser, *args, **kwargs):
    raise _ParserCaptured(parser)


def _parser_for(module) -> argparse.ArgumentParser:
    build_parser = getattr(module, "build_parser", None)
    if callable(build_parser):
        return build_parser()

    with patch.object(argparse.ArgumentParser, "parse_args", _capture_parse_args):
        with pytest.raises(_ParserCaptured) as captured:
            module.main()
    return captured.value.parser


def _command_config(path: Path) -> DictConfig:
    config = OmegaConf.load(path)
    assert isinstance(config, DictConfig)
    return config


@pytest.mark.parametrize("config_path", sorted(COMMAND_DIR.glob("*.yaml")), ids=lambda path: path.stem)
def test_hydra_command_matches_argparse_contract(config_path: Path) -> None:
    config = _command_config(config_path)
    module = importlib.import_module(str(config.module))
    assert callable(getattr(module, "main", None))

    parser = _parser_for(module)
    parser_destinations = {
        action.dest
        for action in parser._actions
        if action.option_strings and action.dest != "help"
    }
    config_destinations = set(config.keys()) - {"name", "module", "args", "description"}
    assert config_destinations <= parser_destinations

    if config.get("args") is None:
        required_destinations = {
            action.dest
            for action in parser._actions
            if action.option_strings and action.required
        }
        assert required_destinations <= config_destinations


def test_tier_a_stage1_experiment_overrides_the_selected_command() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "config")):
        config = compose(
            config_name="config",
            overrides=["command=tier_a", "experiment=tier_a_stage1"],
        )

    assert config.command.module == "stl.commands.tier_a"
    assert config.command.stage == 1
    assert config.command.limit == 1
    assert config.command.workers == 1


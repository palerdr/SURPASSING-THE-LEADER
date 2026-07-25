"""Hydra command dispatcher for the compact STL package."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf


_CONTROL_KEYS = {"module", "name", "description"}


def _flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def _append_arg(argv: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        argv.append(_flag_name(key) if value else "--no-" + key.replace("_", "-"))
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _append_arg(argv, key, item)
        return
    argv.extend([_flag_name(key), str(value)])


def _argv_from_command(command_cfg: DictConfig) -> list[str]:
    data = OmegaConf.to_container(command_cfg, resolve=True)
    if not isinstance(data, dict):
        raise TypeError("command config must be a mapping")
    argv: list[str] = []
    explicit_args = data.get("args")
    if explicit_args is not None:
        if not isinstance(explicit_args, Iterable) or isinstance(explicit_args, (str, bytes)):
            raise TypeError("command.args must be a list of CLI tokens")
        return [str(item) for item in explicit_args]
    for key, value in data.items():
        if key in _CONTROL_KEYS or key == "args":
            continue
        _append_arg(argv, key, value)
    return argv


@contextmanager
def _patched_argv(module_name: str, args: list[str]):
    original = sys.argv[:]
    sys.argv = [module_name, *args]
    try:
        yield
    finally:
        sys.argv = original


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    if "command" not in cfg or "module" not in cfg.command:
        raise ValueError("Hydra config must select a command with a module")
    if "rl" not in cfg:
        raise ValueError("Hydra config must select a versioned rl contract")
    from stl.learning.contracts import validate_rl_config

    validate_rl_config(cfg.rl)
    module_name = str(cfg.command.module)
    args = _argv_from_command(cfg.command)
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise AttributeError(f"{module_name} has no main()")
    with _patched_argv(module_name, args):
        result = module.main()
    if isinstance(result, int) and result:
        raise SystemExit(result)


if __name__ == "__main__":
    main()

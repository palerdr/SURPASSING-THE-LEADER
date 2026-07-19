"""Thin command router for DTH's independently configured workflows."""

from __future__ import annotations

import importlib
import sys


COMMANDS = {
    "dataset": "dth.generate_dataset",
    "train": "dth.train",
    "self-play": "dth.self_play",
    "mcts-audit": "dth.mcts",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        names = ", ".join(COMMANDS)
        print(f"usage: python -m dth <command> [hydra overrides]\ncommands: {names}")
        return
    command = sys.argv.pop(1)
    try:
        module_name = COMMANDS[command]
    except KeyError as exc:
        raise SystemExit(f"unknown DTH command {command!r}") from exc
    module = importlib.import_module(module_name)
    sys.argv[0] = module_name
    module.main()


if __name__ == "__main__":
    main()


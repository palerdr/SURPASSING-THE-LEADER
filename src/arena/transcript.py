"""The public play transcript file that the terminal and the browser write."""

from __future__ import annotations

import json
from pathlib import Path


def write_play_transcript(destination: str | Path, transcript: dict[str, object]) -> Path:
    """Atomically write a public transcript as sorted, indented JSON."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(transcript, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path

"""Learning kernels plus core support for exact-to-self-play training."""

from __future__ import annotations

from pathlib import Path


_SUPPORT_PATH = Path(__file__).resolve().parent / "support"
if _SUPPORT_PATH.exists():
    __path__.append(str(_SUPPORT_PATH))

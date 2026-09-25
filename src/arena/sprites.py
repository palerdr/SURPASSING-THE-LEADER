"""Compatibility alias for :mod:`arena.presentation.sprites`."""

import sys

from arena.presentation import sprites as _sprites

sys.modules[__name__] = _sprites

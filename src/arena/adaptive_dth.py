"""Compatibility alias for :mod:`arena.policies.adaptive`."""

import sys

from arena.policies import adaptive as _adaptive

sys.modules[__name__] = _adaptive

"""Checkpoint private opponent evidence beside the hosted command log."""
from __future__ import annotations

import base64
from dataclasses import asdict
import json
import zlib

import numpy as np

from arena.policies.translated_hal import TranslatedHalOpponentModel

SCHEMA = "arena-translated-hal-memory-v1"
MAX_BYTES = 4_000_000


class OpponentMemory:
    """Use bounded JSON checkpoints without executable serialization."""

    def dump(self, app):
        provider = app.state.hal_provider
        if provider.inner._pending is not None or provider._leap_pending is not None:
            raise RuntimeError("cannot checkpoint an unrevealed action")
        model = provider.opponent_model
        roles = {}
        for role, state in model._roles.items():
            fields = asdict(state)
            for key, value in fields.items():
                if isinstance(value, np.ndarray):
                    fields[key] = value.tolist()
            fields["recency_counts"] = [a.tolist() for a in state.recency_counts]
            for key in ("state_counts", "phase_counts"):
                fields[key] = [[list(k), v.tolist()] for k, v in getattr(state, key).items()]
            roles[role] = fields
        data = {"schema": SCHEMA, "config": asdict(model.config), "roles": roles,
            "offsets": {r: a.tolist() for r, a in model._offsets.items()},
            "last_self": model._last_self}
        raw = json.dumps(data, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > MAX_BYTES:
            raise ValueError("opponent checkpoint exceeds size limit")
        return base64.b64encode(zlib.compress(raw)).decode("ascii")

    def restore(self, app, memory):
        if not isinstance(memory, str) or len(memory) > MAX_BYTES:
            raise ValueError("invalid opponent checkpoint")
        try:
            packed = base64.b64decode(memory, validate=True)
            inflater = zlib.decompressobj()
            raw = inflater.decompress(packed, MAX_BYTES + 1)
            if len(raw) > MAX_BYTES or not inflater.eof or inflater.unused_data:
                raise ValueError("invalid compressed opponent checkpoint")
            data = json.loads(raw)
            provider = app.state.hal_provider
            model = TranslatedHalOpponentModel(provider.config)
            expected_config = json.loads(json.dumps(asdict(model.config)))
            if data["schema"] != SCHEMA or data["config"] != expected_config:
                raise ValueError("opponent checkpoint version mismatch")

            def array(value, shape, *, negative=False):
                result = np.asarray(value, dtype=np.float64)
                if result.shape != shape or not np.all(np.isfinite(result)) or (not negative and np.any(result < 0)):
                    raise ValueError("invalid opponent evidence array")
                return result.copy()

            def action(value):
                if value is not None and (type(value) is not int or not 1 <= value <= 60):
                    raise ValueError("invalid opponent action reference")
                return value

            for role, state in model._roles.items():
                fields = data["roles"][role]
                count = fields["observations"]
                if type(count) is not int or not 0 <= count <= 10**12:
                    raise ValueError("invalid opponent observation count")
                state.observations = count
                for key in ("global_counts", "transition_counts", "response_counts", "delta_counts", "log_weights"):
                    setattr(state, key, array(fields[key], getattr(state, key).shape, negative=key == "log_weights"))
                recency = fields["recency_counts"]
                if len(recency) != 3:
                    raise ValueError("invalid recency evidence")
                state.recency_counts = tuple(array(a, (60,)) for a in recency)
                for key, width in (("state_counts", 4), ("phase_counts", 2)):
                    counts = {}
                    for k, v in fields[key]:
                        if len(k) != width or any(type(i) is not int or i < 0 for i in k):
                            raise ValueError("invalid opponent context")
                        if key == "state_counts" and any(i > 5 for i in k):
                            raise ValueError("invalid state band")
                        if key == "phase_counts" and (k[0] not in model.config.periodicities or k[1] >= k[0]):
                            raise ValueError("invalid periodic phase")
                        if tuple(k) in counts:
                            raise ValueError("duplicate opponent context")
                        counts[tuple(k)] = array(v, (60,))
                    setattr(state, key, counts)
                for key in ("previous_self_action", "previous_opponent_action"):
                    setattr(state, key, action(fields[key]))
                model._offsets[role] = array(data["offsets"][role], (2, 2, 2, 119))
            model._last_self = action(data["last_self"])
        except (KeyError, TypeError, zlib.error, UnicodeError) as error:
            raise ValueError("invalid opponent checkpoint") from error
        provider.inner.opponent_model = model

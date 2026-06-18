# `environment/opponents/`

Scripted opponents and opponent factories.

Files:

- `base.py` - common opponent interface.
- `factory.py` - names and constructors used by scripts/tests.
- `random_bot.py`, `safe_bot.py`, `pattern_reader.py` - simple baselines.
- `baku_teachers.py`, `hal_teachers.py` - route-aware scripted teachers.
- `*_policy_helpers.py`, `teacher_helpers.py` - shared decision helpers.
- `model_opponent.py` - wrapper for trained model opponents.
- `league.py` - opponent league composition.

These policies are allowed to be heuristic and characterful. They should still
return legal seconds through `environment.legal_actions`.

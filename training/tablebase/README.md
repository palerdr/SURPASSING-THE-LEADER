# `training/tablebase/`

Certified interval tablebase construction.

This is the active dependency of `scripts/run_tier_a.py`.

Files:

- `epoch_sweep.py` - backward induction for one `(ttd_hal, ttd_baku, cprs)`
  epoch. It produces lower/upper value tables and asks the engine referee for
  survival probabilities.
- `__init__.py` - public exports: `EpochSpec`, `solve_epoch`,
  `survival_table`.

Tier A solves all one-death epochs for total time dead 60..299, then solves the
zero-death opening epoch against those completed child epochs.

# Engine Package

`src/` contains the deterministic rules engine: players, referee, constants, and
the half-round/game mechanics. Solver code may call this package, but engine code
should not depend on training, CFR, or learned models.

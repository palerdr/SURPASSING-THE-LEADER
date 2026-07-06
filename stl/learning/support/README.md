# `stl.learning.support/`

Core support for the learning and self-play bridge:

- route math, stage flags, observation/features, and reward helpers
- corpus diagnostics, calibration, audit packs, bootstrap/reanalysis loops
- self-play, tournaments, teacher demos, and strength-gate internals

These modules are part of the active core. `stl.learning.__init__` adds this
directory to the package search path so existing imports such as
`stl.learning.route_math` and `stl.learning.strength` keep working while the
top-level learning directory stays focused on `targets`, `model`, `train`, and
`gates`.

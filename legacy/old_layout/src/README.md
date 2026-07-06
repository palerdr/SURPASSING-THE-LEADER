# `src/`

Core engine truth lives here. Solver and learning code should call into this
package rather than reimplementing rules.

Files:

- `Constants.py` - shared rule constants: cylinder max, timing windows,
  physicality values, and clock anchors.
- `Player.py` - player state such as cylinder, deaths, total time dead, and
  physicality.
- `Referee.py` - survival probability and CPR/fatigue behavior.
- `Game.py` - half-round sequencing, clock advancement, leap-second detection,
  check/drop resolution, death/revival, and history records.

Keep this package free of AI policy, reward shaping, and training concerns.
Engine code should not depend on `environment/`, `training/`, `hal/`, CFR, or
learned models.

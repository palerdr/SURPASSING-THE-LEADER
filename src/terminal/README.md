# Terminal App Instructions

`src/terminal/` is the terminal STL game. A human plays canonical STL, or pure
DTH with `--pure-dth`, against one Hal policy provider:

```powershell
uv run python -m terminal play
uv run python -m terminal play --hal-agent abstract
uv run python -m terminal play --hal-agent abstract --buckets 5
uv run python -m terminal play --tui
```

`cli.py` holds the play command and its public transcript writer. `tui.py`
draws the full-screen ANSI scene that `--tui` selects.

## Boundaries

- [`docs/PROJECTS.toml`](../../docs/PROJECTS.toml) lets terminal import `arena`
  and `stl.engine`. `tests/meta/test_layer_boundaries.py` enforces that list
  and fails any import of `browser` or `hal_lab`. No project imports terminal.
- Terminal imports no `torch`, `gymnasium`, `stable_baselines3`, or
  `sb3_contrib`.
- `arena.policies.registry` builds Hal. It holds the play choices, the agent
  flags, the provider factories, and the pure-DTH gate. Add a provider there,
  and never in this app. `cli.py` keeps private aliases of a few registry
  functions for its tests. `command_play` looks up `_make_hal` in `cli.py` at
  call time, so a monkeypatch of `terminal.cli` intercepts it.
- The STL engine is the only referee. `arena.session.PlaySession` sequences the
  referee calls, and Hal chooses inside `PlaySession.submit`, after the human's
  second is accepted. The functions in `tui.py` take engine objects and return
  text, and they change no game state.
- `tui.py` reads the sprite pipeline, the scene art, and the rules text from
  `arena.presentation`, which the browser shares. The browser never imports
  this app.

## Play invariants

- `python -m terminal play --games N` is the repeated-opponent surface: one Hal
  provider is retained across games while each game receives a fresh canonical
  referee and seed. `--start-clock-sequence` can counterbalance canonical and
  leap-focused starts inside that same posterior. `--transcript` records only
  public pre-decision states, revealed actions, and outcomes for generated
  experiment data.
- `--public-hal-label` permits blinded policy comparisons without changing the
  provider recorded in the transcript. `--conceal-hal-details` keeps provider
  summaries and diagnostics in the transcript without printing them to the
  player.
- `python -m terminal play` opens with the ordinary-turn rules and waits for
  Enter on an interactive terminal before the first action. The full-screen and
  plain-text modes share the same rules text. Piped sessions do not consume an
  action as acknowledgement; automation may suppress the screen with
  `--skip-rules`. The opening screen intentionally does not disclose the
  leap-window advantage.
- The terminal offers `abstract`, and the browser refuses it. The `abstract`
  provider builds a missing bucket tablebase before the first game.

## Working in this subtree

Run `uv run python -m pytest src/terminal/tests -q`.

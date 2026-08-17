# Browser Client

`src/arena/webclient/` is the TypeScript client for canonical STL play, and one
of arena's play surfaces alongside `cli.py` and `tui.py`. It is a rendering and
input surface only. Its server is `src/arena/web/`, which holds every piece of
game state, and the STL engine remains the only referee.

This file is documentation nested inside the arena subtree, not an instruction
file; arena's binding guidance is `src/arena/README.md`.

## Working in this subtree

- **Never derive game rules here.** Legal seconds arrive in the snapshot as
  `legal_seconds`. Only the engine knows that Baku as Dropper may play 61 inside
  the leap window and that the Checker is always capped at 60. A client that
  computes legality will drift from the referee.
- **Never reconstruct hidden information.** The Dropper's and Checker's seconds
  reach the client only in `last_outcome`, after the half-round has resolved.
  If a screen needs a value that is not in the snapshot, the answer is that it
  is not knowable yet, not that the schema needs widening.
- **Sprites are prepared server-side.** `/art/{character}/{pose}/{index}.png`
  returns frames that are already keyed, mirrored, and split. Do not re-implement
  `Sprite.keyed` or the sheet splitter in TypeScript; both are subtle
  (`src/arena/sprites.py`) and already validated by the terminal front end.
- `src/types.ts` mirrors `src/arena/web/schema.py` by hand. A Python test
  asserts field names, types, nullability, and requiredness, so drift fails the
  suite. Update both.
- Text received from the API must be assigned with `textContent` or passed
  through `src/render/escape.ts` before entering an HTML template.
- Display labels are presentation data only. Canonical Hal/Baku identity stays
  on the server and is never inferred from a display label. Each player carries
  an authoritative `character` and `role`, and `winner_is_human` decides the
  victory treatment.
- `last_outcome.game_over` mirrors the canonical referee. The separate
  `session_ending` flag also covers a configured half-round cap and controls the
  acknowledgement button.
- Keep dependencies minimal. The repository hand-rolls a PNG codec rather than
  take Pillow; a framework or game engine here would be out of keeping.

## Layout

- `src/api.ts` — every server call. Mutating calls carry the `sequence` of the
  snapshot they were decided from, so a stale tab gets a 409 instead of a
  replayed move.
- `src/render/` — `sprites.ts` loads frames, `scene.ts` stages the three
  figures, `hud.ts` writes the header and stat columns.
- `src/screens/` — one module per phase: rules, live, outcome, victory.
- `src/main.ts` — holds the latest snapshot and re-renders on change.

The scene is staged after `art/panels/stl1.jpg` and uses the same constants as
`src/arena/tui.py`, so both front ends frame it identically.

## Running

```bash
uv run python -m arena.web                        # server on 127.0.0.1:8000
npm --prefix src/arena/webclient run dev          # client on 127.0.0.1:5173
```

Validate with:

```bash
npm --prefix src/arena/webclient test
npm --prefix src/arena/webclient run typecheck
npm --prefix src/arena/webclient run build
```

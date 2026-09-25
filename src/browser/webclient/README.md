# Browser Client

`src/browser/webclient/` is the TypeScript client for canonical STL play. It
is the browser app's play surface, beside the terminal app in `src/terminal/`.
It is a rendering and input surface only. Its server is the Python package in
`src/browser/`, which holds every piece of game state, and the STL engine
remains the only referee.

This file is documentation nested inside the browser subtree, not an
instruction file; the browser's binding guidance is `src/browser/README.md`.

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
  (`src/arena/presentation/`) and already validated by the terminal front end.
  Frames are reduced once to a coarse pixel grid by a smooth downsample and
  then blown up with hard edges (`pixelated` in `scene.ts`). Sampling the
  full-size frame with nearest neighbour instead picked different source
  pixels on each idle frame, which shimmered white round Yakou's edge.
- `src/types.ts` mirrors `src/browser/schema.py` by hand. A Python test
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
- The page is set in Benne, with lining, tabular Times numerals for the
  count. We self-host Benne from `public/fonts/` under the SIL
  Open Font License; Vite copies `public/` into `dist/` unchanged.
- Keep dependencies minimal. The repository hand-rolls a PNG codec rather than
  take Pillow; a framework or game engine here would be out of keeping.

## What the player gets

[`GAMEPLAY_FLOW.md`](GAMEPLAY_FLOW.md) is the design brief for the stage, in
the author's terms; this section is the implemented summary.

The browser surface covers the whole of `python -m terminal play`, staged as
one full-screen stage rather than a scrolling page:

- The title page shows DROP THE HANDKERCHIEF and Start. Start opens the
  chapter's rules PNG with Begin and one footer about same-second checks and
  modeled revival odds. The opening screens omit the HUD and name form.
  The server fixes the start clock and player identity. Later games skip
  these screens and start play.
- Each decision is two cuts. The scene — the seated Checker, Yakou, the
  Dropper — waits under a caption naming who drops and who checks. A click
  or Enter opens the action screen: one commit and the dial plate ported
  from palerdr.github.io (`src/render/dialplate.ts`), whose hour and minute
  hands read the game clock and whose red pointer steps one mark per second
  heard from the moment the cut lands. The count beneath the plate names the
  second now passing (`src/second.ts`): it reads 1 at the cut and rises on
  each beat, held at the last legal second for the final beat, with a gong in
  place of the final tick (`src/audio/tick.ts`; sample sources and rights in
  `public/audio/ATTRIBUTION.md`). There is no field: Commit or Enter plays
  the second the count names at that gesture. The plate is scenery, since
  the referee has no time limit, but at the gong the client commits the last
  legal second for you, as the chapter's one-minute rule requires. The scene
  has no timer, and other keys do not advance it. The clock starts with a
  full turn after Continue.
- After your commit, we hold the stage black for 1.5 seconds, then fade the
  white result and HUD in over 3.6 seconds. Enter can advance as soon as the
  server supplies the result, including during the hold or fade.
  We start this hold at the commit, without waiting for the transcript.
- The opening HTML displays the title while the session loads. Start opens
  the rules without waiting for the server. Enter advances either screen;
  the client resets and begins the server session while you read. Begin
  reveals the prepared scene without a request. If startup has not finished,
  Begin waits for it. Hal chooses no second during this preparation.
- Opening or reloading the page returns to the title and resets the series,
  with the opening clock and empty score and history. The restart carries the
  current sequence so stale actions cannot enter the fresh game.
- During the opening screens, character scene, clock, and result screens, the client requests the public rules at 15-second intervals
  to keep the policy server warm. These reads carry no session cookie and
  submit no action. They stop on commit or page exit, and before the gong.
  They resume after the result arrives and skip reads in hidden tabs.
  A server restart or network delay can still extend the black hold.
- The decision screen follows the terminal's `render_outcome`, with commas
  for its dashes and numbers in place of arithmetic: the result, both
  seconds, the squandered time, and the injected dose. It omits the revival
  chance and the round, half, and clock, which the corner already carries.
  Enter continues, and the next half-round opens on the scene again. The
  scene is drawn ahead of the acknowledgement; the clock opens only once
  the server's snapshot arrives, and an early gesture is honoured then.
- Four bars in the top-left corner carry both players' ST and TTD throughout;
  the top-right carries the clock, round, half, and the series tally once a
  series exists.
- The end-of-game screen draws the winner's still, the series tally, and
  Hal's match summary unless the server conceals it. "Next game" keeps Hal
  and its opponent model, as `--games N` does.
- A stale tab or a double click gets a 409, and the board reloads from the
  server rather than replaying a move.

## Layout

- `src/api.ts` — every server call. Mutating calls carry the `sequence` of the
  snapshot they were decided from, so a stale tab gets a 409 instead of a
  replayed move.
- `src/render/` — `sprites.ts` loads frames, `scene.ts` stages the three
  figures and the victory still, `dialplate.ts` builds and drives the clock,
  `hud.ts` writes the corner bars, `text.ts` holds the terminal's
  player-facing wording as pure functions.
- `src/audio/tick.ts` schedules ticks and the closing gong on the audio
  timeline. We read the output timestamp for the count and pointer. Late
  loading and audio resume retain the elapsed turn. The server supplies
  the duration: 61 seconds places the gong one second after a normal turn.
- `tools/prepare_clock_audio.py` prepares the samples from the source WAV;
  `public/audio/ATTRIBUTION.md` records the cuts and separation limits.
- `src/second.ts` — the one pure mapping from beats heard and the server's
  legal seconds to the second the count names and Commit plays.
- `src/screens/` — one module per cut: rules, beat, live, outcome, victory.
- `src/main.ts` — holds the latest snapshot and transcript and re-renders on
  change.

The scene is staged after `art/panels/stl1.jpg` and uses the same constants as
`src/terminal/tui.py`, so both front ends frame it identically.
One difference is deliberate: the terminal cycles Yakou's four-frame idle
sheet, while the browser holds his first frame. The sheet's frames are separate
drawings, each re-centred on a shared canvas, and on a screen the cycle read
as the referee twitching.

## Running

One process, once the client is built:

```bash
npm --prefix src/browser/webclient run build       # writes dist/, gitignored
uv run python -m browser                          # game on 127.0.0.1:8000
```

`python -m browser` takes the same agent, seed, start-clock, label,
transcript, and `--pure-dth` options as `python -m terminal play`. For client
development run the Vite server instead, which proxies `/api` and `/art` to the
Python server:

```bash
uv run python -m browser                          # server on 127.0.0.1:8000
npm --prefix src/browser/webclient run dev         # client on 127.0.0.1:5173
```

Validate with:

```bash
npm --prefix src/browser/webclient test
npm --prefix src/browser/webclient run typecheck
npm --prefix src/browser/webclient run build
```

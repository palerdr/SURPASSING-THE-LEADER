# Browser gameplay flow

This is the design brief for the browser surface, in the author's terms. The
server (`src/arena/web/`) and the STL engine own every rule; this document
describes only what the player sees, hears, and does, and in what order. The
second half records every preference the author has stated, so a later
change can be checked against them.

## One stage

The game is one full-screen stage, never a scrolling page. Nothing sits
beside or below it: no half-round log, no rules fold, no settings. Every
moment of play is a full-screen cut on that stage.

The status of the two players is four bars in the top-left corner: each
player's ST (the vial) and TTD (total time dead), with the player's name,
current role, and death count above them. On the result screen, the part of
a bar that the half-round just added is drawn in red; it returns to white
when you continue. The top-right corner carries the
game clock, the round and half, the game number, and the series tally, set
large so the state of the game reads at a glance.

## The cuts, in order

1. **Title, then rules.** The first screen shows DROP THE HANDKERCHIEF
   and one Start button on black. Start or Enter opens the chapter's rules
   PNG (`art/panels/stl_rules.png`) without a server wait. Beneath the panel,
   Begin starts play. One footer explains same-second checks and the modeled
   revival odds. These screens contain no name form or HUD.
   Opening or reloading returns to the title and resets the series, with the
   opening clock and empty score and history. Begin or Enter on the rules
   during startup waits for preparation. The client resets and begins the
   server session while you read, but keeps the scene hidden until Begin.
   Hal chooses no second during preparation. A prepared Begin needs no request.

2. **The scene.** The three figures stand in miniature on the black field:
   the Checker seated with his back to the drop, Yakou standing at the
   centre with his watch raised, the Dropper on the right holding the
   handkerchief. A caption names who drops and who checks this half. The
   figures hold still; nobody twitches. The scene stays until you click or press Enter. Continue opens the
   clock; other keys do not advance it. The clock and sound start at that
   gesture, with the full turn remaining. One gesture advances one screen.

3. **Cut to the action window.** The scene cuts to a full-screen prompt:
   "You are the Checker" or "You are the Dropper", one Commit button, and to
   the right, filling the black space, the dial plate from
   palerdr.github.io. Its hour and minute hands read the game clock; its red
   pointer steps one mark per second of the turn. Beneath the plate a large
   count names the second now passing: it reads 1 when the clock opens and
   rises on the same instant each beat sounds, so after beat k it reads
   k + 1. There is no field. Commit or Enter plays the second the count
   names at that gesture. The instruction states the server's legal range,
   “1–60” in a normal turn. Hal's second does not exist until the commit
   arrives at the server. The count holds at the last legal second for the
   final beat; when the gong rings the minute is up, as the chapter's rules
   say: the turn ends, and that last legal second (60, or 61 for Baku
   dropping in the leap window) is committed for you.

4. **Black.** After the commit the stage goes fully black, bars and all,
   and holds for 1.5 seconds from your commit. The result then fades in over
   3.6 seconds. You can press Enter to advance during the hold or fade once
   the server supplies the result. A slow server
   can extend the black hold until it supplies the result.

5. **Decision screen.** The result of the exchange: the result line, the
   drop and the check on two separate lines with their seconds, the
   squandered time as a plain number when the check succeeded,
   and the dose when it failed. No calculation is shown, only the number.
   The revival chance is not shown. Whether the checker was revived is in
   the result line already, so it is not repeated. The round, half, and
   clock are not repeated either; the top-right corner carries them. We show FAILED in red and the remaining text in white. The verdict uses large, centred type, with the exchange details below
   it. You can press Enter to continue without waiting for the fade.

6. **Repeat** from the scene for the next half-round, with the roles swapped
   when both survived.

7. **Game over.** The winner's still, centred, with the verdict and the
   series tally. "Next game" starts the next game of the series against the
   same Hal, which keeps its opponent model, straight from the scene; the
   rules slide is not shown again.

## Stated preferences

Each line is something the author asked for. Keep them all.

### Rules and settings

- Explain the game in plain terms, in one paragraph, with no solver jargon.
- The start clock is not malleable.
- The first screen uses the actual PNG of the rules, distilled into a
  succinct paragraph.
- No half-round log and no rules fold on the stage.

### Screens

- One screen: the scene, then a cut to full screen with the action window
  and the clock, then a decision screen for the result, then repeat.
- The two players' status is four bars in the top-left corner.
- No grid of sixty squares to pick a second from, and no field to type in:
  one Commit, played on the second the count names.
- The result screen is delayed behind a full black screen, for anticipation:
  1.5 seconds, then a 3.6-second fade in the style of the Dark Souls death screen.
- The result screen does not repeat "revived yes/no"; the result line says
  it.
- We show FAILED in red and the remaining result text in white.
- Result lines use a comma, not an em dash: "CHECK FAILED, died, revived".
- No calculation on the result screen, just the number.

### The scene and its figures

- The referee must not glitch: no white edge shimmer round Yakou.
- The figures must not twitch: Yakou holds one still frame.
- The seated Checker faces away from the Dropper, back to the drop.
- The scene is a miniature, then even smaller: a small group low in the
  field, about a ninth of the stage height.
- The figures are pixelated: reduced to a coarse pixel grid and blown back
  up with hard edges.

### The clock

- The clock started as a tiny minute clock in the corner counting down, then
  moved to the right of the commit field, then to the far right filling the
  black space, white-faced.
- It must look elegant, not blocky. The seconds beneath it must not touch
  the rim.
- Use the exact clock from the author's personal site,
  https://github.com/palerdr/palerdr.github.io: the dial plate with its
  rims, sixty ticks, the 61st-second leap tick, hour and minute hands, red
  pointer, hub, and core readout.
- Keep the number beneath the clock, and keep it in sync with the sound.
- The pointer's steps must be discrete, one mark per second. The count
  below rises 1 to 60 and names the second now passing, one ahead of the
  mark the pointer stands on. During a 61-second turn, the pointer holds at
  the red leap mark for the extra second and Baku's count reads 61.
- When the gong sounds the turn ends and the pick is forced to the last
  legal second, which is the number the count already shows.

### The sound

- A ticking sound, one per second, with the gravitas of a grandfather
  clock, not a bedroom clock: the game's clock hangs in a massive
  watchtower, so the tick must be deep.
- The tick must be distinctive, a real TICK, but never high-pitched.
- The references, in order: "Echoing Clock Tick | HQ Sound Effect"
  (https://www.youtube.com/watch?v=KwGNDJfSmFk); then the sound of
  https://www.tiktok.com/@usogoat/video/7334081665075006763; then, better,
  the tick of https://www.tiktok.com/@2kys3/video/7500673204722208007,
  which is the one in use.
- At the last second, the gong from that same TikTok rings in place of the
  final tick.
  A 61-second turn has a tick at 60 and its gong at 61. The Checker still
  commits the last second in the server-supplied legal set, at most 60.
- The target is the clock sound without the TikTok music. We suppress the
  music with a smooth spectral mask and short cuts. The mono source contains
  both sounds, so we cannot claim a clean isolated stem.
- The sound and number share the audio output timestamp. Before audio loads,
  we use the browser clock. Late audio joins the current turn and skips missed
  beats; it does not restart the minute. After suspension, we replace the
  queued sources at the current second.

### Type

- We use Benne, the serif from https://situational-awareness.ai/,
  self-hosted under the SIL Open Font License from `public/fonts/`.
- For the count, we use Times New Roman with lining, tabular numerals.
  Benne has old-style digits and no lining-number alternate. The count's
  digits need a common baseline and equal widths.

## What implements it

- We use one tick at its source pitch and one gong from the chosen TikTok
  (`src/audio/tick.ts`; source and processing in `public/audio/ATTRIBUTION.md`).
  We no longer alternate a pitch-shifted copy. Browsers require a gesture
  before sound can start. A failed load can retry on your next gesture.
- The dial plate is ported line for line from the site's `ClockFace.astro`
  (`src/render/dialplate.ts`).
- Wording on the decision screen follows the terminal interface, so a player
  moving between the two surfaces reads the same game. Each line opens with
  a capital letter, since every line stands alone.

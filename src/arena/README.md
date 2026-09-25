# Arena Project Instructions

`src/arena/` is the neutral executable surface for matches between peer projects.
It may import public interfaces from `stl`, `dth`, and `abstract`; peer projects
must not import one another in return.

- The STL engine remains the only canonical live-game referee.
- The completed DTH tablebase is the default Hal policy provider.
- Providers return policy distributions; `PolicyDrivenAgent` alone masks and
  samples literal legal seconds.
- `adaptive_dth.py` is the one-step exploit layer: callers supply population
  Dirichlet priors, revealed actions update separate role posteriors, and the
  complete DTH matrix constrains every selected policy under a per-game
  epsilon budget. It falls back to equilibrium when evidence is weak or an
  opponent action lies outside DTH's 1..60 contract.
- `--adaptive-prior-json` accepts a versioned pair of learned role means or a
  mixture of role archetypes and strengths; the CLI never fits that prior
  during live play. Session
  transcripts include bounded per-game exploit, epsilon, fallback, and
  saddle-gap diagnostics for offline validation.
- `arena play --games N` is the repeated-opponent surface: one Hal provider is
  retained across games while each game receives a fresh canonical referee and
  seed. `--start-clock-sequence` can counterbalance canonical and leap-focused
  starts inside that same posterior. `--transcript` records only public
  pre-decision states, revealed actions, and outcomes for generated experiment
  data.
- `--public-hal-label` permits blinded policy comparisons without changing the
  provider recorded in the transcript. `--conceal-hal-details` keeps provider
  summaries and diagnostics in the transcript without printing them to the
  player.
- `arena play` opens with the ordinary-turn rules and waits for Enter on an
  interactive terminal before the first action. The full-screen and plain-text
  modes share the same rules text. Piped sessions do not consume an action as
  acknowledgement; automation may suppress the screen with `--skip-rules`. The
  opening screen intentionally does not disclose the leap-window advantage.
- DTH projection is exact for the shared state and actions 1..60. The only
  prospective mismatch is Baku's legal Dropper action 61 in the public leap
  window; arena keeps that canonical action even though DTH has no 61 policy.
- Projection adapters may not alter canonical game state or transitions.
- Keep generated artifacts in the owning project, never under `src/arena/`.
- `policies/registry.py` is the one Hal provider registry. It holds the agent
  choices, the agent flags, the provider factories, and the pure-DTH gate.
  `cli.py` and `web/__main__.py` build Hal through it, and `cli.py` keeps its
  old private names as aliases of the registry functions.
- `policies/__init__.py` imports nothing. Import each provider from its own
  module, so the hosted runtime never loads torch or the stable-baselines3
  training stack. `tests/test_runtime_imports.py` checks this in a fresh
  interpreter.
- `presentation/` holds the sprite pipeline (`sprites.py`, `scene_art.py`) and
  the player-facing rules text (`rules_text.py`). The terminal and the browser
  both read it, and the browser never imports `tui.py` or `cli.py`. Art paths
  resolve from the package, so the art loads from any working directory.
- `variants.py` holds `PureDTHGame`. `contracts.py`, `agent.py`, `session.py`,
  `match.py`, and `variants.py` import no project except `stl.engine`; `dth`
  and `abstract` enter through the adapters and `policies/`.

## Play surfaces and the session

`session.py` owns the phase machine every interactive surface drives:
`RULES -> AWAITING_ACTION -> AWAITING_ACK -> GAME_OVER`. It performs no I/O and
no rendering; it only sequences the referee calls. `cli.py` and `web/app.py` are
both thin adapters over it, so a rules change lands in one place.

Hal's action is chosen inside `PlaySession.submit`, after the human's second has
been accepted and validated. That ordering is the hidden-information guarantee,
not a convenience: while a client is deciding, Hal's second does not exist in
the process, so no snapshot can leak it. Do not hoist that call earlier to
"prepare" a move.

`web/app.py` serves the TypeScript client in `webclient/`. It builds its
provider once at startup — provider construction memory-maps a
multi-gigabyte artifact and the `abstract` provider can build a tablebase
outright, so neither may happen on a request path; `python -m arena.web`
refuses `--hal-agent abstract` for that reason. `web/schema.py` holds the only
serializer that faces the browser, so the seat-scoping rule has exactly one
place to be enforced and one place to be tested.

The browser server is one repeated-opponent series, the same unit as
`arena play --games N`: one Hal is retained across games, game `N` is seeded
with the base seed plus `N`, and every finished game is appended to a public
transcript in the CLI's `arena-public-play-session-v1` shape. `GET
/api/transcript` serves that transcript plus the live game's resolved
half-rounds, and `--transcript PATH` rewrites the same JSON after every
finished game. `--public-hal-label`, `--conceal-hal-details`, `--pure-dth`,
and every agent option of `arena play` are accepted with the same meaning.
When `webclient/dist/` has been built, the Python server serves it at `/`, so
one process is the whole game.

Engine identities remain exactly `Hal` and `Baku`. `--human-name` and the web
session's `human_name` are presentation labels only; they never replace Baku's
rule-bearing identity. `Hal` is reserved as a display label. Browser session
replacement is a sequenced mutation, is allowed only before play or after a
terminal acknowledgement, and advances the sequence across the replacement.
The browser server's live provider set is `dth`, `adaptive-dth`,
`exploit-hal`, and, behind `--pure-dth`, `perfect-hal` and `pm-hal`; terminal
`perfect-hal --perfect-hal-model translated-v1` also supports canonical play
through its leap fallback. Terminal `arena play` offers `abstract`. The
retired `stl-mcts` surface is not advertised. Browser snapshots carry
server-owned character, role, and winner-seat fields, so the client never
infers identity from presentation labels.

The Vercel entrypoint uses `web/hosted.py` and `web/production.py` to isolate
players with secure cookies and Redis command logs. It replays accepted commands
through the same local HTTP adapter and commits each mutation before returning
a reveal. A process keeps the game it last served and rebuilds it from the
command log only when Redis shows that the game moved on elsewhere.
`web/ledger.py` then writes the game's public history to Supabase,
and `GET /api/leaderboard` ranks each player's best win by the winner's
seconds of life left. A restart and a next game each replace the record with fresh seeds, an empty
command list, and the next sequence number, so replay covers only the current
game. The hosted exact Hal keeps no memory. You can opt into translated Hal
with `STL_HAL_POLICY=translated-v1`. The hosted adapter then stores a private
opponent checkpoint with each game record and restores it before command
replay. Reload and next-game requests retain that evidence. Separate cookies
keep separate models; workers share the immutable tablebase. The local server
keeps one repeated-opponent series.
See [deployment instructions](web/DEPLOYMENT.md).

The browser requests a sequenced restart on page load. This abandons the active
game and clears the visible series before returning the title page. Ordinary
session reads still recover state for stale-request handling and worker replay.

The Hal research narrative, with its training commands and frozen results,
lives in [`src/hal_lab/docs/HAL_RESEARCH.md`](../hal_lab/docs/HAL_RESEARCH.md).

## Browser deployment

See [web/DEPLOYMENT.md](web/DEPLOYMENT.md) for the certified recurrence build,
local launch commands, and the per-player session changes needed for Vercel.
The complete DTH reader accepts its source-bound v3 artifact. The browser
keeps the existing commit-before-sampling and reveal contracts.

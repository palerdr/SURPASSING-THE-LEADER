# Arena Project Instructions

`src/arena/` is the game library that the two game apps and the Hal lab share.
It may import public interfaces from `stl`, `dth`, and `abstract`, and those
projects must not import it.

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
  mixture of role archetypes and strengths; a play surface never fits that
  prior during live play. Session
  transcripts include bounded per-game exploit, epsilon, fallback, and
  saddle-gap diagnostics for offline validation.
- [`src/terminal/`](../terminal/README.md) is the terminal game,
  `python -m terminal play`. Its README holds the invariants of the play flags
  and the rules screen.
- [`src/browser/`](../browser/README.md) is the browser game,
  `python -m browser`, with the hosted Vercel deployment. Its README holds the
  server, hosted-session, and deploy invariants.
- DTH projection is exact for the shared state and actions 1..60. The only
  prospective mismatch is Baku's legal Dropper action 61 in the public leap
  window; arena keeps that canonical action even though DTH has no 61 policy.
- Projection adapters may not alter canonical game state or transitions.
- Keep generated artifacts in the owning project, never under `src/arena/`.
  The ignored prepared-frame cache `art/.sprite-cache/` is the one exception,
  because it derives from the art beside it.
- `policies/` holds the runtime Hal providers: `adaptive.py`,
  `perfect_hal.py`, `bayesian_hal.py`, `ensemble_hal.py`, `translated_hal.py`,
  `exploit_hal.py`, `aggro_hal.py`, and `pm_hal.py`, with `registry.py` and
  `exploit_hal_config.py`. Hal training, evaluation, and study code lives in
  [`src/hal_lab/`](../hal_lab/README.md).
- `config/` holds the runtime configs that the providers read, the two
  Exploit v2 smoke configs that the training contract tests load beside
  `exploit_hal_v2.yaml`, and the six sealed Hal records that hal_lab owns.
  The hal_lab README lists them.
- `policies/registry.py` is the one Hal provider registry. It holds the play
  choices, the play flags, the provider factories, and the pure-DTH gate. The
  terminal app and the browser app build Hal through it.
  `python -m hal_lab match` builds every play agent there too. hal_lab owns
  the research-only Aggro Hal choice, its flags, and its factory; the
  registry's pure-DTH gate still lists `aggro-hal`.
- `policies/__init__.py` imports nothing. Import each provider from its own
  module, so the hosted runtime never loads torch or the stable-baselines3
  training stack. `src/browser/tests/test_runtime_imports.py` checks this in
  a fresh interpreter.
- `presentation/` holds the sprite pipeline (`sprites.py`, `scene_art.py`) and
  the player-facing rules text (`rules_text.py`). The terminal app and the
  browser both read it, and the browser never imports the terminal app. Art
  paths resolve from the package, so the art loads from any working directory.
- `art/` holds the runtime art that `presentation/scene_art.py` reads: the
  character sprites in `art/sprites/` and the rules spread
  `art/panels/stl_rules.png`, which opens the browser game. Git tracks both.
  `browser.deploy.prepare_vercel` pre-renders them into the Vercel bundle.
  The manga reference images live in
  [`docs/game-sources/reference-art/`](../../docs/game-sources/reference-art/).
- `variants.py` holds `PureDTHGame`. `contracts.py`, `agent.py`, `session.py`,
  `match.py`, and `variants.py` import no project except `stl.engine`; `dth`
  and `abstract` enter through the adapters and `policies/`.
- `match.py` plays one agent-versus-agent game. `python -m hal_lab match`
  runs the paired-seat series and its SPRT in
  `src/hal_lab/harness/series.py`.
- `transcript.py` writes the public play transcript file. The terminal's
  `--transcript` and the browser's `--transcript` both call it, so the two
  files keep one format. `session.py` stays free of I/O.
- `testing.py` holds the fakes that the tests of more than one project share,
  such as `StageAgent` and `make_session`. No runtime module imports it.
- No arena module imports `terminal`, `browser`, or `hal_lab`.
  `tests/meta/test_layer_boundaries.py` enforces this rule.

## Play surfaces and the session

`session.py` owns the phase machine every interactive surface drives:
`RULES -> AWAITING_ACTION -> AWAITING_ACK -> GAME_OVER`. It performs no I/O and
no rendering; it only sequences the referee calls. `src/terminal/cli.py` and
`src/browser/app.py` are both thin adapters over it, so a rules change lands in
one place.

Hal's action is chosen inside `PlaySession.submit`, after the human's second has
been accepted and validated. That ordering is the hidden-information guarantee,
not a convenience: while a client is deciding, Hal's second does not exist in
the process, so no snapshot can leak it. Do not hoist that call earlier to
"prepare" a move.

Engine identities remain exactly `Hal` and `Baku`. A display label never
replaces Baku's rule-bearing identity, and `Hal` is reserved as a display
label.

The Hal research narrative, with its training commands and frozen results,
lives in [`src/hal_lab/docs/HAL_RESEARCH.md`](../hal_lab/docs/HAL_RESEARCH.md).

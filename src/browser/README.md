# Browser App Instructions

`src/browser/` is the browser STL game. It holds the FastAPI game server, the
hosted wrapper that Vercel runs, the TypeScript client in
[`webclient/`](webclient/README.md), and the deploy pipeline in `deploy/`.
Build the client once, then run the local server:

```powershell
npm --prefix src/browser/webclient run build
uv run --project src/browser python -m browser
```

## Boundaries

- [`docs/PROJECTS.toml`](../../docs/PROJECTS.toml) lets browser import
  `arena`, `stl.engine`, and `dth.agent`. `tests/meta/test_layer_boundaries.py`
  enforces that list. No project imports browser, and browser imports neither
  `terminal` nor `hal_lab`.
- Browser imports no `torch`, `gymnasium`, `stable_baselines3`, or
  `sb3_contrib`. `tests/test_runtime_imports.py` and `tests/test_manifest.py`
  check the hosted runtime in a fresh interpreter.
- `arena.policies.registry` builds Hal for `python -m browser`, with the same
  provider construction and agent options as `python -m terminal play`. Add a
  provider there, and never in this app.
- The browser reads the rules text and the prepared art from
  `arena.presentation`, which the terminal shares. It never imports the
  terminal app.
- `tests/fakes.py` holds the fakes that more than one browser test shares.
  The root `tests/parity/test_terminal_browser.py` imports them too: it runs the
  terminal and the browser with the same options, and it lives outside both
  apps because neither app imports the other.

## Server invariants

`app.py` serves the TypeScript client in `webclient/`. It builds its
provider once at startup — provider construction memory-maps a
multi-gigabyte artifact and the `abstract` provider can build a tablebase
outright, so neither may happen on a request path; `python -m browser`
refuses `--hal-agent abstract` for that reason. `schema.py` holds the only
serializer that faces the browser, so the seat-scoping rule has exactly one
place to be enforced and one place to be tested. `arena.session.PlaySession`
samples Hal's action inside `submit`, after the human's second is accepted;
`src/arena/README.md` states that hidden-information guarantee.

The browser server is one repeated-opponent series, the same unit as
`python -m terminal play --games N`: one Hal is retained across games, game
`N` is seeded with the base seed plus `N`, and every finished game is appended
to a public transcript in the CLI's `arena-public-play-session-v1` shape. `GET
/api/transcript` serves that transcript plus the live game's resolved
half-rounds, and `--transcript PATH` rewrites the same JSON after every
finished game. `--public-hal-label`, `--conceal-hal-details`, `--pure-dth`,
and every agent option of `python -m terminal play` are accepted with the same
meaning.
When `webclient/dist/` has been built, the Python server serves it at `/`, so
one process is the whole game.

`--human-name` and the web session's `human_name` are presentation labels
only; they never replace Baku's rule-bearing identity. `Hal` is reserved as a
display label. Browser session
replacement is a sequenced mutation, is allowed only before play or after a
terminal acknowledgement, and advances the sequence across the replacement.
The browser server's live provider set is `dth`, `adaptive-dth`,
`exploit-hal`, and, behind `--pure-dth`, `perfect-hal` and `pm-hal`; terminal
`perfect-hal --perfect-hal-model translated-v1` also supports canonical play
through its leap fallback. `python -m terminal play` offers `abstract`. The
retired `stl-mcts` surface is not advertised. Browser snapshots carry
server-owned character, role, and winner-seat fields, so the client never
infers identity from presentation labels.

The browser requests a sequenced restart on page load. This abandons the active
game and clears the visible series before returning the title page. Ordinary
session reads still recover state for stale-request handling and worker replay.

## Hosted sessions

The Vercel entrypoint uses `hosted.py` and `deploy/production.py` to isolate
players with secure cookies and Redis command logs. `redis_store.py` holds the
Upstash store and its atomic compare-and-set. The entrypoint replays accepted commands
through the same local HTTP adapter and commits each mutation before returning
a reveal. A process keeps the game it last served and rebuilds it from the
command log only when Redis shows that the game moved on elsewhere.
`ledger.py` then writes the game's public history to Supabase,
and `GET /api/leaderboard` ranks each player's best win by the winner's
seconds of life left. A restart and a next game each replace the record with fresh seeds, an empty
command list, and the next sequence number, so replay covers only the current
game. The hosted exact Hal keeps no memory. You can opt into translated Hal
with `STL_HAL_POLICY=translated-v1`. The hosted adapter then stores a private
opponent checkpoint with each game record and restores it before command
replay. Reload and next-game requests retain that evidence. Separate cookies
keep separate models; workers share the immutable tablebase. The local server
keeps one repeated-opponent series.

## Deployment

[`deploy/DEPLOYMENT.md`](deploy/DEPLOYMENT.md) holds the certified recurrence
build, the local launch commands, and the Vercel pipeline. The complete DTH
reader accepts its source-bound v3 artifact. The browser keeps the existing
commit-before-sampling and reveal contracts.

- `deploy/manifest.py` is the one list of the hosted runtime's files.
  `deploy/prepare_vercel.py` copies each file to `runtime/<path>` in the
  bundle, and `deploy/production.py` hashes the files with an `in_version`
  flag into the hosted code version. `tests/test_manifest.py` fails when the
  production entry imports a first-party module that the list omits. Add a
  runtime file to the list in the change that adds its import.
- The bundle keeps the repository layout below `runtime/`, because the DTH
  `code_config_digest` labels its inputs `src/dth/...` and `uv.lock`.
  `dth.agent.runtime_source_files()` names the DTH files.
- A change to a file in the version ends every live hosted session at the
  next deploy. `names.py` stays out of the version, so a word-list change does
  not end games.
- `prepare_vercel` reads the pins of the bundle's packages from
  `src/browser/uv.lock`, writes the bundle to the ignored `build/vercel/`, and
  copies the ignored Vercel project link from `.vercel/project.json`. It still
  copies the root `uv.lock` to `runtime/uv.lock`, because the DTH digest
  labels that file.

## Environment

The browser app is its own uv project. `pyproject.toml` declares FastAPI,
uvicorn, and httpx, and it installs the root `stl-solver` package as an
editable path dependency, so the browser code still ships from `src/`. A
browser dependency change edits `src/browser/uv.lock` alone; the root lock,
the DTH artifact digest, and the leap builder hash stay unchanged. Run every
browser command with `uv run --project src/browser`. The environment lives in
the ignored `src/browser/.venv/`.

The hosted runtime never loads torch. Local play against `exploit-hal` or
`pm-hal` loads a torch provider, so add the `learned` group:

```powershell
uv run --project src/browser --group learned python -m browser --hal-agent exploit-hal
```

## Working in this subtree

Run `uv run --project src/browser python -m pytest src/browser/tests tests/parity -q`.
The root suite does not collect these tests, because the root environment
has no FastAPI. The client is not covered by
`pytest`; run `npm --prefix src/browser/webclient test`,
`npm --prefix src/browser/webclient run typecheck`, and
`npm --prefix src/browser/webclient run build` after a client change. The
typecheck needs `npm --prefix src/browser/webclient install` once.

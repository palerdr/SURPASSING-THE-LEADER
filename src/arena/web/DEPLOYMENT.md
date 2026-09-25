# Optimal play and deployment

You can run the browser against the complete policy provider:

```sh
npm --prefix src/arena/webclient run build
uv run python -m arena.web --port 8766
```

The default path is `src/dth/artifacts/complete_full_v1`. The local v3 artifact
at that path passes source checks and file hashes. We preserved the prior
artifact at `src/dth/artifacts/complete_full_v1_pre_recurrence`. Its August
build ceased to match the Rust source after a loop-syntax cleanup. We built
a fresh tablebase; we did not change or bypass its manifest checks.

For a new checkout, build the artifact before starting the server:

```sh
uv run python -m dth complete --config-name complete_fast_v1 output_dir=src/dth/artifacts/complete_full_v1
```

The build needs a C compiler. Requests use the Python policy reference,
with a certified LP fallback. They do not invoke the tablebase sweep.
See [the recurrence contract](../../dth/docs/FAST_TABLEBASE.md) for rebuild
and resume details. Generated artifacts remain outside Git.

## Scope of optimal play

The provider certifies pure DTH on actions 1..60. Use `--pure-dth` for that
game. Without the flag, the STL referee retains Baku's legal Dropper action
61 in the leap window. The DTH artifact gives no whole-game optimality
claim for that extra action.

We tested a browser submission with the real provider. The server returned
Hal's revealed action after the human committed, then continued the game.
The transcript recorded a `2.78e-17` saddle gap. The server constructs the
provider at startup and samples Hal's action inside `PlaySession.submit`.
A pre-commit snapshot contains no unrevealed action.

## Vercel deployment

You can play at [surpassing-the-leader.vercel.app](https://surpassing-the-leader.vercel.app).
We deployed the certified artifact on 2026-09-11. We completed a seven-move
preview game, checked player isolation and competing submissions, and submitted
a move through the public production API. We also checked the browser's clock,
sprites, and result screen. The production URL serves the audio and font files.

You can deploy the game to the `surpassing-the-leader` project under
`james-cenawoods-projects`. The serving package contains FastAPI and the
certified DTH artifact. Vercel serves the browser and prepared art from its CDN.

Build the artifact on your workstation. A Vercel build exceeded 45 minutes;
packaging the existing artifact avoids that solve. The preparation command
checks its source digest, array hashes, and opening policy before copying it.
The deployed function opens that copy with `verify_hashes=False`: the bundle
is immutable, and rehashing 2.3 GB at each cold start delayed the first
answer by about ten seconds. The manifest, schema, source digest, and array
contracts are still checked at every start.
Vercel CLI installs Linux dependencies for the deployment, including when you
run the build on macOS. We checked that Vercel's Linux rule-profile digest
matches the local artifact.

```sh
uv run python -m arena.web.prepare_vercel
npx vercel build --cwd src/arena/web/build/vercel --yes
uv run python -m arena.web.prepare_vercel --bytecode
npx vercel deploy --cwd src/arena/web/build/vercel --prebuilt --archive=tgz --target preview
```

Vercel ships no bytecode and sets `PYTHONDONTWRITEBYTECODE`, so a new process
compiled numpy, scipy, and FastAPI from source. The `--bytecode` step compiles
each Python file that the built function maps, with Python 3.13 and the
`unchecked-hash` mode, and adds the results to the function's file map. It cut
the in-process share of a boot from about 4.3 s to about 2.7 s.

You need a linked Vercel project at `.vercel/project.json`. The preparation
command copies that link into the generated directory. You can pass
`--artifact PATH` to select a different complete artifact. Generated packages
and tablebases remain outside Git. Use the prebuilt archive upload; a direct
upload of the 2.3 GB NumPy array exceeds the upload request limit.

Set `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` as a Config variable for Preview and
Production. Keep Fluid Compute enabled. The function bundle needs Vercel's
5 GB Large Functions beta; the standard Python limit is 500 MB. See
[Vercel's Python runtime](https://vercel.com/docs/functions/runtimes/python).

Connect an Upstash Redis store to the project for Preview and Production.
The server uses `KV_REST_API_URL` and `KV_REST_API_TOKEN`. It also accepts
`UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN`. Keep the token in
Vercel environment settings. The tablebase is part of the function bundle;
Redis stores the per-player command log.

The game ledger and leaderboard use the Supabase project
`surpassing-the-leader` (`zsipmcrvxtgmoufcswgo`). Set `SUPABASE_URL` and
`SUPABASE_SECRET_KEY` for Preview and Production; the server also accepts the
legacy `SUPABASE_SERVICE_ROLE_KEY`. Copy the secret key from the Supabase
dashboard and keep it in Vercel environment settings. Without these two
settings the game plays as before, records nothing, and serves no leaderboard.

Keep one deployed copy of the tablebase. After you verify a replacement on
the production alias, remove the superseded deployments by their IDs. Preserve
the deployment that serves `surpassing-the-leader.vercel.app`. Vercel counts
retained function bundles toward
[Function Storage](https://vercel.com/docs/deployment-storage).

For an external uptime monitor, use `GET /api/health` and expect HTTP 200 with
`status: "ok"`. This route requires no cookie and creates no game session.
The server finishes tablebase validation before it serves this route. Monitor
requests can reduce idle cold starts, but Vercel can replace an instance or
start another instance. A monitor does not guarantee a warm response.

You can check `/api/health`, then play a game through the preview URL. Check
that a fresh browser starts a separate session. Promote the tested deployment:

```sh
npx vercel promote DEPLOYMENT_URL --yes
```

## Hosted sessions

The hosted adapter gives each player an opaque, secure HTTP-only cookie.
Redis retains accepted commands and private random seeds for seven days.
The server reconstructs a session through the existing arena HTTP handlers,
then uses an atomic compare-and-set before it returns a new reveal. A losing
concurrent submission receives a conflict response with no speculative action.
A new worker can recover play from the same cookie and Redis log.
The browser requests `/api/session/restart` on page load, so opening or reloading
starts a fresh game. "Next game" opens a fresh record in the same way: each
replayed command costs the next request about 2.6 ms, and a five-game series
under one record reached 0.5 s for each click. Worker recovery replays the
current game's commands; an ordinary session read does not restart the game.

Each process keeps the games it last served, up to `HELD_GAMES`, beside the
Redis text each game matches. Redis stays the authority: every request reads
the record first. A held game whose text still matches plays the request with
no rebuild. A held game that another process moved past plays only the
commands it lacks. A lost compare-and-set or a refused command drops the held
game, so the next request rebuilds it from the command list. The client's
cookie-less `/api/rules` warm-up read is answered from one unplayed game.

Vercel gives a request to a new process when the request arrives 0.55 s to
0.80 s after a process's last answer, even while that process sits idle. We
swept the gap between one answer and the next request on a preview: gaps in
that window reached a new process in 25 of 86 requests, and gaps of 0.3 s,
0.5 s, and 0.85 s to 1.5 s did so in 0 of 108. The route, memory use (245 MB
of 2 GB), and log volume made no difference. A new process costs the request
5 s to 7 s: near 3.5 s before the Python process exists and near 2.5 s for
Python and its imports. A fast player's Continue and Commit presses fall in
that window, so about one move in three stalled. The browser client therefore
holds any request that would leave 0.35 s to 1.0 s after the last hosted
answer until that second ends (`webclient/src/pace.ts`). Outside the window a
long run still met one new process in about 60 s to 80 s of steady play.

The client also sends a second copy of a request that has no answer after
0.9 s (`webclient/src/hedge.ts`). The compare-and-set commits one copy of a
move, both copies compute the same reveal from the same seeds, and the refused
copy reveals nothing.

Each response carries an `x-stl-diagnosis` header: a random process id, the
process's uptime and request count, the handler's time, the held-game result
(`hit`, `behind`, `miss`, or `rules`), the process age, and the bytecode state.
A stalled request with `n=1` and a small `up` is a process boot.

The server shares one immutable tablebase reader across players. Each player
has a separate policy sampler. The hosted API hides random seeds and refuses
client overrides of the seed, opening clock, and round limit. Code-version
changes invalidate old command logs so the server cannot replay a game with
a different referee or sampler.

## Game ledger and leaderboard

`web/ledger.py` writes one Supabase row for each game. The server rewrites the
row after each resolved half-round, once the session store has committed that
half-round, so a closed tab leaves its moves behind and a losing concurrent
request writes nothing. A restart marks the old series' open game `abandoned`.
A ledger failure is logged and does not block the reveal; the closing
acknowledgement repeats the final write. Rows hold the public history, both
players' final loads, the code version, and the private seeds for offline
replay. No seed or player identifier reaches a browser.

A second cookie, `stl_player`, identifies the browser for one year. The session
cookie changes with each code version; the player cookie keeps a standing
across deployments. The ledger stores its SHA-256 digest only.

The leaderboard ranks each player's best win. A win scores
`max(0, 300 - ttd_seconds - cylinder_seconds)` for the winner, the seconds of
life left. A later loss, stopped game, or abandoned game leaves the standing
in place. The rule lives in the `leaderboard` database function (migration
`leaderboard_ranks_best_win`), so a change to it needs no deployment.
Ties order by fewer half-rounds, then by earlier finish. The score uses game
facts only, so it holds in the leap window, where DTH has no value for 61. The
browser shows the board after the win screen. A winner posts a name of at most
16 characters. The server normalizes it to NFC and refuses control, format, and
other invisible characters, names with no letter or digit, and runs of more
than two combining marks. `web/names.py` refuses slurs and a few hate terms,
through spacing, repeated letters, digit substitutions, and Cyrillic or Greek
lookalikes; ordinary profanity passes. `names.py` stays out of the version
digest on purpose, because a word-list change must not end active games. A
review left three known gaps in the filter. A short word beside a whole-word
term spelled with gaps hides that term. One stem in the inside-a-word tier
refuses ordinary derivatives such as "Retardant". A doubled letter lets
"Rapping" and "Whittler" match a term. A name needs a recorded
game, and a posted name holds for one minute before the next change. The
leaderboard routes carry no rate limit; use the Vercel firewall if scripts
abuse them. A misconfigured ledger setting is logged and disables the ledger;
it never stops the game.
Both tables enable row level security with no policy, and the four database
functions grant execute to `service_role` alone.

The local `python -m arena.web` surface retains its one-player process model.
It keeps no ledger, answers 404 for the leaderboard, and the browser then skips
that screen.
Neither deployment mode changes the canonical referee or the leap rule.

## Translated Hal candidate

You can run the frozen candidate on the canonical local browser surface:

```sh
uv run python -m arena.web --hal-agent perfect-hal --perfect-hal-model translated-v1 --dth-complete-tablebase outputs/perfect-hal-bayes-v2/tablebase --conceal-hal-details
```

Add `--pure-dth` to use the benchmark's permanent 1..60 game. Old remains
available as `--perfect-hal-model v1`; the Bayesian and ensemble selectors
keep their existing behavior. The selected variant reads its frozen parameters
from `src/arena/config/translated_hal_v1_selection.json`.

For hosted activation, set `STL_HAL_POLICY=translated-v1` in the target
environment and use the existing preparation and preview commands above.
Use `--artifact outputs/perfect-hal-bayes-v2/tablebase` with the preparation
command to package the evaluated artifact. The package includes the candidate
and its memory adapter. Its policy package uses direct imports so the serving
process does not need Torch or training modules. This work did not publish a
preview or change production settings.

Check that `/api/health` reports `policy: "translated-hal-v1"` on your preview.
Play through a reload and a next game, then use another browser to check
isolation. The tests cover worker replacement and rejected compare-and-set
requests. After you review the preview, you can use the existing promotion
command. Set `STL_HAL_POLICY=exact` and redeploy to restore hosted exact Hal.
An unset variable also selects exact Hal. A policy or source-version change
invalidates old session records and their memory.

The memory boundary is the secure `stl_session` cookie, with the existing
seven-day session lifetime. Reloads and next games retain revealed evidence;
cookie removal, expiry, or a version change starts a new model. The separate
leaderboard cookie does not restore model memory across versions or devices.
The server keeps a compressed JSON checkpoint at each game boundary and
replays the current game's accepted commands after restoring it. One Redis
compare-and-set commits both. A rejected mutation changes neither durable
evidence nor the revealed response. Malformed memory fails with HTTP 503.
No new Redis key family or database migration is required.

The candidate uses DTH equilibrium during canonical leap turns. Baku may
submit 61 as Dropper; Checker remains capped at 60. The adapter excludes leap
reveals from its 60-action model and clears stale sequence references.

You can repeat the local operational check with a new output path:

```sh
uv run python -m arena.web.check_translated_hal --artifact outputs/perfect-hal-bayes-v2/tablebase --output outputs/translated-hal-v1/runtime-review.json
uv run python -m pytest src/arena/tests src/terminal/tests tests/meta -q
npm --prefix src/arena/webclient run typecheck
```

The recorded local run covered 32 games and 16 action-61 reveals. It measured
1.56 ms p95 policy latency, 13.2 ms p95 request latency, and 62.2 ms p95 worker
recovery. The largest checkpoint was 6,788 bytes. These ASGI measurements use
the real tablebase and an in-memory CAS store; they exclude Redis network
latency and hosted cold starts. Keep the preview check before production
promotion. The compact evidence record links the statistical and runtime
reports, including the first holdout's failed uncertainty gate.

The arena and architecture suite passes 395 tests. The full repository suite
passes 837, skips one, and fails eight checks against existing stale Rust
extensions and `src/dth/artifacts/complete_fast_v1`. We preserved those
artifacts and the solver validation gates. Package the evaluated artifact
named above; its source and array checks pass.

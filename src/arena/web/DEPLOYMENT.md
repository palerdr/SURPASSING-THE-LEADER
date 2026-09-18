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
npx vercel deploy --cwd src/arena/web/build/vercel --prebuilt --archive=tgz --target preview
```

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
starts a fresh series. Worker recovery still replays the accepted restart and
subsequent commands; an ordinary session read does not restart the game.

The server shares one immutable tablebase reader across players. Each player
has a separate policy sampler. The hosted API hides random seeds and refuses
client overrides of the seed, opening clock, and round limit. Code-version
changes invalidate old command logs so the server cannot replay a game with
a different referee or sampler.

The local `python -m arena.web` surface retains its one-player process model.
Neither deployment mode changes the canonical referee or the leap rule.

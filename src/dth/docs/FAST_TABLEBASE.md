# Certified recurrence tablebase

You can build the arena's complete pure-DTH policy artifact with:

```sh
uv run python -m dth complete --config-name complete_fast_v1
uv run --project src/browser python -m browser --dth-complete-tablebase src/dth/artifacts/complete_fast_v1
```

Use `output_dir=src/dth/artifacts/complete_full_v1` on the build command
if you want the arena's default location. Keep an existing incompatible
artifact in a separate directory before building into that location. The
builder rejects a stale checkpoint; it does not replace its provenance.
You need a C11 compiler (`CC`, default `cc`) for this backend. The loader
compiles a content-addressed library under `src/dth/artifacts/kernel-cache/`.
The play-time Python provider does not compile or load that library.

## Method and certificate

We adapted the recurrence from [palerdr/dth, commit 1ef73c93](https://github.com/palerdr/dth/tree/1ef73c93ea99a00854ed3b6fea10fc3e30045f2a).
Python owns the profile tables and game rules. The C kernel processes 16
classes per batch with binary64 arithmetic, round-to-nearest, gradual
underflow, and no fast-math or fused multiply-add.

For the 60 success continuation values `S` and failed-check value `F`, set
`delta = S[0] - F`, `r[0] = 1`, and:

```text
b[k] = (S[k-1] - S[k]) / delta
r[k] = sum(b[k-j] * r[j], j = 0..k-1), k = 1..59
p = normalize(max(r, 0))
q = reverse(p)
```

These indices address arrays. Players choose literal seconds 1..60.
The recurrence costs O(60²), replacing a general equalizer factorization.
We reject nonfinite weights and degenerate denominators.

We evaluate all pure deviations against the full matrix. Its reversal
identity, `(pᵀ M)[j] = (M q)[59-j]`, lets the batch kernel obtain both bounds
from 60 payoffs. We require `upper - lower <= 1e-6` and store their midpoint.
We retain this numerical certificate on the upstream fast route, including
nonnegative weights. Python reconstructs both products with NumPy for live
policies. A rejected recurrence enters the existing certified LP ladder.

## Artifact contract

The fast backend uses artifact schema `dth.complete-tablebase.v3`, build
schema `dth.complete-tablebase-build.v3`, backend `c`, and ladder
`pure/recurrence/lp-v1`. It requires `warm_start=false`. Route bytes remain
0 for pure, 1 for support, and 2 for LP. `full_support_hits` counts accepted
recurrence policies, including clipped policies that pass the certificate.
The v2 Python/Rust ladder and its byte-parity contract retain their order.

The manifest binds both kernel sources, the existing rule/solver sources,
and the lockfile. The reader checks source digests and array hashes, with
no stale-artifact fallback. The builder retains the independent LP-based
recertification of four classes per layer, 4,792 classes in a full build.

The fast preset commits after each group of 50 layers. A requested layer
limit or completion forces a commit. A crash can lose work in the current
group; resume starts from the last durable group and overwrites its unfinished
successor layers. The builder joins workers before crossing each layer.
Set `checkpoint_every=1` for a durable commit after each layer.

## Local verification, 2026-09-11

On this Mac with 12 workers, the build took 30.59 seconds including final
validation. It solved 289,374,121 classes: 334,177 pure and 289,039,944
recurrence, with zero LP residues. The root value is
`0.08985007281413855`. Independent sampled recertification had maximum value
difference `1.477e-7`, within the frozen gate.

We compared all values with the prior Rust artifact. The maximum difference
was `5.063e-7`. The independent C++ recurrence implementation matched all
289,374,121 Python values byte for byte on this machine. The opening Python
policy took about 0.75 ms after artifact loading; this is one local latency
sample. Neither result establishes cross-machine performance or byte parity.

The values occupy 2,314,992,968 bytes before the NumPy header. Route bytes
add 289,374,121 bytes. The recurrence changes build and policy reconstruction
time. It does not compress the tablebase or extend pure DTH to STL's leap
window. See the [browser deployment notes](../../browser/deploy/DEPLOYMENT.md).

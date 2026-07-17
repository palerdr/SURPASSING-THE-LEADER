# Action Timing Authority

This document is the repository-wide authority for Drop the Handkerchief action
times and Squandered Time (ST). Engine, solver, toy, learning, Rust, tests, and
generated-artifact schemas must use this convention.

## Documentary basis

The Chapter 493 rules spread reproduced on PDF page 6 of
[`SURPASSING THE LEADER- HAL DOC.pdf`](./SURPASSING%20THE%20LEADER-%20HAL%20DOC.pdf)
states that a turn is limited to one minute and that D and C each have to drop
and check once. It defines ST as the time before the handkerchief is found. The
accompanying analysis records Yakou rejecting the possibility of zero ST.

The fifth-round account on PDF page 32 is the tie-breaking example. Baku tries
to check in the first second, Hal responds by dropping immediately, and the
nominal "0 seconds" successful instant check produces one second of ST. The
same source later describes an instant turn as accumulating a single second.

These passages establish three constraints:

1. both players must act once; passing or `NO_CHECK` is not a legal action;
2. "0 seconds" is narrative shorthand for acting immediately in the first
   second, not a literal zero-valued solver action;
3. every successful check accumulates at least one second of ST.

The manga does not provide integer-indexed source code. The convention below
is the unique simple whole-second model used by this project that satisfies all
three constraints and reproduces the documented instant-success example.

## Authoritative literal-second convention

For a normal half-turn:

```text
drop action d  in {1, 2, ..., 60}
check action c in {1, 2, ..., 60}

success iff c >= d
successful ST = c - d + 1
failure iff c < d
failed-check penalty = 60 seconds
```

Action `1` means acting immediately during the first second. There is no action
`0`. A same-second pair succeeds and accumulates one second because the chosen
second is counted inclusively. Examples:

```text
d=1,  c=1  -> success, ST=1
d=1,  c=60 -> success, ST=60
d=20, c=25 -> success, ST=6
d=25, c=20 -> failed check
```

An immediate check is risky, not a guaranteed voluntary death: `c=1` fails
when `d>1`, but succeeds with one ST when `d=1`. This matches Baku attempting
an instant failure and Hal defeating it by dropping immediately.

During the Leap Second window, only Baku as Dropper may additionally choose
`d=61`; Checker actions remain capped at `60`, so that cell necessarily fails.
The `pure` game omits the leap-second extension entirely.

## Dense action indices

The canonical 62-wide policy tensors retain their existing indexing:

```text
index 0     illegal padding
indices 1..60 literal normal action seconds
index 61    Baku-only leap drop when legal
```

Changing ST from an exclusive difference to inclusive elapsed seconds does not
change canonical policy-head width or action indices.

## Bucketed toy convention

Toy bucket actions are also one-based. For five-second buckets, action `1`
means the first five-second bucket and action `12` means the final bucket.
ToySTL uses the same inclusive formula in bucket units:

```text
squandered_units = check_bucket - drop_bucket + 1
```

Thus a same-bucket toy success adds one five-second unit. This is deliberate
quantization: the exact manga-faithful minimum is one second, while the coarse
toy model rounds it to its minimum representable unit.

## Artifact compatibility

Any tablebase, exact target, checkpoint, replay, or report generated with
`ST = check - drop`, a literal action `0`, or a pass/no-check action is stale.
Such artifacts must not be mixed with artifacts generated under this timing
authority. New manifests must identify the inclusive-ST schema or bind a code
digest that includes this convention.

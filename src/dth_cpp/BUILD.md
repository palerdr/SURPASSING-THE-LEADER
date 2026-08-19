# Build the optimized C++ DTH solver

This is the single mathematical and implementation guide for the complete
pure-DTH backward induction. Start with the empty translation units in this
directory and work through the sections exactly in order. A section may use
only objects completed and tested in earlier sections. Do not skip ahead,
because later performance objects deliberately rely on earlier correctness
objects rather than restating their contracts.

The mathematical authority is:

- [`../../docs/ACTION_TIMING.md`](../../docs/ACTION_TIMING.md) for literal
  seconds and inclusive elapsed time;
- [`../../docs/REVIVAL_MODEL.md`](../../docs/REVIVAL_MODEL.md) for the single
  frozen revival surface;
- [`../dth/docs/GAME_AND_SOLVER.md`](../dth/docs/GAME_AND_SOLVER.md) for pure-DTH
  state order and Bellman signs;
- [`../../paper/dth_exact_solution.tex`](../../paper/dth_exact_solution.tex) for
  the quotient, potential, 61-class stage matrix, certificates, and sweep proof.

The finished executable computes the value of every one of the
`17,011^2 = 289,374,121` quotient classes. It stores one `float64` midpoint and
one routing byte per class. Every stored midpoint must be derived from policies
whose saddle gap against the full literal 60 by 60 matrix is at most `1e-6`.
Any missing child, malformed probability, failed numerical solve, or wider gap
terminates the build.

HiGHS is the sole production numerical backend. It owns model lifecycle,
simplex implementation, feasibility tolerances, status reporting, and raw
solution extraction. DTH code still owns every mathematical rung: the implicit
matrix, O(60) pure reduction, support selection, equalizer and fallback model
formulations, policy embedding, ladder order, and independent full-matrix
certificate. A HiGHS objective value is never accepted as a game value without
that certificate.

Each section has four parts:

1. **Prerequisites** names everything it is allowed to use.
2. **Object** defines the new mathematical or systems object completely.
3. **Algorithm** gives implementation-level pseudocode.
4. **Gate** is the test that must pass before continuing.

Pseudocode uses zero-based action indices `0..59`. A zero-based drop `d` and
check `c` still represent literal seconds `d+1` and `c+1`.

Use dollar delimiters for mathematics in this Markdown file: `$x$` inline and
`$$` on separate lines around display mathematics. Do not use `\(...\)` or
`\[...\]`; they are valid LaTeX but are not rendered consistently by the
repository's Markdown viewer.

The computation is a dense table fill, not a game-tree search:

$$
x\longrightarrow Q(x)\longrightarrow \Phi(Q(x))
\longrightarrow \{F,S_1,\ldots,S_{60}\}
\longrightarrow M(x)\longrightarrow \widehat V(Q(x)).
$$

Precompute one-player quotient transitions, visit each two-player class once
in descending potential, solve its implicit matrix game, and store its
certified value. Never allocate a state object or retain a matrix for each of
the 289,374,121 classes.

## 0. Establish the native build graph and pinned HiGHS dependency

### Prerequisites

Only the scaffold files already present in this directory.

### Object

The Mac already has Apple Clang through Xcode. Install CMake and Ninja once:

```sh
brew install cmake ninja
```

Keep C++20, warnings, and `-ffp-contract=off` from `CMakeLists.txt`. Do not add
`-ffast-math`: the implementation uses NaN as the unsolved sentinel, requires
finite-value checks, and intentionally fixes multiply/add order for numerical
reproducibility.

Populate the empty files initially with the smallest compilable graph:

- `dth.hpp`: include guard or `#pragma once`, namespace `dth`, no declarations
  yet;
- `exact.cpp` and `matrix_game.cpp`: include `dth.hpp`;
- `solve_tablebase.cpp`: a `main` that reports that the solver is not built and
  returns failure;
- `tests.cpp`: a `main` that returns success.

Then extend `CMakeLists.txt` with:

- static library `dth_solver` from `exact.cpp`, `highs_backend.cpp`, and
  `matrix_game.cpp`;
- executable `dth-solve-tablebase` from `solve_tablebase.cpp`;
- executable `dth-tests` from `tests.cpp`;
- executable `dth-highs-backend-tests` from `highs_backend_tests.cpp`;
- `dth_project_options` linked to the library and all three executables;
- `dth_solver` linked into all three executables;
- `find_package(Threads REQUIRED)` and `Threads::Threads` linked to the solver;
- both native test executables registered with CTest.

Pin HiGHS exactly to release `1.15.1`, commit
`04024d701f79feb8e2f18bc3df0dffc04ef05088`. First try an exact,
toolchain-compatible CMake package:

```cmake
find_package(HIGHS 1.15.1 EXACT CONFIG QUIET)
```

If that package is unavailable and `DTH_FETCH_HIGHS=ON`, use CMake
`FetchContent` with the pinned commit. Never track `master`, `latest`, or an
unpinned release tag. Disable the HiGHS executable, examples, tests, HiPO,
zlib, shared extras, and implicit threading for the embedded build. Require
the target `highs::highs` and link it privately into `dth_solver`.

An installed HiGHS library must use the same C++ compiler and runtime as this
project. In particular, do not link a prebuilt MSVC library into a MinGW build.
`HIGHS_DIR` may point to an exact compatible installation; the default
FetchContent path avoids that ABI choice.

`highs_backend.hpp` is a DTH-owned boundary and must not include `Highs.h`.
Only `highs_backend.cpp` includes the third-party header. No HiGHS type may
appear in `dth.hpp`, `matrix_game.cpp` APIs, checkpoint schemas, or storage
objects.

### Algorithm

```text
configure(debug preset)
build(debug preset)
run ctest(debug preset)

configure(release preset)
build(release preset)
run ctest(release preset)
```

Commands:

```sh
cmake --preset debug -S src/dth_cpp
cmake --build src/dth_cpp/build/debug
ctest --test-dir src/dth_cpp/build/debug --output-on-failure
```

### Gate

Both presets configure. The debug build links with AddressSanitizer and
UndefinedBehaviorSanitizer. Both test executables link the exact HiGHS release
and exit zero. The backend test reports version `1.15.1`. The placeholder
solver exits nonzero, proving that an incomplete implementation cannot be
mistaken for a tablebase build.

## 1. Define constants, identifiers, and inert data types

### Prerequisites

The native build graph from Section 0.

### Object

Add the following constants to `dth.hpp`. They are solver constants and must
not be configurable at runtime:

| name | value | meaning |
| --- | ---: | --- |
| `kActions` | 60 | actions per player |
| `kCapacity` | 300 | fatal ST capacity |
| `kPenalty` | 60 | fixed failed-check dose addition |
| `kAliveProfiles` | 16,711 | distinct eligible profiles |
| `kDeadProfileBase` | 16,711 | first failure-fatal sentinel |
| `kCanonicalProfiles` | 17,011 | quotient profiles per player |
| `kCanonicalClasses` | 289,374,121 | canonical two-profile classes |
| `kDeadRho` | 301 | potential replacement for dead TTD |
| `kMaxProfilePotential` | 600 | largest one-profile potential |
| `kMaxClassPotential` | 1,200 | largest class potential |
| `kSaddleTolerance` | `1e-6` | immutable acceptance gate |
| `kPolicyMassFloor` | `1e-9` | recorded-support threshold |
| `kUnsolvedKind` | 255 | routing byte before solution |

Use these scalar types everywhere:

```text
ProfileId  := unsigned 32-bit integer
ChildId    := signed 32-bit integer      // -1 means terminal Dropper win
ClassId    := unsigned 64-bit integer    // multiplication is never 32-bit
Potential  := unsigned 16-bit integer
```

Define inert aggregate types without adding behavior yet:

```text
ProfileTable:
    profile_count: size
    st[profile_count]: signed 16-bit
    ttd[profile_count]: signed 16-bit       // -1 means dead sentinel
    potential[profile_count]: Potential
    revival[profile_count]: float64
    success_child[profile_count][60]: ChildId
    failure_child[profile_count]: ChildId
    buckets[0..max_profile_potential]: arrays of ProfileId
    alive_id[300][301]: ChildId             // -1 when no alive quotient id

TransitionValues:
    success[60]: float64
    failed: float64

Policy:
    mass[60]: float64

Certificate:
    lower: float64
    upper: float64
    midpoint: float64
    gap: float64

SolverKind:
    Pure = 0
    Support = 1
    LinearProgram = 2

SolverRoute:
    Pure
    WarmSupport
    FullSupport
    LinearProgram

SolveResult:
    certificate: Certificate
    drop_policy: Policy
    check_policy: Policy
    route: SolverRoute

solver_kind_for(route):
    Pure route -> Pure kind
    WarmSupport or FullSupport route -> Support kind
    LinearProgram route -> LinearProgram kind

RouteCounters:
    pure, warm_support, full_support, linear_program: unsigned 64-bit
```

`ProfileTable` uses dynamic storage even for the canonical table. That permits
small synthetic tables to exercise the complete sweep without allocating the
2.4 GiB canonical artifact. Canonical construction later asserts the frozen
counts above.

In `tests.cpp`, build a dependency-free test runner: a `require(condition,
message)` helper throws on failure; `main` calls named test functions, reports
the first exception, and returns nonzero. No test framework is needed.

### Algorithm

```text
test_constant_products:
    require(kCanonicalProfiles * kCanonicalProfiles == kCanonicalClasses)
    require(kCanonicalClasses fits in uint32)
    require(ClassId can represent kCanonicalClasses - 1)
    require(kDeadProfileBase + kCapacity == kCanonicalProfiles)
```

Even though a canonical class fits in 32 bits, calculate every class address
and byte offset in `ClassId` or `size_t`. This prevents an accidental signed or
32-bit intermediate from corrupting the mapped file.

### Gate

The constants test passes under both debug sanitizers and Release. Empty
default instances of every aggregate can be constructed without allocation
errors or uninitialized reads.

## 2. Implement the scalar game rules

### Prerequisites

The constants and inert types from Section 1.

### Object

Write a live state from the current Dropper's perspective as

$$
x=(s_c,t_c,s_d,t_d),
$$

where the first profile belongs to the current Checker. Both players choose
literal seconds $d,c\in\{1,\ldots,60\}$. A check succeeds exactly when
$c\ge d$, with inclusive lag

$$
\ell=c-d+1.
$$

The `+1` is load-bearing: equal actions add one second. If
$s_c+\ell\ge300$, the current Dropper receives terminal payoff $+1$;
otherwise the successful child swaps roles:

$$
(s_c,t_c,s_d,t_d)\longrightarrow(s_d,t_d,s_c+\ell,t_c).
$$

On a failed check the dose is $q=s_c+60$. A successful revival resets the
Checker's ST, adds the dose to TTD, and swaps roles:

$$
(s_c,t_c,s_d,t_d)\longrightarrow(s_d,t_d,0,t_c+s_c+60).
$$

Failed revival again pays the current Dropper $+1$. These role swaps explain
the negated child values assembled later in Section 8.

Implement these pure functions in `exact.cpp` and declare them in `dth.hpp`:

```text
survives_injection(st, ttd)
revival_probability(st, ttd)
```

The dose is `q = st + 60`. Revival is possible exactly when:

```text
q < 300 and ttd + q <= 300
```

or equivalently:

```text
st <= 239 and st + ttd <= 240.
```

If eligible, the frozen probability is:

```text
0.95 * (1 - st / 240) * 0.75^(ttd / 60).
```

All divisions in that expression are floating-point divisions. Invalid scalar
coordinates are programming errors: ST must be `0..299` and TTD `0..300`.

### Algorithm

```text
function survives_injection(st, ttd):
    require 0 <= st < 300
    require 0 <= ttd <= 300
    dose := st + 60
    return dose < 300 AND ttd + dose <= 300

function revival_probability(st, ttd):
    if NOT survives_injection(st, ttd):
        return 0
    acute := 1 - float64(st) / 240
    chronic := pow(0.75, float64(ttd) / 60)
    probability := 0.95 * acute * chronic
    require probability is finite
    require 0 < probability AND probability < 1
    return probability
```

The complete sweep will call `pow` only while building the 17,011-row profile
table. It must never evaluate a transcendental function per class.

### Gate

Test all boundaries:

- `(0,0)`, `(239,0)`, `(0,240)`, and `(180,60)` are eligible;
- `(240,0)`, `(239,2)`, `(0,241)`, and every `st >= 240` are fatal;
- equality `ttd + st + 60 == 300` is eligible;
- fatal profiles return exactly zero;
- `(0,0)` returns `0.95` within machine rounding;
- every eligible point in the full `300 x 301` coordinate box produces a
  finite probability strictly between zero and one.

## 3. Enumerate the failure-fatal quotient

### Prerequisites

The scalar rule functions from Section 2 and the `ProfileTable` storage shape
from Section 1.

### Object

A profile is `(st, ttd)` while the next injection can be survived. Once it is
failure-fatal, TTD can never affect play again, so all fatal profiles at a
fixed ST share one sentinel `Dead(st)`.

Formally, with $A(s,t)$ denoting the eligibility predicate from Section 2,

$$
\varphi(s,t)=
\begin{cases}
(s,t), & A(s,t)=1,\\
(s,\bot), & A(s,t)=0.
\end{cases}
$$

This quotient is exact: successful checks only increase ST and therefore
cannot restore eligibility, while the next failed check from an ineligible
profile is terminal regardless of TTD.

The exact count is not a magic constant. TTD zero contributes 240 eligible
profiles; each TTD $t=60,\ldots,240$ contributes $241-t$. Thus

$$
240+\sum_{t=60}^{240}(241-t)
=240+\sum_{k=1}^{181}k
=16{,}711.
$$

Adding one dead sentinel for each ST $0,\ldots,299$ gives
$16{,}711+300=17{,}011$ canonical profiles.

Enumeration order is part of the artifact schema:

1. eligible profiles with TTD ascending over `{0} union [60,300]` and ST
   ascending inside each TTD;
2. 300 dead sentinels with ST ascending.

TTDs `1..59` are not enumerated. An eligible profile in that band is outside
the transition-closed artifact domain and must be rejected by state-facing
lookup code. Fatal profiles in that band still map to `Dead(st)` because fatal
TTD is behaviorally discarded.

### Algorithm

```text
function begin_canonical_profile_table:
    table.profile_count := 17,011
    fill table.alive_id with -1
    resize st, ttd, potential, revival, success_child, failure_child

    next := 0
    for ttd in sequence [0, 60, 61, ..., 300]:
        for st in 0..299:
            if survives_injection(st, ttd):
                table.alive_id[st][ttd] := next
                table.st[next] := st
                table.ttd[next] := ttd
                next := next + 1

    require next == 16,711

    for st in 0..299:
        id := 16,711 + st
        table.st[id] := st
        table.ttd[id] := -1

    return the partially built table
```

Define lookup only after enumeration:

```text
function quotient_profile_id(table, st, ttd):
    validate scalar coordinates
    if NOT survives_injection(st, ttd):
        return 16,711 + st
    id := table.alive_id[st][ttd]
    if id == -1:
        fail "eligible profile has off-domain TTD 1..59"
    return id
```

### Gate

Exhaustively verify:

- exactly 16,711 eligible profiles were emitted;
- the first profile is `(0,0)`;
- ids `16,711..17,010` are precisely dead ST `0..299`;
- enumeration contains no eligible TTD in `1..59`;
- representative-to-id round trips for every enumerated eligible profile;
- all fatal coordinate pairs at a given ST map to the same dead sentinel;
- eligible coordinate pairs with TTD `1..59` fail rather than aliasing another
  state.

## 4. Complete the per-profile transition table

### Prerequisites

The partially populated quotient table from Section 3 and scalar rules from
Section 2.

### Object

For every profile, precompute all rule-dependent quantities later class solves
will gather:

- potential `st + ttd` for an eligible profile, `st + 301` for a dead
  sentinel;
- revival probability, exactly zero for a dead sentinel;
- 60 successful-check child profiles, one for each inclusive lag `1..60`;
- one survived-failure child profile, or terminal when the current profile is
  already fatal.

`ChildId == -1` means the current Dropper wins immediately. No other negative
value is valid.

For a successful check, `grown = st + lag`. `grown >= 300` is terminal.
Otherwise TTD is unchanged and the grown profile is quotiented.

For a survived failure, the revived player's new profile is:

```text
(0, ttd + st + 60).
```

The resulting profile may itself be failure-fatal and therefore map to the
dead sentinel at ST zero.

### Algorithm

```text
function finish_profile_table(table):
    for id in 0..table.profile_count-1:
        st := table.st[id]
        ttd := table.ttd[id]
        alive := (ttd >= 0)

        if alive:
            table.potential[id] := st + ttd
            table.revival[id] := revival_probability(st, ttd)
        else:
            table.potential[id] := st + 301
            table.revival[id] := 0

        for lag in 1..60:
            grown := st + lag
            if grown >= 300:
                table.success_child[id][lag-1] := -1
            else if alive:
                table.success_child[id][lag-1] :=
                    quotient_profile_id(table, grown, ttd)
            else:
                table.success_child[id][lag-1] := 16,711 + grown

        if alive:
            revived_ttd := ttd + st + 60
            table.failure_child[id] :=
                quotient_profile_id(table, 0, revived_ttd)
        else:
            table.failure_child[id] := -1
```

The call to `quotient_profile_id` for `revived_ttd` cannot hit the forbidden
TTD band: an eligible parent has nonnegative TTD and the failure adds at least
60.

### Gate

Exhaustively verify every row:

- potential is in `0..600`;
- success children are either `-1` or a valid profile id;
- failure children are either `-1` or a valid profile id;
- dead success preserves the dead sentinel and increases ST by the lag;
- alive success preserves TTD until it crosses to a dead sentinel;
- every alive profile has a failure child and every dead profile has terminal
  failure;
- exactly 1,018,830 successful entries are live;
- exactly 16,711 failure entries are live.

## 5. Define class encoding and Bellman role swaps

### Prerequisites

The completed profile table from Section 4 and the identifier types from
Section 1.

### Object

A class is the ordered pair `(checker_profile, dropper_profile)`. Its dense
address is:

```text
class_id = checker_profile * profile_count + dropper_profile.
```

Every live transition swaps roles. If the current checker moves to
`child_profile`, the child class is therefore:

```text
child_class = dropper_profile * profile_count + child_profile.
```

This order is load-bearing. Reversing either multiplication silently reads a
different state while remaining in bounds.

### Algorithm

```text
function encode_class(table, checker, dropper):
    require checker < table.profile_count
    require dropper < table.profile_count
    return ClassId(checker) * ClassId(table.profile_count) + dropper

function decode_class(table, class_id):
    require class_id < profile_count * profile_count
    checker := class_id / profile_count
    dropper := class_id % profile_count
    return (checker, dropper)

function swapped_child_class(table, dropper, child_profile):
    require child_profile is valid
    return encode_class(table, dropper, child_profile)

function class_potential(table, class_id):
    (checker, dropper) := decode_class(table, class_id)
    return table.potential[checker] + table.potential[dropper]
```

### Gate

Verify encode/decode for the first class, last class, all four corners of the
profile square, and a deterministic sample of at least 100,000 pairs. Verify
that `(0,0,0,0)` encodes to class zero under canonical enumeration.

## 6. Build potential buckets and prove the executable DAG schedule

### Prerequisites

The transition table from Section 4 and class encoding from Section 5.

### Object

The per-profile potential is already stored. Group profile ids into buckets:

```text
B[a] = all profile ids with profile potential a, for a in 0..600.
```

A class layer is the disjoint union:

```text
Layer(P) = union of B[a] x B[P-a]
```

over values of `a` for which both bucket indices lie in `0..600`.

Every live profile transition must strictly increase profile potential. Since
a class transition merely swaps the unchanged dropper profile with the moved
checker profile, that profile-level inequality proves every live class child
lies in a strictly higher layer. The backward sweep can therefore process
`P = 1200, 1199, ..., 0`, with no search or same-layer dependency.

The potential is based on

$$
\rho(u)=
\begin{cases}
t, & u=(s,t)\text{ is eligible},\\
301, & u=(s,\bot)\text{ is dead},
\end{cases}
\qquad
\psi(u)=s+\rho(u),
$$

and $\Phi(p_c,p_d)=\psi(p_c)+\psi(p_d)$. The strict increase has four
exhaustive live cases:

- eligible success staying eligible increases $\psi$ by $\ell\ge1$;
- eligible success crossing to dead increases it by
  $\ell+301-t\ge\ell+61$ because eligible $t\le240$;
- survived failure staying eligible increases it by exactly 60;
- survived failure crossing to dead increases it by
  $301-(s+t)\ge61$ because eligible $s+t\le240$.

A dead profile's only live transition is success, which increases $\psi$ by
$\ell$; dead failure is terminal. Role swapping reorders the two addends of
$\Phi$ but does not change their sum. This is the mathematical reason a layer
barrier is sufficient and same-layer locking is unnecessary.

### Algorithm

```text
function build_buckets(table):
    make 601 empty arrays
    for profile in 0..profile_count-1:
        append profile to buckets[table.potential[profile]]
    preserve ascending profile-id order inside each bucket

function validate_profile_edges(table):
    live_success := 0
    live_failure := 0
    for profile in 0..profile_count-1:
        parent_phi := table.potential[profile]
        for child in table.success_child[profile]:
            if child >= 0:
                require table.potential[child] > parent_phi
                live_success := live_success + 1
        child := table.failure_child[profile]
        if child >= 0:
            require table.potential[child] > parent_phi
            live_failure := live_failure + 1
    require live_success == 1,018,830
    require live_failure == 16,711

function layer_size(P):
    total := 0
    for a from max(0, P-600) to min(600, P):
        total += size(B[a]) * size(B[P-a])
    return total
```

Compute all layer sizes before any solve. Their sum must equal the full class
count; this proves that rectangle enumeration visits every class once.

### Gate

Require all of the following exact results:

- 1,035,541 total live profile transitions were checked;
- buckets `241..300` are empty and all other structural buckets are as
  generated by the quotient;
- every class layer `0..1200` is nonempty;
- all 1,201 layer sizes sum to 289,374,121;
- the largest layer is `P=374` with 1,678,715 classes;
- no live profile edge has equal or lower potential.

Do not continue if any count differs. A fast sweep over the wrong DAG is not a
solver.

## 7. Create durable dense value and routing stores

### Prerequisites

Class counts and layer bounds from Section 6, plus `SolverKind` from Section 1.

### Object

Keep storage separate from the exact game sweep:

- declare `MappedFile`, `MappedArray<T>`, checkpoint records, and store
  lifecycle functions in `storage/durable_store.hpp`;
- implement `MappedArray<T>` in `storage/mapped_array.tpp` because template
  definitions must be visible wherever the template is instantiated;
- implement checkpoint serialization and store lifecycle in
  `storage/durable_store.cpp`;
- implement the mapping system calls in `storage/mapped_file_posix.cpp` and
  `storage/mapped_file_win32.cpp`, with CMake selecting the matching backend.

Implement a move-only `MappedArray<T>` with:

- a move-only `MappedFile` backend;
- element count and byte count;
- `create(path, count, initial_value)`;
- `open_existing(path, expected_count)`;
- bounds-checked indexing in debug builds;
- synchronous `flush()`;
- automatic unmapping and handle closure in `MappedFile`'s destructor.

Use raw data files with no embedded header so the mapping begins at file offset
zero, which is page aligned on both 4 KiB and 16 KiB Macs:

```text
values.bin       class_count float64 entries
solver_kind.bin  class_count uint8 entries
```

Initialize every value to quiet NaN and every routing byte to 255. Creation
uses `open(O_CREAT|O_EXCL|O_RDWR)`, `ftruncate`, `mmap`, chunked initialization,
`msync`, and `fsync`. Refuse to overwrite an existing artifact unless the CLI
later receives an explicit fresh-build option.

Define a fixed-version checkpoint record, serialized field by field rather
than by dumping a padded C++ struct:

```text
magic: 8 fixed bytes "DTHCPV1\0"
schema_version: uint32 = 1
config_id_length followed by bytes: "dth-cpp-complete-v1"
profile_count: uint64
class_count: uint64
completed_potential: int32
route counters: four uint64 values
```

`completed_potential` means that this layer and every higher layer are durable.
Its initial value is 1201. The next layer to solve is always
`completed_potential - 1`; complete is zero.

### Algorithm

```text
function create_stores(output_dir, class_count):
    create values.bin with class_count NaNs
    create solver_kind.bin with class_count bytes equal to 255
    flush both mappings
    atomically_write_checkpoint(completed_potential=1201, counters=0)

function atomically_write_checkpoint(record):
    open checkpoint.tmp with create/truncate
    serialize every integer in fixed little-endian order
    write all bytes, retrying interrupted writes
    fsync temporary file
    close temporary file
    rename checkpoint.tmp to checkpoint.bin
    fsync output directory

function open_resume(output_dir, expected_table):
    parse checkpoint.bin and reject wrong magic, schema, config, or counts
    validate exact byte sizes of values.bin and solver_kind.bin
    map both files read/write
    return mappings and checkpoint
```

The durable commit order after a layer is:

```text
flush values
flush solver kinds
atomically replace checkpoint
```

If interrupted before checkpoint replacement, resume repeats that layer. Each
class write is deterministic and idempotent, so repetition is safe.

### Gate

On a temporary 1,000-element mapping, test creation, NaN initialization,
writes, flush, close, reopen, exact persistence, byte-size rejection, config
rejection, and idempotent rewrite. Kill a test child process after flushing
data but before replacing the checkpoint; reopening must report the previous
completed layer.

## 8. Assemble the 61 continuation values of one class

### Prerequisites

The complete profile table, class encoding, strict layer order, and mapped
value store from Sections 4 through 7.

### Object

At class `(checker, dropper)`, the stage matrix has only 61 transition values:

- `success[lag-1]` for successful inclusive lag `lag=1..60`;
- one common `failed` value for every `check < drop` cell.

All values are from the current Dropper's perspective. A live transition swaps
roles, so a stored child value is negated. Terminal Dropper victory is `+1`.

For a survived failure, death occurs with probability `1-p`, while revival
continues to the role-swapped child with probability `p`. Preserve this exact
arithmetic order:

```text
failed = p * (-child_value) + (1 - p).
```

Do not allow fused multiply-add if parity with the pinned sweep is desired.

Writing the 61 values as $S_1,\ldots,S_{60}$ and $F$, the literal stage matrix
is

$$
M[d,c]=
\begin{cases}
F, & c<d,\\
S_{c-d+1}, & c\ge d,
\end{cases}
$$

where $d,c$ are zero-based indices but the subscript is the physical lag. For
four actions the structure is

$$
\begin{bmatrix}
S_1&S_2&S_3&S_4\\
F&S_1&S_2&S_3\\
F&F&S_1&S_2\\
F&F&F&S_1
\end{bmatrix}.
$$

The 3,600 cells therefore contain only 61 distinct continuation values. Keep
this matrix implicit during transition assembly, certification, and the pure
rung. A support rung copies only its selected `k` by `k` submatrix; the general
LP rung copies the full shifted matrix into reusable fixed scratch.

### Algorithm

```text
function assemble_transition_values(table, values, checker, dropper):
    result := TransitionValues

    for index in 0..59:
        child_profile := table.success_child[checker][index]
        if child_profile == -1:
            result.success[index] := 1
        else:
            child_class := encode_class(table, dropper, child_profile)
            stored := values[child_class]
            require stored is finite              // schedule assertion
            result.success[index] := -stored

    failure_profile := table.failure_child[checker]
    if failure_profile == -1:
        result.failed := 1
    else:
        child_class := encode_class(table, dropper, failure_profile)
        stored := values[child_class]
        require stored is finite
        p := table.revival[checker]
        result.failed := p * (-stored) + (1 - p)

    require all 61 results are finite and lie in [-1-1e-9, 1+1e-9]
    return result

function matrix_cell(t, drop, check):
    require drop and check are in 0..59
    if check >= drop:
        return t.success[check - drop]
    return t.failed
```

The successful array index is `check-drop`, not `check-drop+1`, because index
zero represents physical lag one.

### Gate

Construct deterministic synthetic child values and compare all 3,600 calls to
`matrix_cell` against an independent literal action expansion. Include:

- the main diagonal reading `success[0]`;
- the top-right cell reading `success[59]`;
- every cell below the diagonal reading the identical failure value;
- terminal success and terminal failure;
- one probabilistic failure with a hand-computed expectation;
- a deliberately unsolved NaN child causing immediate failure.

## 9. Normalize policies and certify against the full matrix

### Prerequisites

`TransitionValues` and `matrix_cell` from Section 8, and `Policy` and
`Certificate` from Section 1.

### Object

A numerical candidate is never accepted from its reported objective. It is
accepted only after both policies are cleaned, normalized, and checked against
all literal actions.

For Dropper policy `p` and Checker policy `q`, compute:

```text
L = min_c sum_d p[d] M[d,c]
U = max_d sum_c M[d,c] q[c]
gap = max(0, U-L)
midpoint = (L+U)/2.
```

`L` is what the Dropper guarantees; `U` is what the Checker concedes. The true
matrix value lies between them.

In mathematical form, for policies $p,q\in\Delta_{60}$,

$$
L=\min_c\sum_d p_dM[d,c],
\qquad
U=\max_d\sum_c M[d,c]q_c,
$$

so weak duality gives

$$
L\le\operatorname{val}(M)\le U.
$$

Only $U-L\le10^{-6}$ accepts a result, and the stored value is
$\widehat V=(L+U)/2$. This same test governs every route; an equalizer value or
LP objective is diagnostic, never independent proof of correctness.

Candidate-specific code decides the largest negative mass it will tolerate.
The common normalizer receives that limit, rejects anything below it, clips
remaining negative rounding noise to zero, sums in ascending action order,
and divides each mass by the sum. It rejects zero, nonfinite, or nonpositive
total mass.

### Algorithm

```text
function normalize_policy(raw, negative_limit):
    total := 0
    for action in 0..59:
        require raw[action] is finite
        require raw[action] >= -negative_limit
        if raw[action] < 0:
            raw[action] := 0
        total := total + raw[action]          // ascending sequential sum
    require total is finite and total > 0
    for action in 0..59:
        raw[action] := raw[action] / total
    return raw

function certify(t, raw_drop, raw_check, negative_limit):
    p := normalize_policy(raw_drop, negative_limit)
    q := normalize_policy(raw_check, negative_limit)

    lower := +infinity
    for check in 0..59:
        payoff := 0
        for drop in 0..59:
            payoff := payoff + p[drop] * matrix_cell(t, drop, check)
        lower := min(lower, payoff)

    upper := -infinity
    for drop in 0..59:
        payoff := 0
        for check in 0..59:
            payoff := payoff + matrix_cell(t, drop, check) * q[check]
        upper := max(upper, payoff)

    gap := max(0, upper-lower)
    require gap <= 1e-6
    midpoint := (lower+upper)/2
    require midpoint is finite and in [-1-1e-9, 1+1e-9]
    return cleaned policies and Certificate(lower, upper, midpoint, gap)
```

### Gate

Test a constant matrix, an asymmetric matrix with a known pure saddle, matching
pennies embedded in the first two actions, slightly negative rounding noise,
materially negative mass, nonfinite mass, zero total mass, and a deliberately
exploitable policy pair. The asymmetric test is mandatory: uniform policies on
symmetric games can conceal transposition and sign errors.

## 10. Implement the O(60) pure-saddle reduction

### Prerequisites

The implicit matrix from Section 8 and the certifier from Section 9.

### Object

Scanning all 3,600 cells merely to decide whether a pure saddle exists would
discard the matrix's Toeplitz structure. For a fixed Dropper row `d`:

- successful cells `c >= d` read the prefix
  `success[0], ..., success[59-d]`;
- if `d > 0`, every cell `c < d` reads `failed`.

Therefore, with `prefix_min[j] = min(success[0..j])`:

```text
row_min[d] = prefix_min[59-d]                         when d == 0
row_min[d] = min(prefix_min[59-d], failed)            when d > 0.
```

For a fixed Checker column `c`:

- successful cells `d <= c` read the prefix
  `success[0], ..., success[c]`, in reverse order;
- if `c < 59`, every cell `d > c` reads `failed`.

Therefore, with `prefix_max[j] = max(success[0..j])`:

```text
column_max[c] = max(prefix_max[c], failed)             when c < 59
column_max[c] = prefix_max[59]                         when c == 59.
```

The pure maximin is `max_d row_min[d]`; the pure minimax is
`min_c column_max[c]`. Min and max select existing values and introduce no
rounding. If `minimax-maximin <= 1e-6`, the lowest-index maximizing row and
lowest-index minimizing column form a certified near-saddle pair.

This reduction is required before any linear algebra because terminal and
nearly terminal regions contain pure games. It avoids allocating systems or
invoking the LP engine for those classes and provides exact security bounds
for only O(60) work.

### Algorithm

```text
function try_pure_saddle(t):
    running_min := +infinity
    running_max := -infinity
    for j in 0..59:
        running_min := min(running_min, t.success[j])
        running_max := max(running_max, t.success[j])
        prefix_min[j] := running_min
        prefix_max[j] := running_max

    maximin := -infinity
    best_drop := 0
    for drop in 0..59:
        candidate := prefix_min[59-drop]
        if drop > 0:
            candidate := min(candidate, t.failed)
        if candidate > maximin:              // strict keeps lowest tie
            maximin := candidate
            best_drop := drop

    minimax := +infinity
    best_check := 0
    for check in 0..59:
        candidate := prefix_max[check]
        if check < 59:
            candidate := max(candidate, t.failed)
        if candidate < minimax:              // strict keeps lowest tie
            minimax := candidate
            best_check := check

    if minimax-maximin > 1e-6:
        return no solution

    p := zero Policy; p[best_drop] := 1
    q := zero Policy; q[best_check] := 1
    return certify(t, p, q, negative_limit=0)
```

The certifier recomputes the same full security bounds and protects the code
from an incorrect prefix index even though the reduction is mathematical.

### Gate

For at least 10,000 deterministic random `TransitionValues`, expand the full
matrix and require bit-exact equality between:

- O(60) maximin and `max(row minima)`;
- O(60) minimax and `min(column maxima)`.

Also test pure acceptance, mixed rejection, exact ties, and all-failure/all-
success edge layouts.

## 11. Build the scoped HiGHS numerical backend

### Prerequisites

The pinned dependency from Section 0 and finite scalar arithmetic from earlier
sections. This backend does not know about `TransitionValues`, action indices,
warm neighbors, solver routes, or certificates.

### Object

Create `highs_backend.hpp` and `highs_backend.cpp`. The header exposes only
DTH-owned types:

```text
NumericStatus:
    Optimal
    Infeasible
    Unbounded
    InfeasibleOrUnbounded
    IterationLimit
    InvalidInput
    Failure

EqualizerRaw:
    drop_mass[60]
    check_mass[60]
    drop_value
    check_value
    iterations

CoveringRaw:
    x: Policy
    y: Policy
    sum_x
    sum_y
    iterations

HighsBackend:
    solve_equalizer(row_major_support_matrix, dimension, output)
    solve_covering(row_major_shifted_matrix, dimension, output)
    version()
    last_error()
```

`EqualizerRaw` indices are positions within the supplied support, not literal
actions. `CoveringRaw.x` and `.y` are unnormalized optimization variables, not
certified policies. The rung that calls the backend remains responsible for
interpreting either result.

Hide `Highs` behind a noncopyable, movable pImpl. Construct one persistent
backend per future DTH worker. Configure and check every option exactly once:

```text
output_flag = false
log_to_console = false
solver = "simplex"
simplex_strategy = 1        // serial dual simplex
parallel = "off"
threads = 1
random_seed = 0
presolve = "off"
primal_feasibility_tolerance = 1e-10
dual_feasibility_tolerance = 1e-10
small_matrix_value = 1e-12
simplex_iteration_limit = 10000
```

Every `clearModel`, `passModel`, and `run` status must be checked. Translate
model statuses into `NumericStatus`; do not leak a HiGHS enum. An optimal
result additionally requires a valid finite primal solution, feasible primal
status, the expected number of columns, a finite objective, and a nonnegative
iteration count.

`passModel` returning a warning means HiGHS changed or rejected part of the
model, commonly by dropping a coefficient at the small-matrix threshold. It
is not an optimal result. The support rung may treat that as a rejected
heuristic and continue to the general LP; the final LP rung must fail closed.

Remove the former hand-written `solve_linear` declaration, implementation, and
dedicated gate. HiGHS now owns that numerical plumbing; keeping two production
paths would make it unclear which contract each rung actually exercises.

### Algorithm

```text
function solve_model(model, output_columns):
    clear the prior HiGHS model without resetting options
    require passModel(model) == OK
    run
    translate model status
    if status is not Optimal:
        return translated status
    require run status == OK
    require primal solution is valid and feasible
    require every requested column and objective is finite
    copy raw columns in model order
    return Optimal
```

Never call `resetGlobalScheduler` while any worker exists. HiGHS uses a static
scheduler; all concurrent contexts therefore use the same pinned one-thread
setting. DTH supplies parallelism across independent class solves later.

### Gate

The isolated backend test must verify the exact version, invalid and nonfinite
input rejection, a feasible 2x2 equalizer, an infeasible equalizer, and a
shifted matching-pennies covering/packing pair with known raw solutions. Repeat
the cases through the same backend instance to prove model clearing preserves
options and does not reuse a stale solution.

## 12. Implement square-support equalizer solving

### Prerequisites

The HiGHS backend from Section 11, implicit matrix and policies from Sections
8 and 9, and full certifier from Section 9.

### Object

Add one rung-local storage type before implementing the function:

```text
MatrixScratch:
    matrix[60*60]: float64
```

`MatrixScratch.matrix` is a reusable row-major buffer. This section views its
first `k*k` entries as a compact support matrix; Section 13 reuses all 3,600
entries for the shifted full matrix. It contains no solver state and performs
no allocation. Section 18 later embeds it in permanent per-worker scratch.

For candidate Dropper support `D` and Checker support `C`, let
`k = min(size(D), size(C))` and retain the first `k` ascending indices from
each. The square submatrix is `M[D,C]`.

The Checker mixture `q_C` and equalized value `u_c` satisfy:

```text
[ M[D,C]   -1 ] [ q_C ] = [ 0 ]
[   1^T     0 ] [ u_c ]   [ 1 ].
```

The Dropper mixture `p_D` and value `u_d` satisfy:

```text
[ M[D,C]^T -1 ] [ p_D ] = [ 0 ]
[    1^T    0 ] [ u_d ]   [ 1 ].
```

Submit each system to HiGHS as a zero-objective equality-feasibility LP. The
probability variables have bounds `[0,+infinity]`; the equalized-value variable
is free. Identical row lower and upper bounds encode equality. Solving both
systems is required: one candidate mixture does not certify what the other
player can guarantee. `u_c` and `u_d` are diagnostic only; final value comes
from the full certificate.

The square-support object is a heuristic. Degenerate zero-sum games can have
supports of different sizes. Trimming to `k` and certifying against the full
matrix make an incorrect guess fail closed.

This intentionally changes one routing detail from the old Gaussian
primitive: a singular but feasible equality model can now succeed because
HiGHS may choose one feasible point. That is safe because nonnegative bounds,
normalization, and the full certificate still apply. It can change degenerate
policies, route counts, and warm-support records, so the artifact configuration
id must include `highs-equalizer-v1` and exact backend version/options.

### Algorithm

```text
function try_support(t, drop_indices, check_indices, backend, scratch):
    k := min(size(drop_indices), size(check_indices))
    if k == 0:
        return no solution
    D := first k ascending drop indices
    C := first k ascending check indices
    zero scratch.matrix over k*k
    for i in 0..k-1:
        for j in 0..k-1:
            scratch.matrix[i*k+j] := matrix_cell(t,D[i],C[j])

    status := backend.solve_equalizer(scratch.matrix[0:k*k],k,raw)
    if status is Infeasible, InfeasibleOrUnbounded,
       IterationLimit, or Failure:
        return no solution
    if status is anything other than Optimal:
        fail "invalid equalizer model or impossible backend status"

    for i in 0..k-1:
        require raw masses are finite

    raw_drop := zero Policy
    raw_check := zero Policy
    for i in 0..k-1:
        raw_drop[D[i]] := raw.drop_mass[i]
        raw_check[C[i]] := raw.check_mass[i]

    return certify(t,raw_drop,raw_check,negative_limit=1e-10)
```

The full-support attempt passes `D=C=[0,1,...,59]`, producing two 61x61
equality-feasibility models. This is the dominant solver path: the completed
reference artifact accepted roughly 99.8% of all classes through support
equalization.

Do not silently replace the two systems with the tempting upper-triangular
Toeplitz recurrence obtained after subtracting the failure constant. That
shortcut needs additional invertibility, full-support, symmetry, and numerical
assumptions; it does not cover partial warm supports and changes arithmetic
order. It can become a new measured rung only after independent derivation,
differential tests, and unchanged full-matrix certification.

### Gate

Test:

- a full-support matching-pennies embedding;
- an asymmetric full-support game;
- singular infeasible supports returning no solution;
- singular feasible supports producing a candidate that must still certify;
- a candidate with mass below `-1e-10` returning no solution;
- a wrong support that either fails or nevertheless passes the full certificate;
- full support on deterministic real transition values sampled from the
  existing artifact, compared with the stored value within `1e-6`.

No support attempt may return an uncertified value.

HiGHS model setup is much heavier than dense elimination. The reference route
counts imply hundreds of millions of full-support attempts and therefore two
HiGHS runs for nearly every non-pure class. Before a canonical build, benchmark
at least one million real full-support tuples using persistent worker-local
backends. Report models/second, backend iterations, and projected canonical
wall time at the intended outer worker count. Do not start the canonical sweep
if the projection exceeds the project's production budget; either retain
HiGHS-only semantics and optimize model reuse, or explicitly revise this guide
to a separately tested hybrid backend. Never silently claim the original
hours-scale performance after changing the dominant numerical engine.

## 13. Implement the HiGHS matrix-game LP fallback

### Prerequisites

The implicit full matrix and certifier from Sections 8 and 9, plus the scoped
HiGHS backend from Section 11. This section does not depend on successful
equalization; it comes later because it is the general fallback, not the
dominant fast path.

### Object

The original payoff matrix `M` lies in `[-1,1]`. Shift it by two:

```text
A = M + 2,
```

so every entry lies in `[1,3]` and the shifted value is strictly positive.
The row player's covering program is:

```text
minimize    1^T x
subject to  A^T x >= 1
            x >= 0.
```

Its dual is the column player's packing program:

```text
maximize    1^T y
subject to  A y <= 1
            y >= 0.
```

At optimum, normalize `x` into the Dropper policy and `y` into the Checker
policy. The shifted game value is `1/sum(x) = 1/sum(y)`; the original value is
that number minus two. As everywhere else, store only the independently
certified midpoint.

The rung builds `A` in fixed row-major Dropper-then-Checker order and hands it
to `HighsBackend::solve_covering`. The backend deliberately solves two
explicit models and returns both primal column vectors:

1. the covering minimization above, producing `x`;
2. the packing maximization above, producing `y`.

Do not recover `y` from HiGHS row-dual signs. Two explicit models make policy
orientation reviewable and keep the backend result independent of library
dual-sign conventions. HiGHS owns Phase I, bases, pivots, anti-cycling,
scaling, and termination statuses. DTH owns coefficient orientation and every
post-solve check.

For this shifted game, infeasible, unbounded, ambiguous, limited, warning, or
invalid-solution statuses indicate a broken numerical contract and are fatal.
Unlike support equalization, there is no later rung to absorb a rejection.

### Algorithm

```text
function try_linear_program(t, backend, scratch):
    for drop in 0..59:
        for check in 0..59:
            scratch.matrix[drop*60+check] :=
                matrix_cell(t,drop,check) + 2
            require 1 <= scratch.matrix[drop*60+check] <= 3

    status := backend.solve_covering(scratch.matrix,60,raw)
    if status is not Optimal:
        fail with backend status and last_error

    sum_x := sum raw.x in ascending action order
    sum_y := sum raw.y in ascending action order
    require sums and every raw mass are finite
    require sum_x > 0 AND sum_y > 0

    for check in 0..59:
        activity := 0
        for drop in 0..59:
            activity := activity + scratch.matrix[drop*60+check]*raw.x[drop]
        require activity >= 1-1e-9

    for drop in 0..59:
        activity := 0
        for check in 0..59:
            activity := activity + scratch.matrix[drop*60+check]*raw.y[check]
        require activity <= 1+1e-9

    require abs(sum_x-raw.sum_x) <= 1e-10*max(1,sum_x)
    require abs(sum_y-raw.sum_y) <= 1e-10*max(1,sum_y)
    require abs(sum_x-sum_y) <= 1e-8*max(1,sum_x,sum_y)

    shifted_from_x := 1/sum_x
    shifted_from_y := 1/sum_y
    require both shifted values are finite
    diagnostic_value :=
        (shifted_from_x + shifted_from_y)/2 - 2

    candidate := certify(t,raw.x,raw.y,negative_limit=1e-10)
    if candidate does not exist:
        fail "optimal HiGHS policies failed the full matrix certificate"
    require abs(candidate.midpoint-diagnostic_value) <= 1e-6
    return candidate
```

The shift is never present in the certificate. `certify` normalizes the raw
covering and packing columns into policies and evaluates the original
`TransitionValues` through `matrix_cell`.

### Gate

Test the complete HiGHS LP rung on:

- 1x1 analogues before fixing the dimension at 60;
- constant, diagonal, asymmetric, and matching-pennies matrices;
- highly degenerate matrices with repeated lower triangles;
- deterministic random `TransitionValues`, comparing the certified result to
  the established Python artifact/oracle within `1e-6`;
- injected nonfinite coefficients and each non-optimal backend status;
- repeated solves through one backend, proving no stale model, basis, or
  solution survives `clearModel`;
- identical inputs under outer worker counts 1, 2, and the production width.

Every successful result must pass both shifted-LP feasibility checks and the
unshifted full-matrix saddle certificate.

## 14. Assemble the first complete certified solver ladder

### Prerequisites

Pure reduction, full-support equalizer, HiGHS LP fallback, and common
certificate from Sections 10 through 13.

### Object

Create one function that accepts `TransitionValues`, a persistent worker-local
backend, and thread-local scratch. Its initial ladder is:

1. O(60) pure saddle;
2. full-support equalizer with all actions `0..59`;
3. HiGHS covering/packing fallback.

The function returns `SolveResult`, including both cleaned policies and its
route. A numerical rejection at one rung is not an error; it advances to the
next rung. Exhausting the final rung is a fatal error containing the class id
once the sweep calls it.

### Algorithm

```text
function solve_stage_initial(t, backend, scratch):
    if pure := try_pure_saddle(t):
        pure.route := Pure
        return pure

    full := [0,1,...,59]
    if support := try_support(t,full,full,backend,scratch):
        support.route := FullSupport
        return support

    if lp := try_linear_program(t,backend,scratch):
        lp.route := LinearProgram
        return lp

    fail "no solver rung produced a certified equilibrium"
```

Do not catch certificate failures from the LP and convert them into a value.
The only valid response after the final failure is to abort the tablebase
build.

### Gate

Build a corpus that forces each route. Require route kind, policy validity,
gap at most `1e-6`, and midpoint in range. Compare all results with an
independent oracle. Run at least one million stage solves in Release while
tracking DTH-owned allocations and route throughput. After setup, the pure path
must allocate zero heap objects per solve. The DTH scratch/model assembly on
the equalizer and LP paths must not allocate; HiGHS may allocate internally,
so report those allocations and elapsed time instead of asserting they do not
exist.

## 15. Build a complete sequential backward sweep

### Prerequisites

The potential buckets and strict DAG proof from Section 6, durable stores from
Section 7, transition assembly from Section 8, and certified initial ladder
from Section 14.

### Object

Implement the complete Bellman recurrence in one thread before adding warm
supports or parallel execution. This is the first point at which the project
can construct a complete tablebase for a small synthetic `ProfileTable`.

For layer `P`, enumerate each nonempty rectangle exactly once. Use Dropper as
the outer class loop and Checker as the inner loop. For a fixed Dropper, all 61
child reads lie in the same dense row `dropper * profile_count + child`, which
improves locality.

No class in a layer reads another class in that layer. No reachability search,
priority queue, recursion, hash table, or transposition table is involved.

### Algorithm

```text
function solve_one_class(
    table, stores, checker, dropper, backend, scratch):
    class_id := encode_class(table, checker, dropper)
    transitions := assemble_transition_values(
        table, stores.values, checker, dropper)
    result := solve_stage_initial(transitions, backend, scratch)
    stores.values[class_id] := result.certificate.midpoint
    stores.solver_kind[class_id] := byte(solver_kind_for(result.route))
    increment route counter for result.route
    return result

function solve_layer_sequential(
    table, stores, P, counters, backend, scratch):
    for a from max(0,P-max_profile_potential)
             to min(max_profile_potential,P):
        checker_bucket := table.buckets[a]
        dropper_bucket := table.buckets[P-a]
        if either bucket is empty:
            continue

        for dropper in dropper_bucket ascending:
            for checker in checker_bucket ascending:
                solve_one_class(
                    table, stores, checker, dropper, backend, scratch)

function sweep_sequential(table, stores, checkpoint, stop_after_layers):
    backend := one persistent HighsBackend
    scratch := one persistent MatrixScratch
    P := checkpoint.completed_potential - 1
    layers_done := 0
    while P >= 0:
        expected := precomputed layer_size[P]
        before := sum(route counters)
        solve_layer_sequential(
            table, stores, P, counters, backend, scratch)
        require sum(route counters)-before == expected

        stores.values.flush()
        stores.solver_kind.flush()
        checkpoint.completed_potential := P
        checkpoint.counters := counters
        atomically_write_checkpoint(checkpoint)

        layers_done := layers_done + 1
        if stop_after_layers exists AND layers_done == stop_after_layers:
            return incomplete
        P := P-1
    return complete
```

Build a synthetic profile table in `tests.cpp` with `N=40`:

```text
profile i has potential i
success lag L moves to i+L when in range, otherwise terminal
failure moves to i+7 while i+7<N, otherwise terminal
revival is a deterministic finite value in (0,1) for live failure rows
buckets group the one profile at each potential
```

This table has the same role swap, terminal sentinels, 61-class matrix, strict
potential increase, layer rectangles, mapped storage, and solver ladder as the
canonical sweep while requiring only 1,600 classes.

Independently resolve the synthetic classes by sorting all class ids in
descending class potential, without using layer rectangles. The independent
ordering may call the already-built stage solver, because this test isolates
the sweep schedule and addressing rather than matrix solving.

### Gate

- Sequential layer values and independently ordered values differ by at most
  `1e-6` for every synthetic class.
- The sum of route counters is exactly `N^2`.
- No unsolved NaN remains and every route byte is `0..2`.
- Interrupt after several layers, reopen, resume, and compare the complete raw
  files byte-for-byte with an uninterrupted sequential build.
- Deliberately reverse a child address in a test-only copy and require the NaN
  child guard to expose the schedule/address error.

At this gate the implementation is correct and completely runnable on a small
game, but not yet the final optimized canonical solver.

## 16. Add recorded support extraction and warm guesses

### Prerequisites

Clean policies returned by the complete initial ladder from Section 14 and a
working layer sweep from Section 15.

### Object

The optimized ladder first tries a small support copied from a spatial
neighbor in the immediately previous completed layer. A recorded support is:

```text
SupportRecord:
    class_id: ClassId
    drop[12]: signed 8-bit action ids, -1 padding
    check[12]: signed 8-bit action ids, -1 padding
```

Extract each side independently from a cleaned 60-action policy:

1. keep actions with mass strictly greater than `1e-9`;
2. if more than 12 remain, order by descending mass with ascending action id
   on exact ties and retain 12;
3. sort retained action ids ascending before recording;
4. pad unused slots with `-1`.

Pure solutions need not be recorded. Support and LP solutions are recorded,
because both carry policies that may seed the next layer.

For class `(checker,dropper)`, the two possible lag-one neighbors are already
solved and lie exactly one profile-potential step or more ahead:

```text
checker_neighbor = encode(success_child[checker][0], dropper)
dropper_neighbor = encode(checker, success_child[dropper][0]).
```

Apply the class encoding carefully: a support neighbor is a nearby class, not
a live role-swapped Bellman child. Therefore the formulas are:

```text
encode_class(table, shifted_checker, dropper)
encode_class(table, checker, shifted_dropper).
```

Try the checker-shift neighbor first, then the dropper-shift neighbor. Look up
records only in the previous layer's immutable, class-id-sorted support vector.
Use binary search. Do not chain from a support discovered earlier in the
current layer; chaining would make results depend on traversal and thread
order.

### Algorithm

```text
function extract_support(policy):
    candidates := actions with policy[action] > 1e-9
    if size(candidates) > 12:
        sort candidates by (-mass, +action)
        truncate to 12
    sort candidates by action ascending
    return candidates padded with -1

function warm_neighbor_ids(table, checker, dropper):
    ids := empty list
    shifted := table.success_child[checker][0]
    if shifted >= 0:
        append encode_class(table, shifted, dropper)
    shifted := table.success_child[dropper][0]
    if shifted >= 0:
        append encode_class(table, checker, shifted)
    return ids

function solve_stage_optimized(
    t, warm_records, neighbor_ids, backend, scratch):
    if pure := try_pure_saddle(t):
        return pure with Pure route

    for neighbor_id in neighbor_ids in stated order:
        record := binary_search(warm_records, neighbor_id)
        if found:
            D := nonnegative actions from record.drop
            C := nonnegative actions from record.check
            if warm := try_support(t,D,C,backend,scratch):
                return warm with WarmSupport route

    full := [0..59]
    if support := try_support(t,full,full,backend,scratch):
        return support with FullSupport route

    if lp := try_linear_program(t,backend,scratch):
        return lp with LinearProgram route

    fail closed
```

After a non-pure solve, create a `SupportRecord` for the current class and add
it to the current layer's output vector. At layer completion, sort that vector
by class id. Class ids are unique, so duplicates indicate duplicate work and
must fail.

### Gate

Test exact support thresholding, top-12 trimming, mass ties, ascending storage,
padding, neighbor address formulas, checker-first ordering, missing records,
and a wrong warm support failing into full support or LP. Run the same synthetic
layer in forward and reverse rectangle order; values and recorded supports
must match byte-for-byte because current-layer chaining is forbidden.

## 17. Make warm supports resumable and finish checkpoint semantics

### Prerequisites

Atomic base checkpointing from Section 7 and immutable previous-layer support
records from Section 16.

### Object

Add `warm_supports.bin`, an atomic snapshot containing only the most recently
completed layer's sorted records:

```text
magic: 8 bytes "DTHSPV1\0"
schema_version: uint32 = 1
potential: int32
record_count: uint64
records in ascending class-id order, serialized field by field
```

The support file's potential must equal `checkpoint.completed_potential`.
Initial potential 1201 has an empty support set. A mismatch is a fatal resume
error; silently dropping warm records would change routing and potentially
floating-point results.

The final durable commit order becomes:

1. flush values;
2. flush route bytes;
3. atomically replace `warm_supports.bin` with the current layer's support
   records and `fsync` the directory;
4. atomically replace `checkpoint.bin` with the completed potential and
   counters.

If interruption occurs after support replacement but before checkpoint
replacement, the tags disagree on resume. The safe recovery is to rerun the
last uncommitted layer using the support snapshot named by the checkpoint. To
make that possible without retaining two files, write support files by layer
name, for example `warm_supports_0374.bin`, atomically update the checkpoint,
then delete the older support file only after the checkpoint is durable. The
checkpoint names the authoritative support filename explicitly.

### Algorithm

```text
function commit_layer(P, current_supports):
    flush value and kind mappings

    support_name := format("warm_supports_%04d.bin", P)
    atomically_write_support_file(support_name, P, current_supports)

    new_checkpoint.completed_potential := P
    new_checkpoint.support_filename := support_name
    new_checkpoint.counters := cumulative counters
    atomically_write_checkpoint(new_checkpoint)

    delete prior support file if it is not support_name
    fsync output directory

function resume:
    read checkpoint
    read exactly checkpoint.support_filename
    require its embedded potential == checkpoint.completed_potential
    validate sorted unique class ids and action ranges
    rerun only checkpoint.completed_potential-1 and lower
```

Extend the checkpoint record from Section 7 with a length-prefixed support
filename. Increment its schema/config id when this field is introduced; old
incomplete checkpoints must fail rather than be guessed compatible.

### Gate

Exercise process termination at every boundary in the four-step commit. Every
restart must either load the last complete layer or fail with a precise schema
error; it must never consume supports from a different layer. Interrupted and
uninterrupted synthetic builds must produce byte-identical values, routes,
support records, and cumulative counters.

## 18. Define permanent per-worker scratch and work items

### Prerequisites

The optimized per-class solver from Section 16 and complete layer rectangle
enumeration from Section 15.

### Object

Parallelism is only within a potential layer. Define one `WorkerScratch` per
thread containing all mutable per-class numerical state:

```text
TransitionValues
prefix minima and maxima
two Policies and temporary Policies
MatrixScratch for row-major support/shifted matrix storage
fixed DTH-owned coefficient and bound assembly buffers
one persistent HighsBackend
thread-local RouteCounters
thread-local vector<SupportRecord>
thread-local first error string
```

Reserve the support vector before processing a layer. Reuse every numerical
array. There must be no DTH-owned allocation in transition assembly, pure
scanning, support-matrix gathering, or LP-matrix gathering. HiGHS owns its
internal memory and may allocate during `passModel` or `run`; the Release
benchmark from Section 12 measures that cost. Never construct a `HighsBackend`
per class.

A `WorkItem` identifies one checker bucket and a slice of one dropper bucket:

```text
WorkItem:
    checker_bucket pointer/span
    dropper_bucket pointer/span limited to this chunk
```

For every layer rectangle, divide the dropper bucket into chunks of at most
4,096 profiles. Bucket storage is immutable for the program's lifetime, so
work-item spans remain valid.

### Algorithm

```text
function build_work_items(table, P):
    items := empty
    for every nonempty rectangle B[a] x B[P-a]:
        checker_span := B[a]
        dropper_span := B[P-a]
        for start in 0,4096,8192,...:
            append WorkItem(
                checker_span,
                dropper_span[start : min(start+4096,end)])
    return items

function process_work_item(item, previous_supports, scratch):
    for dropper in item.dropper_span ascending:
        for checker in item.checker_span ascending:
            if shared_error_flag is set:
                return
            solve class with optimized ladder
            write its unique value and route byte
            update only scratch counters
            append only to scratch support vector
```

### Gate

For every synthetic and canonical layer-size calculation, require that work
items cover exactly the expected class count with no duplicate `(checker,
dropper)` pair. Under sanitizers, process representative maximum-size work
items with one scratch instance and verify no DTH-owned allocation occurs
inside the class loops after vector reservation. Separately record HiGHS-owned
allocation and solve time; do not fold it into the zero-allocation claim.

## 19. Implement the fixed thread pool and parallel layer barrier

### Prerequisites

Independent `WorkItem`s and per-worker state from Section 18. The strict DAG
proof from Section 6 guarantees there are no same-layer reads.

### Object

Create worker threads once at program startup and reuse them for all 1,201
layers. Spawning threads per layer adds avoidable overhead in the narrow end
layers. The pool needs:

```text
threads[]
mutex
condition_variable start_cv
condition_variable done_cv
generation counter
stop flag
pointer/span to current work items
pointer/span to current previous supports
atomic next_work_index
atomic workers_remaining
atomic error flag
```

Each worker owns a fixed `WorkerScratch`. At dispatch, all workers observe the
new generation, fetch work-item indices dynamically, and decrement
`workers_remaining` exactly once. The main thread does not advance or flush
until every worker has finished.

Each scratch's `HighsBackend` remains owned and used by that worker only.
Every backend is fixed to `threads=1` and `parallel=off`; the DTH pool is the
only source of parallelism. All concurrent HiGHS instances must use that same
thread setting because HiGHS coordinates them through a static scheduler.
Never reset the global HiGHS scheduler until the pool has stopped and every
backend has been destroyed.

All mapped writes are race-free because Section 18 proved work-item ownership
is disjoint. All mapped reads target higher completed layers. Previous support
records are immutable. Counters and current supports stay thread-local until
the barrier.

### Algorithm

```text
worker_loop(worker_id):
    seen_generation := 0
    forever:
        lock mutex
        wait start_cv until stop OR generation != seen_generation
        if stop: return
        seen_generation := generation
        unlock mutex

        while NOT error_flag:
            index := next_work_index.fetch_add(1)
            if index >= number_of_work_items: break
            try process_work_item(work_items[index], previous_supports, scratch[id])
            catch error:
                atomically set first error and error_flag

        if workers_remaining.fetch_sub(1) == 1:
            lock mutex
            notify done_cv

dispatch_layer(work_items, previous_supports):
    clear every worker's counters, support vector, and error
    publish immutable input spans
    next_work_index := 0
    workers_remaining := thread_count
    error_flag := false

    lock mutex
    generation := generation+1
    notify_all start_cv
    wait done_cv until workers_remaining == 0
    unlock mutex

    if error_flag: throw first recorded error

    sum counters in worker-id order
    concatenate support vectors in worker-id order
    sort supports by class id
    require no duplicate class id
    require solved class count == expected layer size
    return merged counters and supports
```

Pool destruction sets `stop`, increments the generation, notifies all workers,
and joins every thread. Do not detach threads.

### Gate

Run every synthetic test with thread counts 1, 2, and the machine default.
Require byte-identical value files, route files, final support records, and
counters. Run ThreadSanitizer in a separate build if supported by the installed
Apple toolchain; sanitizer incompatibility is not permission to skip the
deterministic multi-thread tests.

## 20. Replace the sequential layer body with the optimized parallel body

### Prerequisites

The correct sequential sweep from Section 15, warm ladder/checkpointing from
Sections 16 and 17, and fixed pool from Section 19.

### Object

The final layer operation is now:

1. build work items from potential rectangles;
2. dispatch every class using the same immutable previous-layer supports;
3. merge and validate results at the barrier;
4. durably commit value bytes, route bytes, current support records, and
   progress;
5. make current supports the next layer's immutable previous supports.

Only the scheduling mechanism changes. Class assembly, solver order,
certificate, storage address, and checkpoint semantics remain the already
tested objects from earlier sections.

### Algorithm

```text
function sweep_parallel(table, stores, checkpoint, pool, stop_after_layers):
    previous_supports := load support file named by checkpoint
    counters := checkpoint.counters
    P := checkpoint.completed_potential - 1
    layers_done := 0

    while P >= 0:
        items := build_work_items(table,P)
        expected := layer_size[P]
        (delta, current_supports) :=
            pool.dispatch_layer(items,previous_supports)
        require sum(delta routes) == expected

        counters := counters + delta
        commit_layer(P,current_supports)
        report elapsed solve time, commit time, classes/second, routes, ETA

        previous_supports := move(current_supports)
        P := P-1
        layers_done := layers_done+1
        if requested stop reached: return incomplete

    return complete
```

The report's ETA should use a rolling window of recent `(classes, elapsed)`
pairs rather than averaging from layer 1200, because layer widths and solver
routes change substantially over the sweep.

### Gate

- Full synthetic parallel output is byte-identical to the sequential output.
- Stop/resume at two different layers is byte-identical to an uninterrupted
  parallel run.
- A missing child, solver failure, duplicate work item, and support checkpoint
  mismatch each abort before committing the layer.
- Re-running an uncommitted layer produces identical bytes.
- Thread counts 1 and the chosen production width produce identical class
  values and routing.

## 21. Add finalization and post-build recertification

### Prerequisites

A completed parallel sweep and every earlier certificate/storage object.

### Object

Completion is not merely reaching potential zero. Finalization scans and
recertifies the artifact before writing `tablebase.json`.

First scan the dense files in bounded chunks:

- every value is finite and in `[-1-1e-9,1+1e-9]`;
- every route byte is `0`, `1`, or `2`;
- route counters sum to the exact class count;
- checkpoint potential is zero.

Then deterministically recertify four classes from every layer. Choose the
first four classes in the same rectangle/dropper/checker order used by the
sequential sweep. For each sample:

1. assemble its 61 transition values from the finished mapped table;
2. run the complete optimized ladder again;
3. require the new certificate gap at most `1e-6`;
4. require the stored midpoint and fresh midpoint differ by at most `1e-6`.

Using `min(4, layer_size)` gives 4,792 samples for the canonical table. Compute
that count from the generated layer sizes rather than hard-coding it.

The local certificate also gives a global error bound. If every accepted
matrix interval has width at most $\tau=10^{-6}$, its stored midpoint differs
from that local matrix value by at most $\tau/2$. Every live edge raises
potential by at least one, so induction over the already-built DAG gives

$$
|\widehat V(x)-V(x)|
\le\frac{1201-\Phi(x)}{2}\tau.
$$

At the root this is at most $0.0006005$. Because first-Dropper win probability
is $(1+V)/2$, its propagated probability error is half that value error. This
bound explains the published root interval; it does not weaken any local
$10^{-6}$ certificate.

Write a final JSON manifest using a small deterministic writer, not a JSON
dependency. Include:

```text
schema: "dth.cpp-complete-tablebase.v1"
config id
profile and class counts
maximum potential
saddle tolerance
route order: pure/warm-support/full-support/highs-covering-v1
HiGHS semantic version and source commit
pinned HiGHS options and feasibility tolerances
cumulative route counters
recertified sample count and worst midpoint difference
root class id and root midpoint
data filenames, byte sizes, and scalar types
```

The JSON manifest is informative; `checkpoint.bin` remains the resume
authority. Write the manifest atomically after all validation. Then update the
checkpoint phase from `sweep` to `complete` using a schema field added when the
checkpoint parser is implemented.

### Algorithm

```text
function finalize(table, stores, checkpoint):
    require checkpoint.completed_potential == 0
    scan values and routes in chunks
    require all validity conditions

    samples := 0
    worst_difference := 0
    for P from 1200 down to 0:
        for first min(4,layer_size[P]) classes in canonical layer order:
            t := assemble_transition_values(...)
            fresh := solve_stage_optimized(t, supports unavailable, scratch)
            // With no warm record, the ladder proceeds pure/full/LP and still
            // certifies the same matrix; warm support is an optimization only.
            difference := abs(fresh.midpoint - stored[class_id])
            require difference <= 1e-6
            worst_difference := max(worst_difference,difference)
            samples := samples+1

    root_profile := quotient_profile_id(table,0,0)
    root_class := encode_class(table,root_profile,root_profile)
    require root_class == 0
    root_value := stores.values[root_class]
    require root_value lies in paper interval 0.08985 +/- 0.00061

    atomically write tablebase.json
    atomically write complete checkpoint
```

Also verify the independent recorded anchor class:

```text
V(240,0,240,0) approximately 0.3372132166291093.
```

Both profiles are failure-fatal at ST 240, so state encoding maps their TTDs
to the dead sentinel. Use a `1e-9` comparison for this anchor. The paper's root
interval is wider because it includes accumulated local certificate error; do
not replace the mathematical bound with an unjustified bit-equality demand.

### Gate

Corrupt one value byte, one route byte, a checkpoint count, and a final
manifest field in isolated test artifacts. Finalization or subsequent
verification must reject each corruption it is responsible for detecting.
Every sampled class must recertify, the dead-band anchor must match, and the
root must lie in the published interval.

## 22. Finish the command-line executable

### Prerequisites

Canonical table construction, parallel sweep, checkpoint/resume, and
finalization from all preceding sections.

### Object

Replace the placeholder `solve_tablebase.cpp` with a thin CLI that owns no game
or solver logic. Supported options:

```text
--output PATH             required artifact directory
--threads N               positive worker count
--resume                  require and continue a compatible checkpoint
--fresh                   require an absent/empty target and create stores
--stop-after-layers N     test/checkpoint session limit
--verify-only             open a completed artifact and rerun final checks
--progress-every N        report every N completed layers; zero is quiet
```

Exactly one of `--fresh`, `--resume`, or `--verify-only` is required. Refuse
unknown options, missing values, nonpositive counts, `--fresh` on an existing
artifact, and `--resume` without a compatible checkpoint. Never recursively
delete or overwrite an output directory.

Default `--threads` to a configurable value derived from
`std::thread::hardware_concurrency()`, but print the selected number before the
sweep. On this 15-core Mac, benchmark 12, 14, and 15 workers; efficiency cores,
thermal limits, and storage commits make maximum reported concurrency an input
to measure rather than an axiom.

Install signal handlers only for `SIGINT` and `SIGTERM`. The handler sets an
atomic stop-request flag. The main loop observes it only after a layer barrier
and commits that completed layer before exiting. Signal handlers must not call
allocation, I/O, mutex, or checkpoint code.

### Algorithm

```text
main(arguments):
    parse and validate options
    table := build and exhaustively validate canonical profile table
    layer_sizes := build and validate all potential layers

    if verify-only:
        open completed stores read-only
        run final verification
        return success

    if fresh:
        create stores and initial checkpoint
    else:
        open compatible resume state

    create fixed worker pool
    finished := sweep_parallel(...)
    if finished:
        finalize(...)
        print root value and route totals
    else:
        print last durable potential and exit success-as-checkpointed
```

Use distinct exit codes for argument error, incompatible artifact, numerical
failure, I/O failure, incomplete requested checkpoint session, and completed
success. Document them in `README.md` when the implementation is finished.

### Gate

CLI tests cover every option combination, refusal to overwrite, fresh build,
two-stage resume, signal-at-barrier behavior, verification, and corrupted
artifacts. The CLI contains no duplicated formulas, transition logic, matrix
logic, or certificate arithmetic.

## 23. Run the implementation in the only safe order

### Prerequisites

Every preceding gate passes.

### Object

The final validation and production sequence is deliberately staged so a
multi-hour canonical run is the last action, not a debugging mechanism.

### Algorithm

Format and run the debug suite:

```sh
xcrun clang-format -i \
  src/dth_cpp/dth.hpp \
  src/dth_cpp/storage/durable_store.hpp \
  src/dth_cpp/storage/durable_store.cpp \
  src/dth_cpp/exact.cpp \
  src/dth_cpp/storage/mapped_array.tpp \
  src/dth_cpp/storage/mapped_file_posix.cpp \
  src/dth_cpp/storage/mapped_file_win32.cpp \
  src/dth_cpp/highs_backend.hpp \
  src/dth_cpp/highs_backend.cpp \
  src/dth_cpp/highs_backend_tests.cpp \
  src/dth_cpp/matrix_game.cpp \
  src/dth_cpp/solve_tablebase.cpp \
  src/dth_cpp/tests.cpp
cmake --preset debug -S src/dth_cpp
cmake --build src/dth_cpp/build/debug
ctest --test-dir src/dth_cpp/build/debug --output-on-failure
```

Build and test Release:

```sh
cmake --preset release -S src/dth_cpp
cmake --build src/dth_cpp/build/release
ctest --test-dir src/dth_cpp/build/release --output-on-failure
```

Run a checkpoint smoke session into this project's ignored outputs:

```sh
src/dth_cpp/build/release/dth-solve-tablebase \
  --fresh \
  --output src/dth_cpp/outputs/complete-v1 \
  --threads 12 \
  --stop-after-layers 20 \
  --progress-every 1
```

Resume it for another set of layers, then verify that a second resume is
accepted. Benchmark production worker counts on the same potential range using
separate output directories; never reuse a checkpoint with a different
configuration id.

After choosing the thread count, resume without a layer limit:

```sh
src/dth_cpp/build/release/dth-solve-tablebase \
  --resume \
  --output src/dth_cpp/outputs/complete-v1 \
  --threads 12 \
  --progress-every 1
```

Verify the finished artifact independently through its own read-only path:

```sh
src/dth_cpp/build/release/dth-solve-tablebase \
  --verify-only \
  --output src/dth_cpp/outputs/complete-v1 \
  --threads 1
```

Finally run the repository-wide checks required after integration:

```sh
uv run python -m pytest --collect-only -q
uv run python -m pytest -q
cargo test --workspace
npm --prefix src/arena/webclient run typecheck
opam exec --switch=stl-dth-ocaml -- dune build --root src/dth_ocaml
opam exec --switch=stl-dth-ocaml -- dune runtest --root src/dth_ocaml
graphify update .
```

### Gate

The implementation is complete only when:

- native debug and Release tests pass;
- synthetic sequential, parallel, interrupted, and resumed outputs agree;
- every canonical class is finite, routed, and covered exactly once;
- all deterministic recertification samples pass the unchanged `1e-6` gate;
- the dead-band anchor and root interval pass;
- `tablebase.json` reports 289,374,121 solved classes;
- `--verify-only` succeeds after a fresh process start;
- repository-wide validation and `graphify update .` succeed.

The expected canonical hot data is about 2.43 GiB: roughly 2.16 GiB of values
and 276 MiB of route bytes, plus small transition, support, checkpoint, and
manifest files. Claim an hours-scale build only if Section 12's one-million-
tuple HiGHS benchmark supports it at the intended worker count. If an estimate
returns to days, profile before changing mathematics; the usual causes are
scanning all classes once per potential, materializing 3,600 cells
unnecessarily, invoking the general HiGHS LP before the equalizer,
reconstructing a HiGHS context per class, or serializing work that is
independent within a layer.

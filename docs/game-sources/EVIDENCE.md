# Canonical game evidence

This ledger records the minimum documentary evidence used to freeze executable
rules. Transcriptions are intentionally short; consult the cited source for full
context.

Stable evidence IDs are the durable reference. Each section carries an HTML
comment marker holding its uppercase ID, plus a matching lowercase HTML anchor,
so other documents cite it as `EVIDENCE.md#e-cylinder-cap`. Line-number links
were used previously and are no longer permitted — they silently rot on every
edit of this file, and did.

## Sources

- `SURPASSING THE LEADER- HAL DOC.pdf` — rule commentary and chronology.
- `Leader-Deviation-Strategy.pdf` — deviation analysis and overflow evidence.
- [`IN_DEPTH_SUMMARY.md`](IN_DEPTH_SUMMARY.md) — a full round-by-round reading of
  the arc. Its per-turn state headers form a complete numeric ledger of the
  canonical match and are the strongest quantitative evidence available.

<!-- evidence:E-ST-NONZERO -->
<a id="e-st-nonzero"></a>
## E-ST-NONZERO — squandered time is not zero

Source: `SURPASSING THE LEADER- HAL DOC.pdf`, PDF page 6. Hal asks whether
squandered time can equal zero seconds, and Yakou denies that possibility.

<!-- evidence:E-INSTANT-CHECK -->
<a id="e-instant-check"></a>
## E-INSTANT-CHECK — an immediate check produces one second

Source: `SURPASSING THE LEADER- HAL DOC.pdf`, PDF page 32. Baku checks in the
first second, Hal drops immediately, and the narrated “0 seconds” instant check
is recorded as one second of squandered time.

Corroborated by the match ledger: at Round 5 Turn 1 Leader drops instantly and
Baku checks instantly, and Baku's accumulation moves `0M32S → 0M33S`, i.e.
exactly one second.

<!-- evidence:E-CYLINDER-CAP -->
<a id="e-cylinder-cap"></a>
## E-CYLINDER-CAP — the cylinder holds at most five minutes

Source: `SURPASSING THE LEADER- HAL DOC.pdf`, PDF pages 12–13. The account says
the cylinder holds at most five minutes and only five minutes are injected even
when broader accumulated exposure is greater.

<!-- evidence:E-TTD-EXACT -->
<a id="e-ttd-exact"></a>
## E-TTD-EXACT — exactly five cumulative minutes remains eligible

Source: `SURPASSING THE LEADER- HAL DOC.pdf`, PDF page 43. The account states
that revival remains possible at exactly five minutes and is lost only after
going over five minutes.

<!-- evidence:E-OVERFLOW -->
<a id="e-overflow"></a>
## E-OVERFLOW — crossing the cylinder limit loses immediately

Source: `Leader-Deviation-Strategy.pdf`, PDF page 127. The analysis describes a
state where failing to check before accumulation goes over five minutes causes
an immediate loss.

<!-- evidence:E-DOSE-COMPOSITION -->
<a id="e-dose-composition"></a>
## E-DOSE-COMPOSITION — the injected dose is vial contents plus sixty seconds

Source: [`IN_DEPTH_SUMMARY.md`](IN_DEPTH_SUMMARY.md) per-turn state headers.
Every failed check in the canonical match satisfies `dose = accumulation + 60`:

| Failed check | Accumulation before | Dose added to TTD | `s + 60` |
|---|---:|---:|---:|
| R1T1 Baku | `0M0S` | `1M0S` | 60 |
| R2T2 Leader | `0M24S` | `1M24S` | 84 |
| R6T1 Baku | `0M33S` | `1M33S` | 93 |
| R8T2 Leader | `1M34S` | `2M34S` | 154 |
| R9T2 Leader | `0M0S` | `1M0S` | 60 |

Five independent confirmations, zero exceptions. The failed-check penalty is
therefore a fixed 60-second injection added to whatever the vial already holds.

<!-- evidence:E-RESET-ON-REVIVAL -->
<a id="e-reset-on-revival"></a>
## E-RESET-ON-REVIVAL — surviving resets the vial and adds the dose to TTD

Source: same ledger. After each of the five injections above the surviving
player's accumulation reads `0M0S` on the next header, and their near-death
total is the previous total plus the dose. TTD is strictly cumulative and is
never reduced.

<!-- evidence:E-ST-INCLUSIVE-LEDGER -->
<a id="e-st-inclusive-ledger"></a>
## E-ST-INCLUSIVE-LEDGER — successful squandered time is inclusive

Source: same ledger, corroborating [E-INSTANT-CHECK](#e-instant-check). Narrated
drop and check seconds reproduce the stated accumulation only under
`ST = check − drop + 1`:

- R3T2: Baku drops at second 25, Leader checks at 60 → `60 − 25 + 1 = 36`,
  and the summary states “accumulating a full 36 seconds on Leader”.
- R4T1: Leader waits 7 seconds and drops at second 8, Baku checks at 10 →
  `10 − 8 + 1 = 3`, matching `0M29S → 0M32S`.
- R8T1: Leader waits 5 seconds and drops at second 6, Baku checks at 60 →
  `60 − 6 + 1 = 55`, and the summary states “accumulating 55 seconds”.

Twelve successful checks appear in the ledger; all lie in `1..60` and none is
zero.

<!-- evidence:E-MAX-ST -->
<a id="e-max-st"></a>
## E-MAX-ST — a single half-round contributes at most sixty seconds

Source: same ledger, R7T1. Baku is forced into the safe strategy, “Leader is
free to accumulate the maximum amount in Baku's cylinder”, and Baku's
accumulation moves `0M0S → 1M0S`. The maximum single-turn squandered time is
exactly 60 seconds, consistent with a 60-second half-round and inclusive ST.

<!-- evidence:E-TTD-SLACK -->
<a id="e-ttd-slack"></a>
## E-TTD-SLACK — the strict cumulative boundary, measured

Source: same ledger, final injection and the hallucination-round header. This is
the sharpest quantitative evidence in the corpus.

Before the final failed check Leader holds `s = 0`, `t = 3M58S = 238`. The dose
is `0 + 60 = 60`, giving `t + q = 298 ≤ 300`, and Leader survives. The summary
describes this as “reviving with 2 seconds of deviation”, and `300 − 298 = 2`.

The same passage states that the two seconds exist only because Leader “limited
his accumulation to 9 seconds” across Round 6 Turn 2 and Round 7 Turn 2 — the
ledger shows 8 and 1, summing to exactly 9. Had Leader accumulated `x` more
there, his TTD would have finished at `298 + x`, surviving only while `x ≤ 2`.

This confirms three separate rules at once: the cap is exactly 300 seconds, the
lethal test is on cumulative TTD rather than on a single dose, and equality at
300 remains revival-eligible — corroborating [E-TTD-EXACT](#e-ttd-exact) with a
worked case.

<!-- evidence:E-PROXIMITY-RISK -->
<a id="e-proximity-risk"></a>
## E-PROXIMITY-RISK — dying again soon is materially worse

Source: [`IN_DEPTH_SUMMARY.md`](IN_DEPTH_SUMMARY.md), Round 3 Turn 2 and Round 7
Turn 1. Leader “took the safe strategy to avoid dying too close together and
lowering his chances of revival”; later, “since Baku just died, it would be
extremely risky to die again so soon, so he is forced to use the safe strategy”.

Revival probability is therefore a decreasing function of prior accrued
time-to-death. The passages establish the direction and that the effect is large
enough to change optimal play. They do not supply a rate.

<!-- evidence:E-REFEREE-CONDITION -->
<a id="e-referee-condition"></a>
## E-REFEREE-CONDITION — the referee's condition affects revival

Source: same document, Round 1 Turn 1 and Round 5 Turn 1. Leader tries to
worsen Baku's revival odds by demoralising him and “therefore worsening his
physical condition”; later he draws attention to “Yakou's deteriorating physical
condition”, and the reverse psychology is described as “bolstering his strength
for Leader's next revival”. Yakou revives Baku by punching him hard enough to
break ribs.

Resuscitation is an active act by a referee whose capacity degrades. In
reduced formulations this is folded into the accrued-TTD term; see
[`REVIVAL_MODEL.md`](../REVIVAL_MODEL.md).

<!-- evidence:E-THREE-DEATHS -->
<a id="e-three-deaths"></a>
## E-THREE-DEATHS — three near-deaths is the practical ceiling

Source: same document, Round 4 Turn 1. The “3ND strategy” is defined as one
“wherein Baku can die thrice to maximise his total accumulation in his body +
cylinder without going overboard”.

“Body + cylinder” is `TTD + vial`, and “going overboard” is crossing the
300-second cap. The ledger bears this out: Leader dies exactly three times, for
84, 154 and 60 seconds, totalling 298 of the available 300. No probability
assumption is needed to explain the ceiling — the cap alone produces it.

<!-- evidence:E-STL-OPENING -->
<a id="e-stl-opening"></a>
## E-STL-OPENING — the match opens at 8:12 with Hal as Dropper

Source: `Leader-Deviation-Strategy.pdf`, PDF page 2, and
[`IN_DEPTH_SUMMARY.md`](IN_DEPTH_SUMMARY.md), Round 1 Turn 1. The route analysis
labels the first node `R1T1, 8:12 AM`. The ledger identifies Leader/Hal as
Dropper and Baku as Checker, with both players at zero accumulation and zero
prior near-death time.

<!-- evidence:E-LSR-VARIANTS -->
<a id="e-lsr-variants"></a>
## E-LSR-VARIANTS — four exact route congruence classes

Source: `Leader-Deviation-Strategy.pdf`, PDF pages 2--9. Round starts divide
into four minute classes: V1 starts at minutes `12 mod 4`, V2 at `13 mod 4`, V3
at `14 mod 4`, and V4 at `15 mod 4`. V2 is the active route. The analysis
tracks the canonical deaths from V1 to V4 to V3 and finally V2, with the active
round beginning at 8:57.

The labels are a presentation of exact clock arithmetic, not independent game
state. A counterfactual death must advance the clock by its actual duration and
then derive the new class.

<!-- evidence:E-HAL-MEMORY-SEQUENCE -->
<a id="e-hal-memory-sequence"></a>
## E-HAL-MEMORY-SEQUENCE — two distinct forgetting events precede the leap

Sources: `SURPASSING THE LEADER- HAL DOC.pdf`, PDF pages 17--18 and 73--74;
`Leader-Deviation-Strategy.pdf`, PDF pages 119--121; and the Round 2, Round 8,
and Round 9 entries in [`IN_DEPTH_SUMMARY.md`](IN_DEPTH_SUMMARY.md).

The analyses place Hal's deliberate first near-death in Round 2 after he has
recognized and opened the leap route, followed by suppression or forgetting of
that realization. Hal later recognizes the plan again before his Round 8
near-death. That second Hal near-death is explicitly labeled as leaving his TTD
at `3M58S`; the scheduled broad memory loss follows before Round 9. Yakou then
recaps the rules and prior events. Hal's Round 9 leap injection is his third
near-death and therefore cannot cause the forgetting that preceded it.

The sources disagree about the degree of subconscious carryover after each
event. They support the chronology, not a numerical off-path cognition model.

<!-- evidence:E-LEDGER-ERRATUM -->
<a id="e-ledger-erratum"></a>
## E-LEDGER-ERRATUM — one transcription error in the summary

The Round 5 Turn 2 header lists Baku's near-death as `1M24S`. Baku did not die
in Round 5 Turn 1 — the summary's own prose has him making “a perfect check” —
and the Round 6 Turn 1 header reads `1M0S` again. The correct value is `1M0S`.

With that one correction, all eighteen state headers replay exactly under the
frozen rules with no residual.

## Modeling boundary — revival odds are not identified

The cited passages establish capacity, injection composition, eligibility
boundaries, and the *direction* of the dose and prior-damage effects. They do
not supply an empirical or canonical numerical revival-probability surface at
any point.

Numerical decay constants, curve shapes, physicality multipliers, and
referee-fatigue surrogates are therefore explicit modeling assumptions. They
must be versioned, must be identical across every formulation in this
repository, and must never be presented as documentary odds. The frozen choice
and its justification live in [`REVIVAL_MODEL.md`](../REVIVAL_MODEL.md).

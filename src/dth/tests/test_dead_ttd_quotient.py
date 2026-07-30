"""The per-player TTD-dead quotient invariant.

A player who fails ``survive_injection`` can never survive one again, and their
TTD reaches the value function only through ``revival_model``, which is then
identically zero.  So that TTD may be canonicalized to a single sentinel without
changing any exact value.  This is the structural claim a quotiented closure
would rest on, so it is tested rather than assumed.
"""

from dth.solver import revival_model, survive_injection

TTD_VALUES = [0] + list(range(60, 301))
ST_VALUES = range(300)
PROFILES = [(st, ttd) for st in ST_VALUES for ttd in TTD_VALUES]


def test_survive_injection_is_the_documented_predicate() -> None:
    # capacity 300, dose q = s + 60: fatal when t + q > 300, eligible at equality
    # only while q < 300.
    for st, ttd in PROFILES:
        assert survive_injection(st, ttd) == (st <= 239 and st + ttd <= 240)


def test_failing_survival_zeroes_the_revival_probability() -> None:
    for st, ttd in PROFILES:
        if not survive_injection(st, ttd):
            assert revival_model(st, ttd) == 0.0


def test_failed_survival_is_absorbing_under_load_growth() -> None:
    # A successful check raises that player's ST and leaves their TTD alone; a
    # failed one ends the game for them unless they survive.  So ST growth is
    # the only motion available to a dead profile, and it must never revive it.
    for st, ttd in PROFILES:
        if survive_injection(st, ttd):
            continue
        for grown in range(st, 300):
            assert not survive_injection(grown, ttd)


def test_quotient_collapses_the_profile_space_as_claimed() -> None:
    quotiented = {
        (st, ttd if survive_injection(st, ttd) else None) for st, ttd in PROFILES
    }
    assert len(PROFILES) == 72_600
    assert len(quotiented) == 17_011
    # Two-player classes, against 5,267,489,760 reachable live states.
    assert len(quotiented) ** 2 == 289_374_121

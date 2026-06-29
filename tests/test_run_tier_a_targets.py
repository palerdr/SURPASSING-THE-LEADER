import os
import sys
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_tier_a_targets import build_state_specs


def _args(**overrides):
    defaults = dict(
        cylinder_points=2,
        near_boundary_points=0,
        ttd_points=2,
        seed=0,
        include_d1=True,
        death_filter="all",
        ttd_min=None,
        ttd_max=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_death_filter_d0_excludes_death_epoch_specs():
    specs = build_state_specs(_args(death_filter="d0"))

    assert specs
    assert all("hal_deaths" not in spec and "baku_deaths" not in spec for spec in specs)


def test_death_filter_d1_hal_only_emits_hal_death_specs():
    specs = build_state_specs(_args(death_filter="d1_hal", ttd_min=100, ttd_max=130))

    assert specs
    assert all(spec.get("hal_deaths") == 1 for spec in specs)
    assert all("baku_deaths" not in spec for spec in specs)
    assert all(100 <= spec["hal_ttd"] <= 130 for spec in specs)


def test_death_filter_d1_baku_only_emits_baku_death_specs():
    specs = build_state_specs(_args(death_filter="d1_baku", ttd_min=230, ttd_max=240))

    assert specs
    assert all(spec.get("baku_deaths") == 1 for spec in specs)
    assert all("hal_deaths" not in spec for spec in specs)
    assert all(230 <= spec["baku_ttd"] <= 240 for spec in specs)

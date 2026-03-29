import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.harvest_model_traces import build_parser, highest_stage


def test_harvest_model_traces_parser_defaults():
    args = build_parser().parse_args(["--model", "foo.zip", "--output", "bar.json"])

    assert args.role == "baku"
    assert args.opponent == "bridge_pressure"
    assert args.games == 128
    assert args.target_stage == "round7_pressure"
    assert args.max_traces == 64


def test_highest_stage_prefers_latest_progress_marker():
    assert highest_stage({"round7_pressure": True, "round8_bridge": True}) == "round8_bridge"
    assert highest_stage({"leap_window": True, "leap_turn": True}) == "leap_turn"

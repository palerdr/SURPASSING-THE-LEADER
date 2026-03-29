import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_retention_slices import GateDecision, SliceEval, build_parser, evaluate_gate


def make_eval(*, bridge_r7: float, hal_trade_r7: float, hal_pressure_r7: float, misaligned: int, seeded_win: float) -> SliceEval:
    return SliceEval(
        opening={
            "bridge_pressure": {"round7_rate": bridge_r7},
            "hal_death_trade": {"round7_rate": hal_trade_r7},
            "hal_pressure": {"round7_rate": hal_pressure_r7},
        },
        alignment={
            "round7_count": 0,
            "round8_count": 0,
            "round9_count": 0,
            "leap_window_count": 0,
            "misaligned_leap_window_count": misaligned,
        },
        seeded_round9={"wins": int(50 * seeded_win), "win_rate": seeded_win},
    )


def test_retention_parser_defaults():
    args = build_parser().parse_args(["--anchor-model", "anchor.zip"])

    assert args.slice_timesteps == 2048
    assert args.max_slices == 4
    assert args.max_misaligned == 5
    assert args.min_seeded_round9_win == 100.0
    assert args.anchor_trace_set is None
    assert args.anchor_trace_file is None
    assert args.anchor_teacher_demo_file == []
    assert args.anchor_bc_epochs == 0


def test_evaluate_gate_passes_when_candidate_matches_or_beats_anchor():
    anchor = make_eval(bridge_r7=0.16, hal_trade_r7=0.16, hal_pressure_r7=0.16, misaligned=0, seeded_win=1.0)
    candidate = make_eval(bridge_r7=0.20, hal_trade_r7=0.16, hal_pressure_r7=0.18, misaligned=0, seeded_win=1.0)

    decision = evaluate_gate(anchor, candidate, max_misaligned=5, min_seeded_round9_win=100.0)

    assert decision == GateDecision(passed=True, reasons=())


def test_evaluate_gate_rejects_bridge_regression_and_misalignment():
    anchor = make_eval(bridge_r7=0.16, hal_trade_r7=0.16, hal_pressure_r7=0.16, misaligned=0, seeded_win=1.0)
    candidate = make_eval(bridge_r7=0.10, hal_trade_r7=0.16, hal_pressure_r7=0.16, misaligned=8, seeded_win=1.0)

    decision = evaluate_gate(anchor, candidate, max_misaligned=5, min_seeded_round9_win=100.0)

    assert decision.passed is False
    assert any("bridge round7 regressed" in reason for reason in decision.reasons)
    assert any("misaligned leap-window count" in reason for reason in decision.reasons)


def test_evaluate_gate_rejects_held_out_regression_and_seeded_late_drop():
    anchor = make_eval(bridge_r7=0.16, hal_trade_r7=0.16, hal_pressure_r7=0.16, misaligned=0, seeded_win=1.0)
    candidate = make_eval(bridge_r7=0.20, hal_trade_r7=0.08, hal_pressure_r7=0.16, misaligned=0, seeded_win=0.9)

    decision = evaluate_gate(anchor, candidate, max_misaligned=5, min_seeded_round9_win=100.0)

    assert decision.passed is False
    assert any("hal_death_trade round7 regressed" in reason for reason in decision.reasons)
    assert any("seeded round9 win" in reason for reason in decision.reasons)

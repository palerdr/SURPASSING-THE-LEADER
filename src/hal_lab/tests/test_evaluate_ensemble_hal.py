"""Tests of the Perfect Hal policy ensemble v1 evaluator."""


def test_evaluation_clusters_emulator_replicates_and_counts_stops_as_nonwins():
    from hal_lab.experiments.perfect_hal_ensemble_v1.evaluate_ensemble_hal import CONTROLLERS, summarize
    sessions = [{"source": "human_fitted", "family": "human_response", "cluster": 1,
                 "controllers": {name: [{"won": True if name == "ensemble" else None, "seat": seat}
                                        for seat in ("Hal", "Baku")] for name in CONTROLLERS}}
                for _ in range(4)]
    results = summarize(sessions, {"bootstrap_seed": 42, "bootstrap_replicates": 100})["human_fitted"]
    assert results["ensemble"]["wins"] == 8
    assert results["old"]["stopped"] == 8
    assert results["old"]["win_rate"] == 0
    comparison = results["ensemble"]["paired_win_rate_vs"]["old"]
    assert comparison == {"mean": 1.0, "interval_95": None, "identities": 1}

"""Render a prior's model card and evaluation as the pull request's markdown body."""

from __future__ import annotations

from collections.abc import Mapping


def _number(value) -> str:
    return "n/a" if value is None else f"{value:+.5f}"


def render_card(card: Mapping, evaluation: Mapping, *, config_name: str) -> str:
    """Return the markdown model card of one prior version."""
    lines = [
        f"# Adaptive prior {card['version']}",
        "",
        f"This change adds `src/arena/config/{config_name}` and copies it to `adaptive_prior_current.json`.",
        f"The hal_loop pipeline fitted it on {card['training_games']} finished, stopped, or abandoned training games from {card['players']} players.",
        f"{card['finished_training_games']} of those games are finished; the verdict needs {card['minimum_training_games']}.",
        f"The fit verdict is **{card['verdict']}**.",
        f"MLflow run `{card.get('mlflow_run_id')}` holds the parameters and artifacts.",
        "",
        "## Prior",
        "",
        "| Role | Strength | Training moves |",
        "| --- | ---: | ---: |",
    ]
    for role, entry in card["selection"].items():
        lines.append(f"| {role} | {entry['strength']:g} | {entry['training_moves']:g} |")
    pooled = evaluation["pooled"]
    lines += [
        "",
        "## Offline evaluation",
        "",
        f"The baseline is the {evaluation['production_source']} production prior.",
        f"The {evaluation['split']} split holds {evaluation['players']} scored players and {pooled['moves']:g} human moves.",
        "",
        "| Scope | Candidate | Uniform | Production | Difference | 95% interval |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for scope, summary in [*evaluation["roles"].items(), ("pooled", pooled)]:
        means = summary["log_likelihood_per_move"]
        low, high = summary["interval"]
        lines.append(
            f"| {scope} | {_number(means['candidate'])} | {_number(means['uniform'])} | "
            f"{_number(means['production'])} | {_number(summary['difference'])} | "
            f"[{_number(low)}, {_number(high)}] |"
        )
    boot = evaluation["bootstrap"]
    lines += [
        "",
        f"The interval comes from {boot['resamples']} player-clustered bootstrap resamples with seed {boot['seed']}.",
        "Scores are mean log-likelihood per human move, in nats.",
        "",
    ]
    return "\n".join(lines)

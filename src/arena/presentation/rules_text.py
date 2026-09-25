"""Player-facing rules text shared by the terminal and the browser."""

from __future__ import annotations


def rules_body() -> tuple[str, ...]:
    """Canonical player-facing rules shared by the plain CLI and TUI."""
    return (
        "GOAL — Be the last player alive.",
        "",
        "EACH HALF-ROUND",
        "Dropper secretly chooses a handkerchief-drop second from 1..60.",
        "Checker independently chooses a check second from 1..60 (1 is immediate).",
        "Second 0 and passing are illegal; check succeeds when check >= drop.",
        "ST means Squandered Time.",
        "A success adds ST = check - drop + 1 to the Checker's vial.",
        "",
        "FAILED CHECK, DEATH, AND REVIVAL",
        "A failed check injects q = current vial ST + 60 seconds.",
        "q >= 300 or TTD + q > 300 is fatal; equality is eligible when q < 300.",
        "TTD means Total Time Dead; every revived dose is added to it.",
        "Revival chance falls as vial ST and prior TTD rise.",
        "A revival clears vial ST.",
        "Roles swap after every half-round that both players survive.",
    )

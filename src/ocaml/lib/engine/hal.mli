(** Canonical-Hal opponent: deterministic replay of the Surpassing The Leader
    manga transcript.

    The action table assumes Hal is the game's [first_dropper] (canonical
    rock-paper-scissors outcome from Round 1). Under that assumption:
    - On [First] halves Hal is the dropper and the returned value is a
      [drop_time].
    - On [Second] halves Hal is the checker and the returned value is a
      [check_time].

    For rounds beyond the canon (>9), a simple fallback policy applies: drop
    instantly when Hal is dropper, safe-strategy otherwise. *)

(** [lookup game] is [Some action] when [(round_num game, current_half game)] is
    in the canonical action table, and [None] otherwise (i.e. rounds > 9). *)
val lookup : Game.t -> int option

(** [choose_action game] is the legal second-in-turn at which Hal acts in the
    current half-round. Hal is always capped at second 60. *)
val choose_action : Game.t -> int

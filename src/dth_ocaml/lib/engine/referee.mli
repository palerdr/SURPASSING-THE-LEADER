(** Unified, identity-neutral revival resolution. *)

(** Stateful random generator used only for live revival rolls. *)
type rng = Random.State.t

(** [compute_revival_prob config ~st_in_vial ~ttd_accrued] is the revival
    probability for the two-variable physical state.

    [st_in_vial] is the ST accumulated before the impending 60-second
    failed-check dose. [ttd_accrued] is the player's prior total time dead.
    CPR count and player identity are deliberately absent. *)
val compute_revival_prob :
  Config.t -> st_in_vial:int -> ttd_accrued:int -> float

(** [attempt_revival config ~st_in_vial ~ttd_accrued rng] samples the unified
    probability once. *)
val attempt_revival :
  Config.t -> st_in_vial:int -> ttd_accrued:int -> rng -> bool

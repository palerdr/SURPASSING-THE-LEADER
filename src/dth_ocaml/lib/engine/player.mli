(** Player state and player-specific update helpers. *)

(** Per-player engine state. Although these fields change often during play,
    state is represented immutably and updated by returning copies. *)
type player_state = {
  id : Domain.player;
  cylinder_seconds : int;
  ttd_seconds : int;
  deaths : int;
  status : Domain.life_state;
  death_history : int list;
}

type cylinder_report =
  | No_overflow
  | Overflow
      (** Result of a cylinder update. [Overflow] means the new total reached or
          exceeded the fatal threshold. *)

(** [create_player config player] creates the initial zeroed state for [player].
    The config argument keeps construction uniform across engines. *)
val create_player : Config.t -> Domain.player -> player_state

(** [add_to_cylinder player_state amount] returns a copy of [player_state] with
    [amount] added to the cylinder total. *)
val add_to_cylinder : player_state -> int -> player_state

(** [add_to_ttd player_state amount] returns a copy of [player_state] with
    [amount] added to the total time dead. *)
val add_to_ttd : player_state -> int -> player_state

(** [get_cylinder player_state] is the player's current cylinder total in
    seconds. *)
val get_cylinder : player_state -> int

(** [get_ttd player_state] is the player's cumulative total time dead in
    seconds. *)
val get_ttd : player_state -> int

(** [atc_checked config player_state amount] adds [amount] to the player's
    cylinder and reports whether the new total crossed the fatal threshold. *)
val atc_checked :
  Config.t -> player_state -> int -> player_state * cylinder_report

(** [reset_cylinder player_state] returns a copy of [player_state] with the
    cylinder cleared back to zero. *)
val reset_cylinder : player_state -> player_state

(** [safe_checks_remaining config player_state] returns how many consecutive
    worst-case safe checks the player can still absorb before the cylinder would
    hit the fatal threshold. The safe condition is strict: reaching the maximum
    triggers injection. *)
val safe_checks_remaining : Config.t -> player_state -> int

(** [on_death player_state time_dead] records a death episode of duration
    [time_dead], increments the player's death count and total time dead,
    appends the duration to [death_history], and marks the player as dead. *)
val on_death : player_state -> int -> player_state

(** [on_revival player_state] clears the player's cylinder and marks the player
    as alive after a successful revival. *)
val on_revival : player_state -> player_state

(** [on_perm_death player_state] marks the player as dead after a failed
    revival. *)
val on_perm_death : player_state -> player_state

(** [remaining_cap config player_state] returns how many additional cylinder
    seconds the player can take before reaching the fatal threshold. A result of
    [0] means any further increase would trigger immediate injection. *)
val remaining_cap : Config.t -> player_state -> int

(** [can_absorb_injection config player_state amount] is [true] if adding
    [amount] cylinder seconds would still leave the player strictly below the
    fatal threshold. Reaching the threshold triggers immediate injection. *)
val can_absorb_injection : Config.t -> player_state -> int -> bool

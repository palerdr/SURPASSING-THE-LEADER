(** Drop The Handkerchief engine constants.

    All timing values are in seconds unless otherwise noted. The game clock is
    absolute: second 0 = 8:00:00 AM. *)

(** General timing and leap-second window. The leap second is inserted at
    exactly 8:59:60 AM; a half-round whose start falls in \[3540, 3600) spans
    it. *)
type time_config = {
  game_start_hour : int;
  opening_start_second : int;  (** canonical R1 start: 8:12:00 AM -> 720 *)
  seconds_per_minute : int;
  minutes_per_hour : int;
  ls_window_start : int;  (** 3540 -- start of the 8:59 minute *)
  ls_window_end : int;  (** 8:59:60 AM *)
  within_round_overhead : int;
      (** procedural time within a round (settling, injection, role swap),
          applied between halves *)
}

(** Turn timing. *)
type turn_config = {
  duration_normal : int;  (** seconds per half-round (normal) *)
  duration_leap : int;  (** seconds per half-round (during LS window) *)
  failed_check_penalty : int;  (** 1 minute NDD added on failed check *)
}

(** Cylinder / NDD thresholds. *)
type cylinder_config = {
  max : int;  (** 5 minutes -- at or above this, instant injection *)
  death_procedure_overhead : int;
      (** injection + waiting + CPR + recovery (~2 min) *)
}

(** Unified two-variable revival-model parameters.

    The only state variables are the ST already in the vial and the player's
    accrued TTD. The effective-referee factor is derived from TTD; there is no
    identity physicality or CPR-count input. *)
type survival_config = {
  baseline : float;  (** fresh-player, empty-vial probability: 0.80 *)
  ttd_half_life_seconds : float;
  ttd_curve_exponent : float;
  effective_referee_decay : float;
      (** decay per 60 seconds of accrued TTD *)
  effective_referee_floor : float;
}

(** Complete engine configuration grouped by subsystem. *)
type t = {
  time : time_config;
  turn : turn_config;
  cylinder : cylinder_config;
  survival : survival_config;
}

(** The repository's canonical default configuration value. *)
val default_config : t

(** [default ()] returns the canonical default configuration. *)
val default : unit -> t

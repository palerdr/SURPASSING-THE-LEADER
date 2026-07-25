(** Top-level game state and engine transitions.

    The game module is the orchestrator for match state. It owns both players,
    the game clock, round/half tracking, and the immutable history of resolved
    half-rounds. *)

type half_round_result =
  | Check_success
  | Check_fail_survived
  | Check_fail_died
  | Cylinder_overflow_survived
  | Cylinder_overflow_died
      (** Outcome of a resolved half-round.

          [Check_success] means the checker found the handkerchief and survived.
          [Check_fail_*] means the checker turned too early and was injected.
          [Cylinder_overflow_*] means the checker found the handkerchief but the
          added squandered time overflowed the cylinder and triggered injection.
      *)

(** Immutable summary of one resolved half-round.

    This record is appended to game history after each completed half-round and
    contains the information needed for replay, testing, and TUI display. Time
    fields are measured in absolute game seconds or per-turn seconds as named.
*)
type half_round_record = {
  round_num : int;
  half : Domain.half_index;
  dropper : Domain.player;
  checker : Domain.player;
  drop_time : int;
  check_time : int;
  turn_duration : int;
  result : half_round_result;
  st_gained : int;
  death_duration : int;
  survived : bool option;
  game_clock_at_start : int;
  survival_probability : float option;
}

(** Abstract game state.

    The representation is hidden so callers must use the public accessors and
    transition functions instead of depending on record fields. *)
type t

(** [create ?config ?seed ?first_dropper ()] constructs a fresh game.

    The initial state uses the supplied config or the default config, starts at
    the canonical opening clock, initializes an empty history, and chooses the
    requested first dropper when provided. If [seed] is supplied, the game's RNG
    is initialized deterministically. The starting time is always 8:12 AM.
    Default first checker is Baku by default *)
val create :
  ?config:Config.t -> ?seed:int -> ?first_dropper:Domain.player -> unit -> t

(** [config game] is the configuration used by [game]. *)
val config : t -> Config.t

(** [game_clock game] is the current absolute game clock in seconds since
    8:00:00. *)
val game_clock : t -> int

(** [round_num game] is the current zero-indexed round number. *)
val round_num : t -> int

(** [current_half game] is the half currently about to be played. *)
val current_half : t -> Domain.half_index

(** [history game] is the resolved half-round history in play order. *)
val history : t -> half_round_record list

(** [get_player_state game player] is the current state for [player]. *)
val get_player_state : t -> Domain.player -> Player.player_state

(** [winner game] is [Some p] only if the game is over and [p] has won. *)
val winner : t -> Domain.player option

(** [loser game] is [Some p] only if the game is over and [p] has lost. *)
val loser : t -> Domain.player option

(** [game_over game] is [true] exactly when the match has ended. *)
val game_over : t -> bool

(** [format_game_clock game] renders the current game clock for display.

    The rendered wall-clock time preserves the leap second when applicable. *)
val format_game_clock : t -> string

(** [get_turn_duration game] is the duration of the next half-round in seconds.

    It is the normal turn duration outside the leap window and the leap-second
    duration when the half-round begins inside that window. *)
val get_turn_duration : t -> int

(** [get_safe_checks] is the amount of safe checks this player has given their
    current cylinder NDD *)
val get_safe_checks : t -> Domain.player -> int

(** [get_roles_for_half game] is [(dropper, checker)] for the half currently
    about to be played.

    The assignment is determined by the game's first dropper and the current
    half index. *)
val get_roles_for_half : t -> Player.player_state * Player.player_state

(** [is_leap_second_turn game] is [true] when the next half-round begins in the
    leap-second window. *)
val is_leap_second_turn : t -> bool

(** [advance_clock game elapsed] returns a copy of [game] with the clock moved
    forward by [elapsed] seconds.

    Precondition: [elapsed >= 0]. *)
val advance_clock : t -> int -> t

(** [snap_clock_to_next_minute game] returns a copy of [game] with the clock
    advanced to the next legal round boundary.

    It always moves to a future boundary, even if the current clock is already
    on one. *)
val snap_clock_to_next_minute : t -> t

(** [validate_drop_time dropper drop_time turn_duration] raises if [drop_time]
    is not in the legal engine range for the half-round. Only Baku as Dropper
    may use second 61 during the leap window.

    Precondition: [turn_duration] is the duration of the half-round being
    resolved. *)
val validate_drop_time : Domain.player -> int -> int -> unit

(** [validate_check_time check_time turn_duration] raises if [check_time] is not
    in the legal engine range for the half-round.

    Checker is always capped at second 60, including in the leap window. *)
val validate_check_time : int -> int -> unit

(** [resolve_half_round game drop_time check_time survived_outcome] resolves
    exactly one half-round and returns the updated game plus its summary record.

    If a death occurs and [survived_outcome] is [None], the game uses its RNG to
    roll revival. If [survived_outcome] is [Some b], that survival outcome is
    forced instead, which is useful for deterministic simulation.

    The transition copies the state's RNG before rolling so callers can safely
    reuse the original immutable game state in search or repeated simulation. *)
val resolve_half_round : t -> int -> int -> bool option -> t * half_round_record

(** [play_half_round game drop_time check_time] resolves one live half-round
    using the game's RNG for any required revival roll.

    This is the normal gameplay entrypoint; deterministic callers should use
    [resolve_half_round] directly. *)
val play_half_round : t -> int -> int -> t * half_round_record

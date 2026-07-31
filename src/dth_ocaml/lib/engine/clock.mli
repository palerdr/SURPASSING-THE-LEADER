(** Clock helpers operate on the absolute game clock, where second 0 = 8:00:00.
    In this model, the inserted leap second is its own distinct absolute second:
    if [ls_window_end = 3600], then [3600 = 8:59:60] and [3601 = 9:00:00]. *)

(** [in_leap_window config abs_game_second] is [true] when the given absolute
    game second falls inside the configured leap-second window, including the
    leap second itself. *)
val in_leap_window : Config.t -> int -> bool

(** [turn_duration_for_start_time config start_time] returns the half-round
    duration for a turn beginning at [start_time]. Turns that begin in the leap
    window use the leap duration; all others use the normal duration. *)
val turn_duration_for_start_time : Config.t -> int -> int

(** [advance abs_clock_time elapsed_time] adds [elapsed_time] seconds to the
    current absolute game clock. *)
val advance : int -> int -> int

(** [snap_to_next_round_boundary config end_time] advances the absolute game
    clock to the next valid round-start wall-clock minute.

    It always advances to the next boundary, even if [end_time] is already on a
    valid boundary. Before the leap second, minute boundaries are multiples of
    60. During the leap-second minute, the next boundary is
    [config.time.ls_window_end + 1]. After the leap second, valid minute
    boundaries are shifted by one absolute second. *)
val snap_to_next_round_boundary : Config.t -> int -> int

(** A wall-clock rendering of the absolute game clock. [seconds] may be [60] for
    the inserted leap second. *)
type wall_time = {
  hours : int;
  minutes : int;
  seconds : int;
}

(** [to_wall_time config gs] converts an absolute game second into a displayed
    wall-clock time, preserving the leap second as [8:59:60]. *)
val to_wall_time : Config.t -> int -> wall_time

(** [format_wall_time wt] formats a wall-clock time as [HH:MM:SS]. *)
val format_wall_time : wall_time -> string

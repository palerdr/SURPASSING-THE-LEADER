open Dth_engine

(** Solver-facing legal timing actions under the frozen public rules. *)

type timing = int

(** [legal_drop_times game player] returns seconds 1..60 normally. During the
    leap window only Baku as Dropper receives second 61. *)
val legal_drop_times : Game.t -> Domain.player -> timing list

(** [legal_check_times game player] always returns seconds 1..60. *)
val legal_check_times : Game.t -> Domain.player -> timing list

(** [legal_times_for_role game player role] dispatches to the frozen
    role-specific action set. *)
val legal_times_for_role :
  Game.t -> Domain.player -> Domain.role -> timing list

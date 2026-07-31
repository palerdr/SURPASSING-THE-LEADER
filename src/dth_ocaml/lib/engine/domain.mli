(** Shared engine-level domain types used across rules, clock, and player state.
*)

type player =
  | Hal
  | Baku  (** The two participants in the match. *)

type role =
  | Dropper
  | Checker  (** A player's role within a half-round. *)

type turn_kind =
  | Normal
  | Leap_second
      (** Whether a half-round uses the normal or leap-second duration. *)

type half_result =
  | Success
  | Fail  (** Outcome of a checker action relative to the drop time. *)

type half_index =
  | First
  | Second
      (** Whether the current half-round is the first or second half of a round.
      *)

type life_state =
  | Alive
  | Dead  (** Whether a player is currently alive or dead. *)

type action =
  | Drop of int
  | Check of int
      (** A concrete player action measured in seconds from half-round start. *)

type winner = Winner of player  (** The winner of a finished game. *)

(* Shared types for game engine *)

type player =
  | Hal
  | Baku

type role =
  | Dropper
  | Checker

type turn_kind =
  | Normal
  | Leap_second

type half_result =
  | Success
  | Fail

type half_index =
  | First
  | Second

type life_state =
  | Alive
  | Dead

type action =
  | Drop of int
  | Check of int

type winner = Winner of player

open Dth_engine

(** Solver-facing transition and utility helpers.

    These functions expose the engine as a zero-sum stochastic game without
    changing the underlying rules. Terminal utility is pure Baku win/loss. *)

type chance_mode =
  | Enumerate
      (** Return all survival/death chance branches with exact probabilities. *)
  | Sample
      (** Use the engine RNG and return the sampled branch with probability 1.
      *)

type branch = {
  probability : float;
  game : Game.t;
  record : Game.half_round_record;
}

(** [baku_terminal_utility game] is [Some 1.0] when Baku has won, [Some -1.0]
    when Baku has lost, and [None] for nonterminal states. *)
val baku_terminal_utility : Game.t -> float option

(** [resolve ?chance_mode game ~drop_time ~check_time] resolves one half-round
    for solver use. [Enumerate] is deterministic and branches survival chance;
    [Sample] delegates to the engine RNG and is intended for MCTS rollouts. *)
val resolve :
  ?chance_mode:chance_mode ->
  Game.t ->
  drop_time:int ->
  check_time:int ->
  branch list

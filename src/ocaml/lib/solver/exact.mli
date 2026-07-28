type state = int * int * int * int

val initial : state

type outcome =
  | Terminal of float
  | Live of state

type branch = {
  probability : float;
  outcome : outcome;
}

type joint_action = int * int
type matrix = float array array

exception Invalid_Joint_Action
exception Invalid_Revival_Probability

type transition_distribution = branch array

val validate_join_action : joint_action -> unit
val squandered_time : joint_action -> int
val is_check_success : joint_action -> bool
val is_overflow : int -> int -> bool
val failed_check_branches : state -> joint_action -> transition_distribution
val successful_check_branches : state -> joint_action -> transition_distribution
val expand_joint_action : state -> joint_action -> transition_distribution
val joint_payoff : state -> joint_action -> float
val payoff_matrix : state -> matrix
val value : state -> float

(** [revival_probability st ttd] is the repository's frozen revival model.

    The failed-check dose is [st + 60]. It is eligible exactly when the dose is
    below 300 and the cumulative load [ttd + dose] is at most 300. *)
val revival_probability : int -> int -> float

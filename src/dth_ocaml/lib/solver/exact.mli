(** Role-relative DTH states, transitions, and the two exact solvers.

    Two solvers live here on purpose. The direct one -- {!val:payoff_matrix}
    over {!val:expand_joint_action} -- states the game as plainly as it can be
    written and is the readable reference. The packed one, {!val:solve_dth},
    is the same mathematics over the TTD-dead quotient and is what a full run
    would use. Both negate a live child's value, because every live edge swaps
    the seats. *)

(** Canonical role-relative order, matching [src/dth/solver.py]:
    [(checker_load, checker_ttd, dropper_load, dropper_ttd)]. *)
type state = int * int * int * int

type outcome =
  | Terminal of float
  | Live of state

type branch = {
  probability : float;
  outcome : outcome;
}

(** [(drop, check)], each a literal second in [1..60]. *)
type joint_action = int * int

type matrix = float array array
type transition_distribution = branch array

exception Invalid_Joint_Action
exception Invalid_Revival_Probability

val initial : state

(** The repository-wide frozen surface,
    [0.95 * (1 - st / 240) * 0.75 ** (ttd / 60)] when the dose is eligible. *)
val revival_probability : int -> int -> float

val squandered_time : joint_action -> int
val is_check_success : joint_action -> bool

(** [is_overflow squandered checker_load] -- the cylinder cap at 300. *)
val is_overflow : int -> int -> bool

(** Raises {!exception:Invalid_Joint_Action} outside [1..60]. *)
val validate_join_action : joint_action -> unit

(** The two halves of {!val:expand_joint_action}, exposed so the branch
    structure can be tested directly rather than through the matrix. *)
val successful_check_branches : state -> joint_action -> transition_distribution

val failed_check_branches : state -> joint_action -> transition_distribution

(** Chance branches of one joint action, Dropper-relative. *)
val expand_joint_action : state -> joint_action -> transition_distribution

(** The 60x60 payoff matrix of a state, rows Dropper, columns Checker. *)
val payoff_matrix : state -> matrix

(** The certified value of a state, Dropper-relative. *)
val value : state -> float

(** Solve every quotient class by descending potential. *)
val solve_dth : unit -> unit

val write_value_table : string -> unit

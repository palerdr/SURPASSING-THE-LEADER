type matrix = float array array

(** A solved simultaneous matrix, with the certificate that accepted it. *)
type solution = {
  value : float;
  dropper : float array;
  checker : float array;
  saddle_gap : float;
}

(** Raised when GLPK fails, or when the returned pair of policies does not
    certify a value. This module never falls back to an uncertified number. *)
exception Uncertified of string

(** The gate every accepted certificate must satisfy,
    [max_d (Mq)_d - min_c (M^T p)_c <= 1e-6]. This is the bound
    [src/dth/docs/GAME_AND_SOLVER.md] states for the peer DTH solver; the two
    projects stay independent but are held to the same number. *)
val saddle_gap_tolerance : float

(** [solve_certified m] solves both players' linear programs with GLPK and
    returns the certified equilibrium of the 60x60 payoff matrix [m], whose
    entries are Dropper payoffs in [[-1, 1]].

    @raise Invalid_argument
      if [m] is not 60x60 or carries a payoff outside [[-1, 1]].
    @raise Uncertified
      if either program fails, a recovered policy is not a probability
      distribution, or the saddle gap exceeds {!val:saddle_gap_tolerance}. *)
val solve_certified : matrix -> solution

(** [solve m] is [(solve_certified m).value]. *)
val solve : matrix -> float

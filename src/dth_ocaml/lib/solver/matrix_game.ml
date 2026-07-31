(* Matrix-game values from GLPK.

   docs/FOUNDATIONS.md makes linear programming the truth oracle for the
   canonical matrix value. This module hands both players' programs to GLPK and
   certifies the returned pair before it will report a value. It does not
   implement a simplex method, and it does not report an uncertified number. *)

type matrix = float array array

type solution = {
  value : float;
  dropper : float array;
  checker : float array;
  saddle_gap : float;
}

exception Uncertified of string

let action_count = 60

(* Payoffs are expectations over outcomes in [-1, 1]. Anything past the bound by
   more than rounding slack means the caller built the matrix wrong. *)
let payoff_slack = 1e-9

(* The bound src/dth/docs/GAME_AND_SOLVER.md publishes for the peer solver. *)
let saddle_gap_tolerance = 1e-6

(* GLPK's default primal feasibility tolerance is 1e-7, so an optimal basis
   legitimately reports probabilities slightly below zero. Anything within this
   bound is clipped and the policy renormalised; anything past it means the
   basis is not one we should be reading. Acceptance is still decided by the
   saddle gap, which is recomputed from the cleaned policies. *)
let policy_slack = 1e-6

let uncertified format =
  Printf.ksprintf (fun message -> raise (Uncertified message)) format

let validate_matrix (matrix : matrix) =
  if
    Array.length matrix <> action_count
    || Array.exists (fun row -> Array.length row <> action_count) matrix
  then invalid_arg "payoff matrix must be 60x60";
  Array.iter
    (Array.iter (fun payoff ->
         if
           (not (Float.is_finite payoff))
           || payoff < -1.0 -. payoff_slack
           || payoff > 1.0 +. payoff_slack
         then invalid_arg "matrix payoffs must be finite values in [-1, 1]"))
    matrix

let read_policy variables assignment =
  Array.map
    (fun variable ->
      match Lp.PMap.find_opt variable assignment with
      | Some probability -> probability
      | None -> 0.0)
    variables

(* The row player's program for [payoff]: maximise v subject to sum_i p_i
   payoff(i,j) >= v for every j, sum_i p_i = 1, p >= 0.

   Both players go through this one shape. lp-glpk mistranslates the mirrored
   minimise-and-[lt] program -- GLPK calls it infeasible on matrices where this
   program solves fine -- so the Checker is obtained from the row player's
   program on the negated transpose instead, which is also how
   src/stl/solver/exact.py recovers its second strategy. *)
let solve_maximin name (payoff : matrix) =
  let probabilities = Lp.range action_count name in
  let vector = Lp.concat probabilities in
  let value =
    Lp.var ~lb:Float.neg_infinity ~ub:Float.infinity (name ^ "_value")
  in
  let constraints = ref [ Lp.eq (Lp.concat probabilities) Lp.one ] in
  for reply = 0 to action_count - 1 do
    let column = Array.init action_count (fun own -> payoff.(own).(reply)) in
    constraints :=
      Lp.gt (Lp.dot (Lp.of_float_array column) vector) value :: !constraints
  done;
  let problem = Lp.Problem.make (Lp.maximize value) !constraints in
  (* These matrices are highly degenerate: a whole triangle of the payoff is one
     repeated number, and the rest is constant along diagonals. GLPK's default
     primal simplex can finish on a primal-infeasible basis there, so fall
     through its other methods before refusing. *)
  let rec attempt = function
    | [] -> None
    | method_ :: rest -> (
        match
          Lp_glpk.Simplex.solve ~term_output:false ~meth:method_ problem
        with
        | Ok (_objective, assignment) -> Some assignment
        | Error _ -> attempt rest)
  in
  (* [None] leaves GLPK on its default, which is the primal simplex. *)
  match
    attempt
      [ None; Some Lp_glpk.T.Smcp.Meth.DUALP; Some Lp_glpk.T.Smcp.Meth.DUAL ]
  with
  | None -> uncertified "GLPK solved no method for the %s program" name
  | Some assignment -> read_policy probabilities assignment

(* The Checker of [matrix] is the row player of its negated transpose:
   maximising min_d sum_c q_c (-M(d,c)) is minimising max_d (Mq)_d. *)
let negated_transpose (matrix : matrix) : matrix =
  Array.init action_count (fun check ->
      Array.init action_count (fun drop -> -.matrix.(drop).(check)))

let solve_dropper (matrix : matrix) = solve_maximin "p" matrix

let solve_checker (matrix : matrix) =
  solve_maximin "q" (negated_transpose matrix)

let certify_policy name raw =
  Array.iter
    (fun probability ->
      if not (Float.is_finite probability) then
        uncertified "%s policy carries a non-finite probability" name;
      if probability < -.policy_slack then
        uncertified "%s policy carries a materially negative probability %g"
          name probability)
    raw;
  let clipped = Array.map (fun p -> if p < 0.0 then 0.0 else p) raw in
  let mass = Array.fold_left ( +. ) 0.0 clipped in
  if (not (Float.is_finite mass)) || mass <= 0.0 then
    uncertified "%s policy has no probability mass" name;
  Array.map (fun p -> p /. mass) clipped

(* What the Dropper's mixture guarantees against every Checker reply. *)
let guaranteed (matrix : matrix) dropper =
  let best = ref infinity in
  for check = 0 to action_count - 1 do
    let total = ref 0.0 in
    for drop = 0 to action_count - 1 do
      total := !total +. (dropper.(drop) *. matrix.(drop).(check))
    done;
    if !total < !best then best := !total
  done;
  !best

(* What the Checker's mixture concedes against every Dropper reply. *)
let conceded (matrix : matrix) checker =
  let best = ref neg_infinity in
  for drop = 0 to action_count - 1 do
    let total = ref 0.0 in
    for check = 0 to action_count - 1 do
      total := !total +. (checker.(check) *. matrix.(drop).(check))
    done;
    if !total > !best then best := !total
  done;
  !best

let solve_certified (matrix : matrix) =
  validate_matrix matrix;
  let dropper = certify_policy "dropper" (solve_dropper matrix) in
  let checker = certify_policy "checker" (solve_checker matrix) in
  (* Recompute both bounds from the policies themselves. The LP objectives are
     not evidence for each other; these two numbers are. *)
  let lower = guaranteed matrix dropper in
  let upper = conceded matrix checker in
  let saddle_gap = Float.max 0.0 (upper -. lower) in
  if saddle_gap > saddle_gap_tolerance then
    uncertified "matrix saddle gap %g exceeds the %g gate" saddle_gap
      saddle_gap_tolerance;
  let value = (lower +. upper) /. 2.0 in
  if
    (not (Float.is_finite value))
    || value < -1.0 -. payoff_slack
    || value > 1.0 +. payoff_slack
  then uncertified "matrix value %g lies outside [-1, 1]" value;
  { value; dropper; checker; saddle_gap }

let solve (matrix : matrix) = (solve_certified matrix).value

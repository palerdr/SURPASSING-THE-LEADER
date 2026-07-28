(* sD, tD, sC, tC *)
type state = int * int * int * int

type outcome =
  | Terminal of float
  | Live of state

type branch = {
  probability : float;
  outcome : outcome;
}

(* drop, check *)
type joint_action = int * int
type matrix = float array array

exception Invalid_Joint_Action
exception Invalid_Revival_Probability

let validate_join_action (a : joint_action) : unit =
  let d, c = (fst a, snd a) in
  if d >= 1 && d <= 60 && c >= 1 && c <= 60 then ()
  else raise Invalid_Joint_Action

type transition_distribution = branch array

(* root value *)
let initial = (0, 0, 0, 0)

(* helpers *)
let fst4 (a, _, _, _) = a
let snd4 (_, b, _, _) = b
let thd4 (_, _, c, _) = c
let fth4 (_, _, _, d) = d

(* revival model *)
let revival_probability st ttd =
  let q = st + 60 in
  if q >= 300 || ttd + q > 300 then 0.0
  else
    let q_factor = 1.0 -. (float_of_int st /. 240.0) in
    let ttd_factor = 0.75 ** (float_of_int ttd /. 60.0) in
    0.95 *. q_factor *. ttd_factor

let squandered_time (a : joint_action) =
  let d, c = (fst a, snd a) in
  c - d + 1

let is_check_success (a : joint_action) =
  let d, c = (fst a, snd a) in
  c >= d

let is_overflow (st : int) (sc : int) = st + sc >= 300

let failed_check_branches (x : state) (a : joint_action) :
    transition_distribution =
  let _ = assert (not (is_check_success a)) in
  let _ = validate_join_action a in
  let sd, td, sc, tc = (fst4 x, snd4 x, thd4 x, fth4 x) in
  let p = revival_probability sc tc in
  if p = 0.0 then [| { probability = 1.0; outcome = Terminal 1.0 } |]
  else
    let q = sc + 60 in
    let x_survive = (0, tc + q, sd, td) in
    if p = 1.0 then [| { probability = 1.0; outcome = Live x_survive } |]
    else begin
      assert (0.0 < p && p < 1.0);
      [|
        { probability = p; outcome = Live x_survive };
        { probability = 1.0 -. p; outcome = Terminal 1.0 };
      |]
    end

let successful_check_branches (x : state) (a : joint_action) :
    transition_distribution =
  let _ = assert (is_check_success a) in
  let _ = validate_join_action a in
  let sd, td, sc, tc = (fst4 x, snd4 x, thd4 x, fth4 x) in
  let st = squandered_time a in
  if is_overflow st sc then [| { probability = 1.0; outcome = Terminal 1.0 } |]
  else
    let x_continue = (sc + st, tc, sd, td) in
    [| { probability = 1.0; outcome = Live x_continue } |]

let expand_joint_action (x : state) (a : joint_action) : transition_distribution
    =
  if is_check_success a then successful_check_branches x a
  else failed_check_branches x a

let value_cache = Hashtbl.create 4096

let rec value (x : state) : float =
  match Hashtbl.find_opt value_cache x with
  | Some result -> result
  | None ->
      let result = Matrix_lp.solve (payoff_matrix x) in
      Hashtbl.add value_cache x result;
      result

and joint_payoff (x : state) (a : joint_action) : float =
  Array.fold_left
    (fun (expectation : float) (branch : branch) ->
      match branch with
      | { probability = p; outcome = Terminal v } -> expectation +. (p *. v)
      | { probability = p; outcome = Live x_prime } ->
          expectation +. (p *. value x_prime))
    0.0 (expand_joint_action x a)

and payoff_matrix (x : state) : matrix =
  Array.init 60 (fun d ->
      Array.init 60 (fun c -> joint_payoff x (d + 1, c + 1)))

(* Canonical role-relative order, matching the Python authority in
   src/dth/solver.py and the class encoding further down this file: Checker
   load, Checker TTD, Dropper load, Dropper TTD. *)
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

(* Canonical Solver Machinery *)
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
  let sc, tc, sd, td = (fst4 x, snd4 x, thd4 x, fth4 x) in
  let p = revival_probability sc tc in
  if p = 0.0 then [| { probability = 1.0; outcome = Terminal 1.0 } |]
  else
    let q = sc + 60 in
    (* Roles swap: the injected Checker becomes the next Dropper at load 0. *)
    let x_survive = (sd, td, 0, tc + q) in
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
  let sc, tc, sd, td = (fst4 x, snd4 x, thd4 x, fth4 x) in
  let st = squandered_time a in
  if is_overflow st sc then [| { probability = 1.0; outcome = Terminal 1.0 } |]
  else
    (* Roles swap: the Checker carries its new load into the Dropper seat. *)
    let x_continue = (sd, td, sc + st, tc) in
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
      let result = Matrix_game.solve (payoff_matrix x) in
      Hashtbl.add value_cache x result;
      result

and joint_payoff (x : state) (a : joint_action) : float =
  (* Every payoff here is stated from the Dropper of [x]. A live child has
     swapped roles, so [value x_prime] is stated from the opponent's seat and
     must be negated on the way back -- the same sign the packed path applies in
     [class_continuation_value]. Terminal 1.0 is already this Dropper's win and
     is not negated. *)
  Array.fold_left
    (fun (expectation : float) (branch : branch) ->
      match branch with
      | { probability = p; outcome = Terminal v } -> expectation +. (p *. v)
      | { probability = p; outcome = Live x_prime } ->
          expectation -. (p *. value x_prime))
    0.0 (expand_joint_action x a)

and payoff_matrix (x : state) : matrix =
  Array.init 60 (fun d ->
      Array.init 60 (fun c -> joint_payoff x (d + 1, c + 1)))

(* Packed Tablebase Machinery *)

type profile =
  | Alive of int * int
  | Dead of int

type child =
  | LiveProfile of profile
  | TerminalWin

type profile_row = {
  rep : profile;
  potential : int;
  p : float;
  success_children : child array;
  fail_child : child;
}

let lag (drop : int) (check : int) = squandered_time (drop, check)

let quotient (s : int) (t : int) : profile =
  if s + 60 >= 300 || s + t + 60 > 300 then Dead s else Alive (s, t)

let create_quotient_profiles () : profile array =
  let qs = Array.make 17011 (Dead 0) in
  let k, t = (ref 0, 0) in
  for s = 0 to 299 do
    match quotient s t with
    | Alive (st, ttd) ->
        qs.(!k) <- Alive (st, ttd);
        k := !k + 1
    | Dead st -> ()
  done;
  for t = 60 to 300 do
    for s = 0 to 299 do
      match quotient s t with
      | Alive (st, ttd) ->
          qs.(!k) <- Alive (st, ttd);
          k := !k + 1
      | Dead st -> ()
    done
  done;
  for s = 0 to 299 do
    qs.(16711 + s) <- Dead s
  done;
  (* now first 16711 slots are Alive and the rest are Dead *)
  qs

let qs = create_quotient_profiles ()

let profile_ids =
  let ids = Hashtbl.create 17011 in
  Array.iteri (fun pi profile -> Hashtbl.add ids profile pi) qs;
  ids

let profile_id (u : profile) : int =
  match Hashtbl.find_opt profile_ids u with
  | Some pi -> pi
  | None -> invalid_arg "profile is not in quotient table"

let rep pi = qs.(pi)

let potential (pi : int) : int =
  match rep pi with
  | Alive (s, t) -> s + t
  | Dead s -> s + 301

let rev (pi : int) : float =
  match rep pi with
  | Alive (s, t) -> revival_probability s t
  | Dead s -> 0.0

let alive_child (a : profile) (l : int) : child =
  match a with
  | Alive (s, t) ->
      if s + l >= 300 then TerminalWin else LiveProfile (quotient (s + l) t)
  | _ -> invalid_arg "cannot produce a successful child from a death"

let death_child (d : profile) (l : int) : child =
  match d with
  | Dead s -> if s + l >= 300 then TerminalWin else LiveProfile (Dead (s + l))
  | _ -> invalid_arg "cannot produce a death child from alive"

let success_children (u : profile) =
  Array.init 60 (fun l ->
      match u with
      | Alive (s, t) -> alive_child u (l + 1)
      | Dead s -> death_child u (l + 1))

let fail_child (u : profile) =
  match u with
  | Dead s -> TerminalWin
  | Alive (s, t) ->
      if revival_probability s t > 0.0 then
        LiveProfile (quotient 0 (t + s + 60))
      else TerminalWin

let profile_row (pi : int) =
  {
    rep = rep pi;
    potential = potential pi;
    p = rev pi;
    success_children = success_children (rep pi);
    fail_child = fail_child (rep pi);
  }

let build_profile_table () = Array.init 17011 (fun pi -> profile_row pi)
let profile_table = build_profile_table ()
let class_id_from_profile (pic : int) (pid : int) = (pic * 17011) + pid
let profile_ids_from_class k = (k / 17011, k mod 17011)

let phi (k : int) =
  let pic, pid = profile_ids_from_class k in
  potential pic + potential pid

let value_table = Array.make 289374121 nan

let class_expand_action (k : int) (action : joint_action) : int option =
  validate_join_action action;
  let class_child (k : int) (child : child) : int option =
    match child with
    | TerminalWin -> None
    | LiveProfile profile ->
        let _, dropper_id = profile_ids_from_class k in
        Some ((dropper_id * 17011) + profile_id profile)
  in
  let drop, check = action in
  let checker_id, _ = profile_ids_from_class k in
  let row = profile_table.(checker_id) in
  let child =
    if check >= drop then
      let lag = check - drop + 1 in
      row.success_children.(lag - 1)
    else row.fail_child
  in
  class_child k child

let class_continuation_value (k_child : int option) =
  match k_child with
  | None -> 1.0
  | Some k_child -> -.value_table.(k_child)

let class_action_payoff (k : int) (action : joint_action) : float =
  let drop, check = action in
  let checker_id, dropper_id = profile_ids_from_class k in
  let continuation_value =
    class_continuation_value (class_expand_action k action)
  in
  let row = profile_table.(checker_id) in
  let p = row.p in
  if check >= drop then continuation_value
  else ((1.0 -. p) *. 1.0) +. (p *. continuation_value)

let class_payoff_matrix (k : int) : matrix =
  Array.init 60 (fun d ->
      Array.init 60 (fun c -> class_action_payoff k (d + 1, c + 1)))

let class_solve_matrix (k : int) : unit =
  value_table.(k) <- Matrix_game.solve (class_payoff_matrix k)

let solve_dth () =
  let solved_classes = ref 0 in
  Printf.eprintf "starting packed DTH solve\n%!";
  for phase = 1200 downto 0 do
    let phase_classes = ref 0 in
    for pic = 0 to 17010 do
      for pid = 0 to 17010 do
        if potential pic + potential pid = phase then (
          let k = (17011 * pic) + pid in
          class_solve_matrix k;
          incr solved_classes;
          incr phase_classes;
          if !solved_classes mod 10000 = 0 then
            Printf.eprintf "solved %d classes (phase %d)\n%!" !solved_classes
              phase)
      done
    done;
    if !phase_classes > 0 then
      Printf.eprintf "completed phase %d: %d classes (total %d)\n%!" phase
        !phase_classes !solved_classes
  done

let write_value_table (path : string) : unit =
  let channel = open_out_bin path in
  Fun.protect
    ~finally:(fun () -> close_out_noerr channel)
    (fun () ->
      Marshal.to_channel channel value_table [];
      flush channel)

open Dth_engine

exception Invalid_distribution of string

type action_prob = {
  action : Solver_actions.timing;
  prob : float;
}

type distribution = action_prob list
type t = Game.t -> Domain.player -> Domain.role -> distribution

let tolerance = 1e-9
let invalid message = raise (Invalid_distribution message)

let legal_actions game player role =
  Solver_actions.legal_times_for_role game player role

let finite x = Float.is_finite x

let sum_prob distribution =
  List.fold_left (fun acc ap -> acc +. ap.prob) 0.0 distribution

let legal_set actions action = List.exists (( = ) action) actions

let rec has_duplicate_action seen = function
  | [] -> false
  | { action; _ } :: rest ->
      List.mem action seen || has_duplicate_action (action :: seen) rest

let validate game player role distribution =
  let actions = legal_actions game player role in
  if distribution = [] then invalid "distribution is empty";
  if has_duplicate_action [] distribution then
    invalid "distribution contains duplicate actions";
  List.iter
    (fun { action; prob } ->
      if not (legal_set actions action) then
        invalid (Printf.sprintf "illegal action %d" action);
      if not (finite prob) then
        invalid (Printf.sprintf "non-finite probability for action %d" action);
      if prob < 0.0 then
        invalid (Printf.sprintf "negative probability for action %d" action))
    distribution;
  let total = sum_prob distribution in
  if not (finite total) then invalid "non-finite probability sum";
  if Float.abs (total -. 1.0) > tolerance then
    invalid (Printf.sprintf "probability sum is %.12g, expected 1" total)

let normalize distribution =
  if distribution = [] then invalid "cannot normalize an empty distribution";
  let total =
    List.fold_left
      (fun acc { action; prob } ->
        if not (finite prob) then
          invalid
            (Printf.sprintf "cannot normalize non-finite probability for %d"
               action);
        if prob < 0.0 then
          invalid
            (Printf.sprintf "cannot normalize negative probability for %d"
               action);
        acc +. max 0.0 prob)
      0.0 distribution
  in
  if total <= tolerance then invalid "cannot normalize zero-mass distribution";
  List.map
    (fun { action; prob } -> { action; prob = max 0.0 prob /. total })
    distribution

let normalize_for game player role distribution =
  let normalized = normalize distribution in
  validate game player role normalized;
  normalized

let deterministic action = [ { action; prob = 1.0 } ]

let deterministic_legal game player role action =
  normalize_for game player role (deterministic action)

let uniform game player role =
  let actions = legal_actions game player role in
  match actions with
  | [] -> invalid "no legal actions"
  | _ ->
      let prob = 1.0 /. float_of_int (List.length actions) in
      List.map (fun action -> { action; prob }) actions

let sample rng distribution =
  let normalized = normalize distribution in
  let threshold = Random.State.float rng 1.0 in
  let rec loop cumulative fallback = function
    | [] -> fallback
    | { action; prob } :: rest ->
        let cumulative = cumulative +. prob in
        if threshold < cumulative then action else loop cumulative action rest
  in
  match normalized with
  | [] -> invalid "cannot sample an empty distribution"
  | { action; _ } :: _ -> loop 0.0 action normalized

let first_legal_action game player role =
  match legal_actions game player role with
  | action :: _ -> action
  | [] -> invalid "no legal actions"

let last_legal_action game player role =
  match List.rev (legal_actions game player role) with
  | action :: _ -> action
  | [] -> invalid "no legal actions"

let instant game player role =
  deterministic_legal game player role (first_legal_action game player role)

let safe game player role =
  let action =
    match role with
    | Domain.Dropper -> first_legal_action game player role
    | Domain.Checker -> last_legal_action game player role
  in
  deterministic_legal game player role action

let uniform_policy game player role = uniform game player role

let canonical_hal game player role =
  if player = Domain.Hal then
    let chosen = Hal.choose_action game in
    let action =
      if legal_set (legal_actions game player role) chosen then chosen
      else last_legal_action game player role
    in
    deterministic_legal game player role action
  else uniform game player role

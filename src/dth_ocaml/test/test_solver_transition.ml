open Dth_engine
open Dth_solver

let float_eps = Alcotest.float 0.000001

let prime_baku_cylinder game =
  let rec loop g n =
    if n = 0 then g
    else
      let g, _ = Game.resolve_half_round g 1 60 (Some true) in
      let g, _ = Game.resolve_half_round g 60 60 (Some true) in
      loop g (n - 1)
  in
  loop game 4

let branch_probability_sum branches =
  List.fold_left
    (fun acc b -> acc +. b.Solver_transition.probability)
    0.0 branches

let test_success_has_single_branch () =
  let game = Game.create () in
  let branches = Solver_transition.resolve game ~drop_time:1 ~check_time:10 in
  Alcotest.(check int) "one deterministic branch" 1 (List.length branches);
  Alcotest.(check float_eps)
    "probability one" 1.0
    (branch_probability_sum branches)

let test_success_branch_records_resolution () =
  let game = Game.create () in
  let branch =
    Solver_transition.resolve game ~drop_time:17 ~check_time:42 |> List.hd
  in
  Alcotest.(check int) "drop time recorded" 17 branch.record.drop_time;
  Alcotest.(check int) "check time recorded" 42 branch.record.check_time;
  Alcotest.(check int) "inclusive squandered time recorded" 26
    branch.record.st_gained;
  Alcotest.(check (option bool)) "no survival roll" None branch.record.survived;
  Alcotest.(check (option float_eps))
    "nonterminal utility" None
    (Solver_transition.baku_terminal_utility branch.game)

let test_failed_check_branches_survival_chance () =
  let game = Game.create () in
  let branches = Solver_transition.resolve game ~drop_time:60 ~check_time:1 in
  Alcotest.(check int) "survival and death branches" 2 (List.length branches);
  Alcotest.(check float_eps)
    "probabilities sum to one" 1.0
    (branch_probability_sum branches);
  Alcotest.(check bool)
    "one branch survives" true
    (List.exists
       (fun b -> b.Solver_transition.record.survived = Some true)
       branches);
  Alcotest.(check bool)
    "one branch dies" true
    (List.exists
       (fun b -> b.Solver_transition.record.survived = Some false)
       branches)

let test_failed_check_branch_probabilities_match_record () =
  let game = Game.create () in
  let branches = Solver_transition.resolve game ~drop_time:60 ~check_time:1 in
  let survival_probability =
    match
      List.find_map
        (fun b -> b.Solver_transition.record.survival_probability)
        branches
    with
    | Some p -> p
    | None -> Alcotest.fail "expected survival probability"
  in
  let survived_branch =
    List.find
      (fun b -> b.Solver_transition.record.survived = Some true)
      branches
  in
  let died_branch =
    List.find
      (fun b -> b.Solver_transition.record.survived = Some false)
      branches
  in
  Alcotest.(check float_eps)
    "survived branch probability" survival_probability
    survived_branch.Solver_transition.probability;
  Alcotest.(check float_eps)
    "died branch probability"
    (1.0 -. survival_probability)
    died_branch.Solver_transition.probability

let test_overflow_is_guaranteed_death_branch () =
  let game = Game.create () |> prime_baku_cylinder in
  let branches = Solver_transition.resolve game ~drop_time:1 ~check_time:60 in
  Alcotest.(check int) "overflow has only death branch" 1 (List.length branches);
  let branch = List.hd branches in
  Alcotest.(check float_eps)
    "probability one" 1.0 branch.Solver_transition.probability;
  Alcotest.(check (option bool))
    "survival is false" (Some false) branch.Solver_transition.record.survived

let test_baku_terminal_utility () =
  let game = Game.create () in
  let baku_dies, _ = Game.resolve_half_round game 60 1 (Some false) in
  Alcotest.(check (option float_eps))
    "Baku loss is -1" (Some (-1.0))
    (Solver_transition.baku_terminal_utility baku_dies);
  let game = Game.create () in
  let game, _ = Game.resolve_half_round game 1 60 (Some true) in
  let hal_dies, _ = Game.resolve_half_round game 60 1 (Some false) in
  Alcotest.(check (option float_eps))
    "Baku win is +1" (Some 1.0)
    (Solver_transition.baku_terminal_utility hal_dies)

let test_sampling_reuses_original_state_deterministically () =
  let game = Game.create ~seed:99 () in
  let branch1 =
    Solver_transition.resolve ~chance_mode:Solver_transition.Sample game
      ~drop_time:60 ~check_time:1
    |> List.hd
  in
  let branch2 =
    Solver_transition.resolve ~chance_mode:Solver_transition.Sample game
      ~drop_time:60 ~check_time:1
    |> List.hd
  in
  Alcotest.(check (option bool))
    "same sampled outcome from the same original state" branch1.record.survived
    branch2.record.survived

let test_sample_mode_returns_single_probability_one_branch () =
  let game = Game.create ~seed:3110 () in
  let branches =
    Solver_transition.resolve ~chance_mode:Solver_transition.Sample game
      ~drop_time:60 ~check_time:1
  in
  Alcotest.(check int) "one sampled branch" 1 (List.length branches);
  Alcotest.(check float_eps)
    "sampled branch probability one" 1.0
    (branch_probability_sum branches)

let test_resolve_rejects_invalid_engine_actions () =
  let game = Game.create () in
  Alcotest.check_raises "drop time below lower bound"
    (Failure "Invalid drop-time") (fun () ->
      ignore (Solver_transition.resolve game ~drop_time:0 ~check_time:1));
  Alcotest.check_raises "check time above upper bound"
    (Failure "Invalid checking-time") (fun () ->
      ignore (Solver_transition.resolve game ~drop_time:1 ~check_time:61))

let tests =
  [
    Alcotest.test_case "successful check has one branch" `Quick
      test_success_has_single_branch;
    Alcotest.test_case "successful check record is faithful" `Quick
      test_success_branch_records_resolution;
    Alcotest.test_case "failed check branches on revival" `Quick
      test_failed_check_branches_survival_chance;
    Alcotest.test_case "branch probabilities match survival record" `Quick
      test_failed_check_branch_probabilities_match_record;
    Alcotest.test_case "overflow is guaranteed death" `Quick
      test_overflow_is_guaranteed_death_branch;
    Alcotest.test_case "Baku terminal utility is pure win/loss" `Quick
      test_baku_terminal_utility;
    Alcotest.test_case "sampling does not mutate original state" `Quick
      test_sampling_reuses_original_state_deterministically;
    Alcotest.test_case "sample mode returns one probability-one branch" `Quick
      test_sample_mode_returns_single_probability_one_branch;
    Alcotest.test_case "invalid actions are rejected by engine validation"
      `Quick test_resolve_rejects_invalid_engine_actions;
  ]

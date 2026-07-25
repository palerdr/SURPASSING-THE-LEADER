open Dth_engine
open Dth_solver

let float_eps = Alcotest.float 0.000001

let move_to_leap_window game =
  let config = Game.config game in
  Game.advance_clock game (config.time.ls_window_start - Game.game_clock game)

let sum_probs distribution =
  List.fold_left (fun acc ap -> acc +. ap.Solver_policy.prob) 0.0 distribution

let action_in_distribution action distribution =
  List.exists (fun ap -> ap.Solver_policy.action = action) distribution

let only_action distribution =
  match distribution with
  | [ { Solver_policy.action; prob } ] ->
      Alcotest.(check float_eps) "probability one" 1.0 prob;
      action
  | _ -> Alcotest.fail "expected deterministic distribution"

let expect_invalid f =
  match f () with
  | exception Solver_policy.Invalid_distribution _ -> ()
  | exception exn ->
      Alcotest.failf "expected Invalid_distribution, got %s"
        (Printexc.to_string exn)
  | _ -> Alcotest.fail "expected Invalid_distribution"

let test_uniform_normal_turn () =
  let game = Game.create () in
  let distribution = Solver_policy.uniform game Domain.Baku Domain.Checker in
  Alcotest.(check int)
    "one action per normal second" 60 (List.length distribution);
  Alcotest.(check float_eps) "probability sum" 1.0 (sum_probs distribution);
  Solver_policy.validate game Domain.Baku Domain.Checker distribution

let test_baku_dropper_uniform_includes_leap_second () =
  let game = Game.create () |> move_to_leap_window in
  let distribution =
    Solver_policy.uniform game Domain.Baku Domain.Dropper
  in
  Alcotest.(check int)
    "Baku Dropper action count" 61 (List.length distribution);
  Alcotest.(check bool)
    "second 61 included" true
    (action_in_distribution 61 distribution);
  Solver_policy.validate game Domain.Baku Domain.Dropper distribution

let test_hal_checker_uniform_excludes_leap_second () =
  let game = Game.create () |> move_to_leap_window in
  let distribution = Solver_policy.uniform game Domain.Hal Domain.Checker in
  Alcotest.(check int)
    "canonical Hal checker action count" 60 (List.length distribution);
  Alcotest.(check bool)
    "no second 61" false
    (action_in_distribution 61 distribution);
  Solver_policy.validate game Domain.Hal Domain.Checker distribution

let test_baku_checker_uniform_excludes_leap_second () =
  let game = Game.create () |> move_to_leap_window in
  let distribution = Solver_policy.uniform game Domain.Baku Domain.Checker in
  Alcotest.(check int) "Baku checker action count" 60 (List.length distribution);
  Alcotest.(check bool)
    "second 61 excluded" false
    (action_in_distribution 61 distribution);
  Solver_policy.validate game Domain.Baku Domain.Checker distribution

let test_invalid_action_rejected () =
  let game = Game.create () |> move_to_leap_window in
  expect_invalid (fun () ->
      Solver_policy.validate game Domain.Hal Domain.Checker
        (Solver_policy.deterministic 61))

let test_invalid_probability_sum_rejected () =
  let game = Game.create () in
  expect_invalid (fun () ->
      Solver_policy.validate game Domain.Baku Domain.Checker
        [ { Solver_policy.action = 1; prob = 0.5 } ])

let test_negative_probability_rejected () =
  let game = Game.create () in
  expect_invalid (fun () ->
      Solver_policy.validate game Domain.Baku Domain.Checker
        [
          { Solver_policy.action = 1; prob = 1.1 };
          { Solver_policy.action = 2; prob = -0.1 };
        ])

let test_duplicate_action_rejected () =
  let game = Game.create () in
  expect_invalid (fun () ->
      Solver_policy.validate game Domain.Baku Domain.Checker
        [
          { Solver_policy.action = 1; prob = 0.5 };
          { Solver_policy.action = 1; prob = 0.5 };
        ])

let test_normalize_for_rejects_illegal_action () =
  let game = Game.create () in
  expect_invalid (fun () ->
      ignore
        (Solver_policy.normalize_for game Domain.Baku Domain.Checker
           [ { Solver_policy.action = 0; prob = 2.0 } ]))

let test_normalize_rescales_weights () =
  let distribution =
    Solver_policy.normalize
      [
        { Solver_policy.action = 10; prob = 2.0 };
        { Solver_policy.action = 20; prob = 6.0 };
      ]
  in
  Alcotest.(check float_eps)
    "first normalized weight" 0.25 (List.hd distribution).Solver_policy.prob;
  Alcotest.(check float_eps) "probability sum" 1.0 (sum_probs distribution)

let test_sample_deterministic () =
  let rng = Random.State.make [| 3110 |] in
  Alcotest.(check int)
    "sampled action" 37
    (Solver_policy.sample rng (Solver_policy.deterministic 37))

let test_safe_checker_uses_last_legal_action () =
  let game = Game.create () |> move_to_leap_window in
  Alcotest.(check int)
    "Hal last legal action" 60
    (only_action (Solver_policy.safe game Domain.Hal Domain.Checker));
  Alcotest.(check int)
    "Baku last legal action" 60
    (only_action (Solver_policy.safe game Domain.Baku Domain.Checker))

let test_instant_uses_first_legal_action () =
  let game = Game.create () |> move_to_leap_window in
  Alcotest.(check int)
    "first legal action" 1
    (only_action (Solver_policy.instant game Domain.Baku Domain.Checker))

let test_deterministic_legal_rejects_illegal_action () =
  let game = Game.create () in
  expect_invalid (fun () ->
      ignore
        (Solver_policy.deterministic_legal game Domain.Baku Domain.Checker 0))

let test_uniform_policy_alias () =
  let game = Game.create () in
  Alcotest.(check int)
    "uniform policy action count" 60
    (List.length (Solver_policy.uniform_policy game Domain.Baku Domain.Checker))

let test_canonical_hal_policy () =
  let game = Game.create ~first_dropper:Domain.Hal () in
  Alcotest.(check int)
    "scripted Hal first action" 60
    (only_action (Solver_policy.canonical_hal game Domain.Hal Domain.Dropper));
  Alcotest.(check int)
    "Baku fallback is uniform" 60
    (List.length (Solver_policy.canonical_hal game Domain.Baku Domain.Checker))

let tests =
  [
    Alcotest.test_case "uniform policy covers normal turn" `Quick
      test_uniform_normal_turn;
    Alcotest.test_case "Baku Dropper uniform includes leap second" `Quick
      test_baku_dropper_uniform_includes_leap_second;
    Alcotest.test_case "Hal checker uniform excludes leap second" `Quick
      test_hal_checker_uniform_excludes_leap_second;
    Alcotest.test_case "Baku checker uniform excludes leap second" `Quick
      test_baku_checker_uniform_excludes_leap_second;
    Alcotest.test_case "invalid action is rejected" `Quick
      test_invalid_action_rejected;
    Alcotest.test_case "invalid probability sum is rejected" `Quick
      test_invalid_probability_sum_rejected;
    Alcotest.test_case "negative probability is rejected" `Quick
      test_negative_probability_rejected;
    Alcotest.test_case "duplicate action is rejected" `Quick
      test_duplicate_action_rejected;
    Alcotest.test_case "normalize_for rejects illegal actions" `Quick
      test_normalize_for_rejects_illegal_action;
    Alcotest.test_case "normalize rescales positive weights" `Quick
      test_normalize_rescales_weights;
    Alcotest.test_case "deterministic distribution samples deterministically"
      `Quick test_sample_deterministic;
    Alcotest.test_case "safe checker uses last legal action" `Quick
      test_safe_checker_uses_last_legal_action;
    Alcotest.test_case "instant uses first legal action" `Quick
      test_instant_uses_first_legal_action;
    Alcotest.test_case "deterministic_legal rejects illegal action" `Quick
      test_deterministic_legal_rejects_illegal_action;
    Alcotest.test_case "uniform_policy delegates to uniform" `Quick
      test_uniform_policy_alias;
    Alcotest.test_case "canonical_hal scripts Hal and defaults Baku to uniform"
      `Quick test_canonical_hal_policy;
  ]

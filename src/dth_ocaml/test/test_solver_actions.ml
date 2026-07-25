open Dth_engine
open Dth_solver

let move_to_leap_window game =
  let config = Game.config game in
  Game.advance_clock game (config.time.ls_window_start - Game.game_clock game)

let test_normal_turn_actions_are_sixty_seconds () =
  let game = Game.create () in
  Alcotest.(check int)
    "normal drop action count" 60
    (List.length (Solver_actions.legal_drop_times game Domain.Baku));
  Alcotest.(check int)
    "normal check action count" 60
    (List.length (Solver_actions.legal_check_times game Domain.Baku))

let test_only_baku_dropper_gets_leap_second () =
  let game = Game.create () |> move_to_leap_window in
  Alcotest.(check bool)
    "Baku Dropper may choose 61" true
    (List.mem 61 (Solver_actions.legal_drop_times game Domain.Baku));
  Alcotest.(check bool)
    "Hal Dropper remains capped at 60" false
    (List.mem 61 (Solver_actions.legal_drop_times game Domain.Hal))

let test_checker_is_always_capped_at_sixty () =
  let game = Game.create () |> move_to_leap_window in
  Alcotest.(check int)
    "Baku Checker action count" 60
    (List.length (Solver_actions.legal_check_times game Domain.Baku));
  Alcotest.(check int)
    "Hal Checker action count" 60
    (List.length (Solver_actions.legal_check_times game Domain.Hal));
  Alcotest.(check bool)
    "no Checker may choose 61" false
    (List.mem 61 (Solver_actions.legal_check_times game Domain.Baku)
    || List.mem 61 (Solver_actions.legal_check_times game Domain.Hal))

let test_role_dispatch_enforces_frozen_rule () =
  let game = Game.create () |> move_to_leap_window in
  Alcotest.(check bool)
    "Baku Dropper dispatch includes 61" true
    (List.mem 61
       (Solver_actions.legal_times_for_role game Domain.Baku Domain.Dropper));
  Alcotest.(check bool)
    "Baku Checker dispatch excludes 61" false
    (List.mem 61
       (Solver_actions.legal_times_for_role game Domain.Baku Domain.Checker))

let tests =
  [
    Alcotest.test_case "normal turns expose 60 actions" `Quick
      test_normal_turn_actions_are_sixty_seconds;
    Alcotest.test_case "only Baku Dropper receives second 61" `Quick
      test_only_baku_dropper_gets_leap_second;
    Alcotest.test_case "Checker is always capped at 60" `Quick
      test_checker_is_always_capped_at_sixty;
    Alcotest.test_case "role dispatch enforces frozen leap rule" `Quick
      test_role_dispatch_enforces_frozen_rule;
  ]

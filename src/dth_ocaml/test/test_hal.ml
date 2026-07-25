open Dth_engine
open Hal

(* helper functions *)

let advance g n =
  let rec loop g n =
    if n = 0 then g
    else
      let g', _ = Dth_engine.Game.resolve_half_round g 1 60 (Some true) in
      loop g' (n - 1)
  in
  loop g n
(* ------------------------------lookup ------------------------------ *)

let test_lookup_returns_none_for_round_10 () =
  let game = Dth_engine.Game.create ~first_dropper:Domain.Hal () in
  (* advance to round 10 by resolving 18 half-rounds *)
  let game10 = advance game 18 in
  Alcotest.(check int) "at round 10" 10 (Dth_engine.Game.round_num game10);
  Alcotest.(check bool) "lookup returns None" true (lookup game10 = None)

let test_lookup_returns_some_for_round_1_first () =
  let game = Dth_engine.Game.create ~first_dropper:Domain.Hal () in
  Alcotest.(check bool) "lookup round 1 First is Some" true (lookup game <> None)

let test_lookup_returns_some_for_round_9_second () =
  let game = Dth_engine.Game.create ~first_dropper:Domain.Hal () in
  (* 16 half-rounds to reach round 9 first half, then one more for second *)
  let game9s = advance game 17 in
  Alcotest.(check bool)
    "lookup round 9 Second is Some" true
    (lookup game9s <> None)

(* ----------choose_action canonical values ------------------------ *)

let test_canonical_round1_first () =
  let game = Dth_engine.Game.create ~first_dropper:Domain.Hal () in
  Alcotest.(check int)
    "round 1 first action" 60
    (Dth_engine.Hal.choose_action game)

let test_canonical_round1_second () =
  let game = Dth_engine.Game.create ~first_dropper:Domain.Hal () in
  let game', _ = Dth_engine.Game.resolve_half_round game 60 60 (Some true) in
  Alcotest.(check int)
    "round 1 second action" 25
    (Dth_engine.Hal.choose_action game')

let test_canonical_round2_first () =
  let game = Dth_engine.Game.create ~first_dropper:Domain.Hal () in
  let game' =
    let g, _ = Dth_engine.Game.resolve_half_round game 60 60 (Some true) in
    let g, _ = Dth_engine.Game.resolve_half_round g 25 25 (Some true) in
    g
  in
  Alcotest.(check int)
    "round 2 first action" 35
    (Dth_engine.Hal.choose_action game')

(* ----------fallback policy ------------------------ *)

let test_fallback_hal_is_dropper_returns_1 () =
  let game = Dth_engine.Game.create ~first_dropper:Domain.Hal () in
  (* round 10 first half: Hal is dropper *)
  let game10 = advance game 18 in
  Alcotest.(check bool)
    "on First half" true
    (Dth_engine.Game.current_half game10 = Domain.First);
  Alcotest.(check int)
    "fallback dropper returns 1" 1
    (Dth_engine.Hal.choose_action game10)

let test_fallback_hal_is_checker_returns_turn_duration () =
  let game = Dth_engine.Game.create ~first_dropper:Domain.Hal () in
  (* round 10 second half: Hal is checker *)
  let game10s = advance game 19 in
  Alcotest.(check bool)
    "on Second half" true
    (Dth_engine.Game.current_half game10s = Domain.Second);
  let expected = Dth_engine.Game.get_turn_duration game10s in
  Alcotest.(check int)
    "fallback checker returns turn_duration" expected
    (Dth_engine.Hal.choose_action game10s)

(* ----------canonical full table ------------------------ *)

let canonical_table =
  [
    (1, Domain.First, 60);
    (1, Domain.Second, 25);
    (2, Domain.First, 35);
    (2, Domain.Second, 5);
    (3, Domain.First, 56);
    (3, Domain.Second, 60);
    (4, Domain.First, 7);
    (4, Domain.Second, 60);
    (5, Domain.First, 1);
    (5, Domain.Second, 16);
    (6, Domain.First, 60);
    (6, Domain.Second, 10);
    (7, Domain.First, 1);
    (7, Domain.Second, 1);
    (8, Domain.First, 1);
    (8, Domain.Second, 1);
    (9, Domain.First, 1);
    (9, Domain.Second, 60);
  ]

let test_canonical_full_table () =
  let game = Dth_engine.Game.create ~first_dropper:Domain.Hal () in
  let _final =
    List.fold_left
      (fun g (expected_round, expected_half, expected_action) ->
        let label =
          Printf.sprintf "round %d %s" expected_round
            (if expected_half = Domain.First then "First" else "Second")
        in
        Alcotest.(check int)
          (label ^ " round_num") expected_round
          (Dth_engine.Game.round_num g);
        Alcotest.(check bool)
          (label ^ " half") true
          (Dth_engine.Game.current_half g = expected_half);
        Alcotest.(check int)
          (label ^ " action") expected_action
          (Dth_engine.Hal.choose_action g);
        let g, _ =
          Dth_engine.Game.resolve_half_round g expected_action expected_action
            (Some true)
        in
        g)
      game canonical_table
  in
  ()

(* ----------choose_action bounds invariant ------------------------ *)

let test_choose_action_bounds_invariant () =
  (* Action must be in [1, turn_duration] for every reachable state, covering
     canonical rounds 1-9 and fallback rounds 10-12. *)
  let game = Dth_engine.Game.create ~first_dropper:Domain.Hal () in
  let rec loop g n =
    if n = 0 then ()
    else
      let action = Dth_engine.Hal.choose_action g in
      let td = Dth_engine.Game.get_turn_duration g in
      Alcotest.(check bool)
        (Printf.sprintf "action >= 1 (halves left=%d, action=%d)" n action)
        true (action >= 1);
      Alcotest.(check bool)
        (Printf.sprintf
           "action <= turn_duration (halves left=%d, action=%d, td=%d)" n action
           td)
        true (action <= td);
      let g, _ = Dth_engine.Game.resolve_half_round g 1 60 (Some true) in
      loop g (n - 1)
  in
  loop game 24

(* ----------deep fallback ------------------------ *)

let test_lookup_none_for_round_15 () =
  let game = Dth_engine.Game.create ~first_dropper:Domain.Hal () in
  let game15 = advance game 28 in
  Alcotest.(check int) "at round 15" 15 (Dth_engine.Game.round_num game15);
  Alcotest.(check bool) "lookup returns None" true (lookup game15 = None)

let test_fallback_persists_past_round_11 () =
  let game = Dth_engine.Game.create ~first_dropper:Domain.Hal () in
  let game11_first = advance game 20 in
  Alcotest.(check int) "at round 11" 11 (Dth_engine.Game.round_num game11_first);
  Alcotest.(check int)
    "fallback dropper still returns 1 at round 11" 1
    (Dth_engine.Hal.choose_action game11_first);
  let game12_second = advance game 23 in
  let expected = Dth_engine.Game.get_turn_duration game12_second in
  Alcotest.(check int)
    "fallback checker still returns turn_duration at round 12" expected
    (Dth_engine.Hal.choose_action game12_second)

(* ----------test list ------------------------ *)

let tests =
  [
    Alcotest.test_case "lookup: None for round 10" `Quick
      test_lookup_returns_none_for_round_10;
    Alcotest.test_case "lookup: Some for round 1 First" `Quick
      test_lookup_returns_some_for_round_1_first;
    Alcotest.test_case "lookup: Some for round 9 Second" `Quick
      test_lookup_returns_some_for_round_9_second;
    Alcotest.test_case "lookup: None for round 15" `Quick
      test_lookup_none_for_round_15;
    Alcotest.test_case "canonical: round 1 First = 60" `Quick
      test_canonical_round1_first;
    Alcotest.test_case "canonical: round 1 Second = 25" `Quick
      test_canonical_round1_second;
    Alcotest.test_case "canonical: round 2 First = 35" `Quick
      test_canonical_round2_first;
    Alcotest.test_case "canonical: full table walk (rounds 1-9)" `Quick
      test_canonical_full_table;
    Alcotest.test_case "choose_action: bounds invariant (1-12)" `Quick
      test_choose_action_bounds_invariant;
    Alcotest.test_case "fallback: Hal dropper returns 1" `Quick
      test_fallback_hal_is_dropper_returns_1;
    Alcotest.test_case "fallback: Hal checker returns turn_duration" `Quick
      test_fallback_hal_is_checker_returns_turn_duration;
    Alcotest.test_case "fallback: persists past round 11" `Quick
      test_fallback_persists_past_round_11;
  ]

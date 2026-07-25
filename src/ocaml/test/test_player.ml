open Dth_engine.Domain
open Dth_engine.Player

let config = Dth_engine.Config.default ()

let pp_player fmt = function
  | Hal -> Format.pp_print_string fmt "Hal"
  | Baku -> Format.pp_print_string fmt "Baku"

let pp_life_state fmt = function
  | Alive -> Format.pp_print_string fmt "Alive"
  | Dead -> Format.pp_print_string fmt "Dead"

let pp_death_history fmt history =
  let pp_int fmt n = Format.pp_print_int fmt n in
  Format.fprintf fmt "[%a]"
    (Format.pp_print_list
       ~pp_sep:(fun fmt () -> Format.pp_print_string fmt "; ")
       pp_int)
    history

let equal_player_state a b =
  a.id = b.id
  && a.cylinder_seconds = b.cylinder_seconds
  && a.ttd_seconds = b.ttd_seconds
  && a.deaths = b.deaths && a.status = b.status
  && a.death_history = b.death_history

let pp_player_state fmt p =
  Format.fprintf fmt
    "{ id = %a; cylinder_seconds = %d; ttd_seconds = %d; deaths = %d; status = \
     %a; death_history = %a }"
    pp_player p.id p.cylinder_seconds p.ttd_seconds p.deaths
    pp_life_state p.status pp_death_history p.death_history

let player_state_testable = Alcotest.testable pp_player_state equal_player_state

let test_create_baku () =
  let expected =
    {
      id = Baku;
      cylinder_seconds = 0;
      ttd_seconds = 0;
      deaths = 0;
      status = Alive;
      death_history = [];
    }
  in
  Alcotest.(check player_state_testable)
    "creates Baku with zeroed state" expected
    (create_player config Baku)

let test_create_hal () =
  let expected =
    {
      id = Hal;
      cylinder_seconds = 0;
      ttd_seconds = 0;
      deaths = 0;
      status = Alive;
      death_history = [];
    }
  in
  Alcotest.(check player_state_testable)
    "creates Hal with zeroed state" expected
    (create_player config Hal)

let test_add_to_cylinder () =
  let expected =
    {
      id = Hal;
      cylinder_seconds = 42;
      ttd_seconds = 0;
      deaths = 0;
      status = Alive;
      death_history = [];
    }
  in
  Alcotest.(check player_state_testable)
    "adds to cylinder without changing other fields" expected
    (create_player config Hal |> fun p -> add_to_cylinder p 42)

let test_add_to_ttd () =
  let expected =
    {
      id = Hal;
      cylinder_seconds = 0;
      ttd_seconds = 75;
      deaths = 0;
      status = Alive;
      death_history = [];
    }
  in
  Alcotest.(check player_state_testable)
    "adds to ttd without changing other fields" expected
    (create_player config Hal |> fun p -> add_to_ttd p 75)

let test_add_to_ttd_accumulates () =
  let player =
    create_player config Baku |> fun p ->
    add_to_ttd p 60 |> fun p -> add_to_ttd p 45
  in
  Alcotest.(check int)
    "successive add_to_ttd calls accumulate" 105 player.ttd_seconds

let test_get_cylinder () =
  let player = create_player config Hal |> fun p -> add_to_cylinder p 123 in
  Alcotest.(check int)
    "get_cylinder returns cylinder_seconds" 123 (get_cylinder player)

let test_get_cylinder_fresh () =
  Alcotest.(check int)
    "fresh player has zero cylinder" 0
    (get_cylinder (create_player config Hal))

let test_get_ttd () =
  let player = create_player config Baku |> fun p -> add_to_ttd p 200 in
  Alcotest.(check int) "get_ttd returns ttd_seconds" 200 (get_ttd player)

let test_get_ttd_fresh () =
  Alcotest.(check int)
    "fresh player has zero ttd" 0
    (get_ttd (create_player config Baku))

let test_atc_checked_overflow () =
  let p = create_player config Hal |> fun p -> add_to_cylinder p 250 in
  let updated, report = atc_checked config p 50 in
  Alcotest.(check int)
    "cylinder updated to max fatal total" 300 updated.cylinder_seconds;
  Alcotest.(check bool) "report is overflow" true (report = Overflow)

let test_atc_checked_safe () =
  let p = create_player config Hal |> fun p -> add_to_cylinder p 250 in
  let updated, report = atc_checked config p 49 in
  Alcotest.(check int)
    "cylinder updated to one under max fatal total" 299 updated.cylinder_seconds;
  Alcotest.(check bool) "report is no overflow" true (report = No_overflow)

let test_on_death () =
  let expected =
    {
      id = Hal;
      cylinder_seconds = 42;
      ttd_seconds = 60;
      deaths = 1;
      status = Dead;
      death_history = [ 60 ];
    }
  in
  let player = create_player config Hal |> fun p -> add_to_cylinder p 42 in
  Alcotest.(check player_state_testable)
    "on_death updates death counters, status, and history" expected
    (on_death player 60)

let test_on_revival () =
  let expected =
    {
      id = Hal;
      cylinder_seconds = 0;
      ttd_seconds = 60;
      deaths = 1;
      status = Alive;
      death_history = [ 60 ];
    }
  in
  let player =
    create_player config Hal |> fun p ->
    add_to_cylinder p 42 |> fun p -> on_death p 60
  in
  Alcotest.(check player_state_testable)
    "on_revival resets cylinder and revives without losing death history"
    expected (on_revival player)

let test_on_perm_death () =
  let expected =
    {
      id = Hal;
      cylinder_seconds = 42;
      ttd_seconds = 60;
      deaths = 1;
      status = Dead;
      death_history = [ 60 ];
    }
  in
  let player =
    create_player config Hal |> fun p ->
    add_to_cylinder p 42 |> fun p -> on_death p 60
  in
  Alcotest.(check player_state_testable)
    "on_perm_death preserves state and leaves player dead" expected
    (on_perm_death player)

let test_remaining_cap () =
  let player = create_player config Baku |> fun p -> add_to_cylinder p 250 in
  Alcotest.(check int)
    "remaining cap after 250 cylinder seconds" 50
    (remaining_cap config player)

let test_can_absorb_injection_rejects_threshold () =
  let player = create_player config Baku |> fun p -> add_to_cylinder p 250 in
  Alcotest.(check bool)
    "rejects amount that reaches fatal threshold" false
    (can_absorb_injection config player 50)

let test_can_absorb_injection_allows_safe_amount () =
  let player = create_player config Baku |> fun p -> add_to_cylinder p 250 in
  Alcotest.(check bool)
    "allows amount that stays under fatal threshold" true
    (can_absorb_injection config player 49)

let test_safe_checks_remaining_at_zero () =
  let player = create_player config Hal in
  Alcotest.(check int)
    "fresh player has four safe checks remaining" 4
    (safe_checks_remaining config player)

let test_safe_checks_remaining_near_threshold () =
  let player = create_player config Hal |> fun p -> add_to_cylinder p 240 in
  Alcotest.(check int)
    "player at 240 cylinder has no safe checks remaining" 0
    (safe_checks_remaining config player)

let tests =
  [
    Alcotest.test_case "create Baku" `Quick test_create_baku;
    Alcotest.test_case "create Hal" `Quick test_create_hal;
    Alcotest.test_case "add to cylinder" `Quick test_add_to_cylinder;
    Alcotest.test_case "add to ttd" `Quick test_add_to_ttd;
    Alcotest.test_case "add to ttd accumulates" `Quick
      test_add_to_ttd_accumulates;
    Alcotest.test_case "get cylinder" `Quick test_get_cylinder;
    Alcotest.test_case "get cylinder fresh" `Quick test_get_cylinder_fresh;
    Alcotest.test_case "get ttd" `Quick test_get_ttd;
    Alcotest.test_case "get ttd fresh" `Quick test_get_ttd_fresh;
    Alcotest.test_case "adding to 300 in cylinder is an overflow" `Quick
      test_atc_checked_overflow;
    Alcotest.test_case "adding to 299 in cylinder is not an overflow" `Quick
      test_atc_checked_safe;
    Alcotest.test_case "on death updates player state" `Quick test_on_death;
    Alcotest.test_case "on revival updates player state" `Quick test_on_revival;
    Alcotest.test_case "on permanent death updates player state" `Quick
      test_on_perm_death;
    Alcotest.test_case "remaining cap" `Quick test_remaining_cap;
    Alcotest.test_case "can absorb rejects threshold" `Quick
      test_can_absorb_injection_rejects_threshold;
    Alcotest.test_case "can absorb allows safe amount" `Quick
      test_can_absorb_injection_allows_safe_amount;
    Alcotest.test_case "safe checks remaining at zero cylinder" `Quick
      test_safe_checks_remaining_at_zero;
    Alcotest.test_case "safe checks remaining near threshold" `Quick
      test_safe_checks_remaining_near_threshold;
  ]

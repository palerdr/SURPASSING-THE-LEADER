open Dth_engine
open Game

let prime_baku_cylinder game =
  let rec loop g n =
    if n = 0 then g
    else
      let g, _ = Game.resolve_half_round g 1 60 (Some true) in
      let g, _ = Game.resolve_half_round g 60 60 (Some true) in
      loop g (n - 1)
  in
  loop game 4

(* validate_drop_time --- *)

let test_validate_drop_time_zero_raises () =
  Alcotest.check_raises "drop_time 0 raises" (Failure "Invalid drop-time")
    (fun () -> Game.validate_drop_time Domain.Hal 0 60)

let test_validate_drop_time_at_turn_duration_ok () =
  Game.validate_drop_time Domain.Hal 60 60
(* no exception means it has passed *)

let test_validate_drop_time_above_turn_duration_raises () =
  Alcotest.check_raises "drop_time > turn_duration raises"
    (Failure "Invalid drop-time") (fun () ->
      Game.validate_drop_time Domain.Baku 61 60)

let test_validate_drop_time_one_ok () =
  Game.validate_drop_time Domain.Hal 1 60

let test_validate_leap_drop_is_baku_only () =
  Game.validate_drop_time Domain.Baku 61 61;
  Alcotest.check_raises "Hal cannot drop on second 61"
    (Failure "Invalid drop-time") (fun () ->
      Game.validate_drop_time Domain.Hal 61 61)

(* validate_check_time ---*)

let test_validate_check_time_zero_raises () =
  Alcotest.check_raises "check_time 0 raises" (Failure "Invalid checking-time")
    (fun () -> Game.validate_check_time 0 60)

let test_validate_check_time_at_turn_duration_ok () =
  Game.validate_check_time 60 60

let test_validate_check_time_above_turn_duration_raises () =
  Alcotest.check_raises "check_time > turn_duration raises"
    (Failure "Invalid checking-time") (fun () -> Game.validate_check_time 61 60)

let test_validate_check_time_one_ok () = Game.validate_check_time 1 60

let test_validate_checker_capped_during_leap () =
  Alcotest.check_raises "Checker cannot use second 61"
    (Failure "Invalid checking-time") (fun () ->
      Game.validate_check_time 61 61)

(* ── snap_clock_to_next_minute ──────────────────────────────────────────── *)

let test_snap_clock_strictly_advances () =
  let game = Game.create () in
  let clock_before = Game.game_clock game in
  let game' = Game.snap_clock_to_next_minute game in
  Alcotest.(check bool)
    "clock strictly advanced" true
    (Game.game_clock game' > clock_before)

let test_snap_clock_on_boundary_still_advances () =
  (* advance the clock to an exact minute boundary then snap again *)
  let game = Game.create () in
  let config = Game.config game in
  let spm = config.time.seconds_per_minute in
  let clock_before = Game.game_clock game in
  (* find the next boundary manually *)
  let boundary = ((clock_before / spm) + 1) * spm in
  let game_on_boundary = Game.advance_clock game (boundary - clock_before) in
  let game' = Game.snap_clock_to_next_minute game_on_boundary in
  Alcotest.(check bool)
    "snap from boundary still moves forward" true
    (Game.game_clock game' > Game.game_clock game_on_boundary)

(* history length --- *)

let test_history_empty_at_start () =
  let game = Game.create () in
  Alcotest.(check int) "history empty" 0 (List.length (Game.history game))

let test_history_length_after_one_half_round () =
  let game = Game.create () in
  let game', _ = Game.play_half_round game 30 45 in
  Alcotest.(check int) "history length 1" 1 (List.length (Game.history game'))

let test_history_length_after_two_half_rounds () =
  let game = Game.create () in
  let game', _ = Game.play_half_round game 30 45 in
  let game'', _ = Game.play_half_round game' 30 45 in
  Alcotest.(check int) "history length 2" 2 (List.length (Game.history game''))

(* half / round advancement --- *)

let test_first_half_advances_to_second () =
  let game = Game.create () in
  Alcotest.(check bool)
    "starts on First" true
    (Game.current_half game = Domain.First);
  let game', _ = Game.play_half_round game 30 45 in
  Alcotest.(check bool)
    "now on Second" true
    (Game.current_half game' = Domain.Second)

let test_second_half_increments_round () =
  let game = Game.create () in
  let game', _ = Game.play_half_round game 30 45 in
  let round_before = Game.round_num game' in
  let game'', _ = Game.play_half_round game' 30 45 in
  Alcotest.(check int)
    "round incremented" (round_before + 1) (Game.round_num game'')

let test_second_half_returns_to_first () =
  let game = Game.create () in
  let game', _ = Game.play_half_round game 30 45 in
  let game'', _ = Game.play_half_round game' 30 45 in
  Alcotest.(check bool)
    "back to First" true
    (Game.current_half game'' = Domain.First)

(* clock advances by turn_duration on check_success --- *)

let test_clock_advances_by_turn_duration_on_success () =
  let game = Game.create () in
  let turn_duration = Game.get_turn_duration game in
  let clock_before = Game.game_clock game in
  (* check_time > drop_time, no overflow possible with small values *)
  let game', record = Game.resolve_half_round game 1 60 None in
  let _ = record in
  (* clock should have advanced by at least turn_duration; it may also include
     within_round_overhead added before the Second half starts *)
  Alcotest.(check bool)
    "clock advanced by at least turn_duration" true
    (Game.game_clock game' >= clock_before + turn_duration)

let test_check_success_result_recorded () =
  let game = Game.create () in
  let _game', record = Game.resolve_half_round game 1 60 None in
  Alcotest.(check bool)
    "result is Check_success" true
    (record.result = Check_success)

(* game_over guard --- *)

let test_play_on_finished_game_raises () =
  let game = Game.create () in
  (* force checker (Baku on First half) to die: check before drop, force
     death *)
  let game', _ = Game.resolve_half_round game 60 1 (Some false) in
  Alcotest.(check bool) "game is over" true (Game.game_over game');
  Alcotest.check_raises "second play raises" (Failure "Game is already over")
    (fun () -> ignore (Game.play_half_round game' 1 60))

(* forced death outcomes ---*)

let test_forced_survival_does_not_end_game () =
  let game = Game.create () in
  let game', _ = Game.resolve_half_round game 60 1 (Some true) in
  Alcotest.(check bool)
    "game not over after survival" false (Game.game_over game');
  Alcotest.(check bool) "winner is None" true (Game.winner game' = None)

let test_forced_death_ends_game_with_correct_winner () =
  let game = Game.create () in
  (* First half: Hal drops, Baku checks... check_time < drop_time => Baku
     dies *)
  let game', _ = Game.resolve_half_round game 60 1 (Some false) in
  Alcotest.(check bool) "game over" true (Game.game_over game');
  Alcotest.(check bool) "Hal wins" true (Game.winner game' = Some Domain.Hal);
  Alcotest.(check bool) "Baku loses" true (Game.loser game' = Some Domain.Baku)

let test_check_fail_survived () =
  let game = Game.create () in
  let game', record = Game.resolve_half_round game 60 1 (Some true) in
  Alcotest.(check bool)
    "result is Check_fail_survived" true
    (record.result = Check_fail_survived);
  Alcotest.(check bool) "game continues" false (Game.game_over game');
  Alcotest.(check bool) "winner still None" true (Game.winner game' = None);
  Alcotest.(check (option bool)) "survived recorded" (Some true) record.survived

let test_cylinder_overflow_survived () =
  let game = Game.create () |> prime_baku_cylinder in
  let game', record = Game.resolve_half_round game 1 60 (Some true) in
  Alcotest.(check bool)
    "result is Cylinder_overflow_survived" true
    (record.result = Cylinder_overflow_survived);
  Alcotest.(check bool)
    "game continues after revival" false (Game.game_over game');
  Alcotest.(check int)
    "death_duration equals cylinder.max" (Game.config game').cylinder.max
    record.death_duration

let test_cylinder_overflow_died () =
  let game = Game.create () |> prime_baku_cylinder in
  let game', record = Game.resolve_half_round game 1 60 (Some false) in
  Alcotest.(check bool)
    "result is Cylinder_overflow_died" true
    (record.result = Cylinder_overflow_died);
  Alcotest.(check bool) "game over" true (Game.game_over game');
  Alcotest.(check bool) "Hal wins" true (Game.winner game' = Some Domain.Hal);
  Alcotest.(check bool) "Baku loses" true (Game.loser game' = Some Domain.Baku)

let test_round_does_not_increment_after_death () =
  let game = Game.create () in
  let round_before = Game.round_num game in
  let game', _ = Game.resolve_half_round game 1 60 None in
  let game'', _ = Game.resolve_half_round game' 60 1 (Some false) in
  Alcotest.(check int)
    "round did not increment past death" round_before (Game.round_num game'')

(* roles & first_dropper --- *)

let test_roles_default_first_dropper_first_half () =
  let game = Game.create () in
  let dropper, checker = Game.get_roles_for_half game in
  Alcotest.(check bool) "Hal drops" true (dropper.id = Domain.Hal);
  Alcotest.(check bool) "Baku checks" true (checker.id = Domain.Baku)

let test_roles_default_first_dropper_second_half () =
  let game = Game.create () in
  let game', _ = Game.resolve_half_round game 30 45 (Some true) in
  let dropper, checker = Game.get_roles_for_half game' in
  Alcotest.(check bool) "Baku drops Second half" true (dropper.id = Domain.Baku);
  Alcotest.(check bool) "Hal checks Second half" true (checker.id = Domain.Hal)

let test_roles_baku_first_dropper_first_half () =
  let game = Game.create ~first_dropper:Domain.Baku () in
  let dropper, checker = Game.get_roles_for_half game in
  Alcotest.(check bool) "Baku drops" true (dropper.id = Domain.Baku);
  Alcotest.(check bool) "Hal checks" true (checker.id = Domain.Hal)

let test_roles_baku_first_dropper_second_half () =
  let game = Game.create ~first_dropper:Domain.Baku () in
  let game', _ = Game.resolve_half_round game 30 45 (Some true) in
  let dropper, checker = Game.get_roles_for_half game' in
  Alcotest.(check bool) "Hal drops Second half" true (dropper.id = Domain.Hal);
  Alcotest.(check bool) "Baku checks Second half" true (checker.id = Domain.Baku)

(* leap window --- *)

let test_is_leap_second_turn_false_at_opening () =
  let game = Game.create () in
  Alcotest.(check bool)
    "opening is not in leap window" false
    (Game.is_leap_second_turn game)

let test_is_leap_second_turn_true_in_window () =
  let game = Game.create () in
  let config = Game.config game in
  let jump = config.time.ls_window_start - Game.game_clock game in
  let game' = Game.advance_clock game jump in
  Alcotest.(check bool) "in leap window" true (Game.is_leap_second_turn game')

let test_get_turn_duration_leap () =
  let game = Game.create () in
  let config = Game.config game in
  let jump = config.time.ls_window_start - Game.game_clock game in
  let game' = Game.advance_clock game jump in
  Alcotest.(check int)
    "leap turn duration" config.turn.duration_leap
    (Game.get_turn_duration game')

let test_get_turn_duration_normal () =
  let game = Game.create () in
  let config = Game.config game in
  Alcotest.(check int)
    "opening turn duration" config.turn.duration_normal
    (Game.get_turn_duration game)

(* seed determinism --- *)

let test_seed_determinism_on_death () =
  let g1 = Game.create ~seed:99 () in
  let g2 = Game.create ~seed:99 () in
  let g1', r1 = Game.play_half_round g1 60 1 in
  let g2', r2 = Game.play_half_round g2 60 1 in
  Alcotest.(check (option bool)) "same survival outcome" r1.survived r2.survived;
  Alcotest.(check bool)
    "same game_over" (Game.game_over g1') (Game.game_over g2');
  Alcotest.(check int)
    "same clock after death" (Game.game_clock g1') (Game.game_clock g2')

(* record field coverage --- *)

let test_record_fields_on_success () =
  let game = Game.create () in
  let start_clock = Game.game_clock game in
  let td = Game.get_turn_duration game in
  let _g', r = Game.resolve_half_round game 20 45 (Some true) in
  Alcotest.(check int) "round_num" 1 r.round_num;
  Alcotest.(check bool) "half = First" true (r.half = Domain.First);
  Alcotest.(check bool) "dropper = Hal" true (r.dropper = Domain.Hal);
  Alcotest.(check bool) "checker = Baku" true (r.checker = Domain.Baku);
  Alcotest.(check int) "drop_time" 20 r.drop_time;
  Alcotest.(check int) "check_time" 45 r.check_time;
  Alcotest.(check int) "turn_duration" td r.turn_duration;
  Alcotest.(check int) "st_gained = check - drop + 1" 26 r.st_gained;
  Alcotest.(check int) "death_duration 0 on success" 0 r.death_duration;
  Alcotest.(check (option bool)) "survived None on success" None r.survived;
  Alcotest.(check int) "game_clock_at_start" start_clock r.game_clock_at_start;
  Alcotest.(check bool)
    "survival_probability None on success" true
    (r.survival_probability = None)

let test_record_fields_on_forced_death () =
  let game = Game.create () in
  let _g', r = Game.resolve_half_round game 60 1 (Some false) in
  Alcotest.(check (option bool)) "survived = Some false" (Some false) r.survived;
  Alcotest.(check bool)
    "survival_probability = Some _" true
    (r.survival_probability <> None);
  Alcotest.(check bool) "death_duration > 0" true (r.death_duration > 0)

(* clock advancement after death --- *)

let test_clock_advances_by_death_overhead () =
  let game = Game.create () in
  let config = Game.config game in
  let before = Game.game_clock game in
  let td = Game.get_turn_duration game in
  let g', r = Game.resolve_half_round game 60 1 (Some true) in
  let expected_min =
    before + td + r.death_duration + config.cylinder.death_procedure_overhead
  in
  Alcotest.(check bool)
    "clock advanced by at least turn + death + overhead" true
    (Game.game_clock g' >= expected_min)

(* accessor smoke tests --- *)

let test_get_player_state_returns_matching_ids () =
  let game = Game.create () in
  let hal = Game.get_player_state game Domain.Hal in
  let baku = Game.get_player_state game Domain.Baku in
  Alcotest.(check bool) "Hal lookup matches id" true (hal.id = Domain.Hal);
  Alcotest.(check bool) "Baku lookup matches id" true (baku.id = Domain.Baku)

let test_format_game_clock_opening () =
  let game = Game.create () in
  Alcotest.(check string)
    "opening wall time" "08:12:00"
    (Game.format_game_clock game)

let test_advance_clock_adds_exactly () =
  let game = Game.create () in
  let before = Game.game_clock game in
  let game' = Game.advance_clock game 123 in
  Alcotest.(check int)
    "clock advanced by 123" (before + 123) (Game.game_clock game')

let test_get_safe_checks_fresh () =
  let game = Game.create () in
  let config = Game.config game in
  let expected_max = config.cylinder.max / config.turn.duration_normal in
  let baku_safe = Game.get_safe_checks game Domain.Baku in
  Alcotest.(check bool)
    "fresh Baku has >= max/turn_duration safe checks" true
    (baku_safe >= expected_max - 1)

(* ----------- test list ----------- *)

let tests =
  [
    Alcotest.test_case "validate_drop_time: 0 raises" `Quick
      test_validate_drop_time_zero_raises;
    Alcotest.test_case "validate_drop_time: at turn_duration ok" `Quick
      test_validate_drop_time_at_turn_duration_ok;
    Alcotest.test_case "validate_drop_time: above turn_duration raises" `Quick
      test_validate_drop_time_above_turn_duration_raises;
    Alcotest.test_case "validate_drop_time: 1 ok" `Quick
      test_validate_drop_time_one_ok;
    Alcotest.test_case "validate_drop_time: leap is Baku-only" `Quick
      test_validate_leap_drop_is_baku_only;
    Alcotest.test_case "validate_check_time: 0 raises" `Quick
      test_validate_check_time_zero_raises;
    Alcotest.test_case "validate_check_time: at turn_duration ok" `Quick
      test_validate_check_time_at_turn_duration_ok;
    Alcotest.test_case "validate_check_time: above turn_duration raises" `Quick
      test_validate_check_time_above_turn_duration_raises;
    Alcotest.test_case "validate_check_time: 1 ok" `Quick
      test_validate_check_time_one_ok;
    Alcotest.test_case "validate_check_time: leap remains capped" `Quick
      test_validate_checker_capped_during_leap;
    Alcotest.test_case "snap_clock strictly advances" `Quick
      test_snap_clock_strictly_advances;
    Alcotest.test_case "snap_clock on boundary still advances" `Quick
      test_snap_clock_on_boundary_still_advances;
    Alcotest.test_case "history empty at start" `Quick
      test_history_empty_at_start;
    Alcotest.test_case "history length after one half" `Quick
      test_history_length_after_one_half_round;
    Alcotest.test_case "history length after two halves" `Quick
      test_history_length_after_two_half_rounds;
    Alcotest.test_case "first half advances to second" `Quick
      test_first_half_advances_to_second;
    Alcotest.test_case "second half increments round" `Quick
      test_second_half_increments_round;
    Alcotest.test_case "second half returns to first" `Quick
      test_second_half_returns_to_first;
    Alcotest.test_case "clock advances by turn_duration on success" `Quick
      test_clock_advances_by_turn_duration_on_success;
    Alcotest.test_case "check_success result recorded" `Quick
      test_check_success_result_recorded;
    Alcotest.test_case "play on finished game raises" `Quick
      test_play_on_finished_game_raises;
    Alcotest.test_case "forced survival does not end game" `Quick
      test_forced_survival_does_not_end_game;
    Alcotest.test_case "forced death ends game with correct winner" `Quick
      test_forced_death_ends_game_with_correct_winner;
    Alcotest.test_case "check fail survived" `Quick test_check_fail_survived;
    Alcotest.test_case "cylinder overflow survived" `Quick
      test_cylinder_overflow_survived;
    Alcotest.test_case "cylinder overflow died" `Quick
      test_cylinder_overflow_died;
    Alcotest.test_case "round does not increment after death" `Quick
      test_round_does_not_increment_after_death;
    Alcotest.test_case "roles: default first_dropper, First half" `Quick
      test_roles_default_first_dropper_first_half;
    Alcotest.test_case "roles: default first_dropper, Second half" `Quick
      test_roles_default_first_dropper_second_half;
    Alcotest.test_case "roles: Baku first_dropper, First half" `Quick
      test_roles_baku_first_dropper_first_half;
    Alcotest.test_case "roles: Baku first_dropper, Second half" `Quick
      test_roles_baku_first_dropper_second_half;
    Alcotest.test_case "leap: is_leap_second_turn false at opening" `Quick
      test_is_leap_second_turn_false_at_opening;
    Alcotest.test_case "leap: is_leap_second_turn true in window" `Quick
      test_is_leap_second_turn_true_in_window;
    Alcotest.test_case "leap: get_turn_duration in window" `Quick
      test_get_turn_duration_leap;
    Alcotest.test_case "leap: get_turn_duration normal" `Quick
      test_get_turn_duration_normal;
    Alcotest.test_case "seed determinism on death" `Quick
      test_seed_determinism_on_death;
    Alcotest.test_case "record fields on success" `Quick
      test_record_fields_on_success;
    Alcotest.test_case "record fields on forced death" `Quick
      test_record_fields_on_forced_death;
    Alcotest.test_case "clock advances by death overhead" `Quick
      test_clock_advances_by_death_overhead;
    Alcotest.test_case "get_player_state returns matching ids" `Quick
      test_get_player_state_returns_matching_ids;
    Alcotest.test_case "format_game_clock at opening" `Quick
      test_format_game_clock_opening;
    Alcotest.test_case "advance_clock adds exactly" `Quick
      test_advance_clock_adds_exactly;
    Alcotest.test_case "get_safe_checks fresh Baku" `Quick
      test_get_safe_checks_fresh;
  ]

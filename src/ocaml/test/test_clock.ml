open Dth_engine.Clock

let config = Dth_engine.Config.default ()

let pp_wall_time fmt (wt : Dth_engine.Clock.wall_time) =
  Format.fprintf fmt "%02d:%02d:%02d" wt.hours wt.minutes wt.seconds

let equal_wall_time (a : Dth_engine.Clock.wall_time)
    (b : Dth_engine.Clock.wall_time) =
  a.hours = b.hours && a.minutes = b.minutes && a.seconds = b.seconds

let wall_time_testable = Alcotest.testable pp_wall_time equal_wall_time

let test_in_leap_window_before_window () =
  Alcotest.(check bool)
    "before leap window" false
    (Dth_engine.Clock.in_leap_window config (config.time.ls_window_start - 1))

let test_in_leap_window_at_window_start () =
  Alcotest.(check bool)
    "window start is included" true
    (Dth_engine.Clock.in_leap_window config config.time.ls_window_start)

let test_in_leap_window_at_window_end () =
  Alcotest.(check bool)
    "window end is included" true
    (Dth_engine.Clock.in_leap_window config config.time.ls_window_end)

let test_in_leap_window_after_window () =
  Alcotest.(check bool)
    "after leap window" false
    (Dth_engine.Clock.in_leap_window config (config.time.ls_window_end + 1))

let test_turn_duration_normal_time () =
  Alcotest.(check int)
    "normal turn duration" config.turn.duration_normal
    (Dth_engine.Clock.turn_duration_for_start_time config 0)

let test_turn_duration_leap_window_time () =
  Alcotest.(check int)
    "leap-window turn duration" config.turn.duration_leap
    (Dth_engine.Clock.turn_duration_for_start_time config
       config.time.ls_window_start)

let test_advance_clock () =
  Alcotest.(check int)
    "advance adds elapsed time" 142
    (Dth_engine.Clock.advance 100 42)

let test_snap_before_ls () =
  Alcotest.(check int)
    "snaps before 3600 to next minute boundary" 3601
    (Dth_engine.Clock.snap_to_next_round_boundary config 3599)

let test_snap_in_ls () =
  Alcotest.(check int)
    "snaps during ls window to 3601" 3601
    (Dth_engine.Clock.snap_to_next_round_boundary config 3580)

let test_snap_after_ls () =
  Alcotest.(check int)
    "snaps after ls window to next boundary + 1" 3721
    (Dth_engine.Clock.snap_to_next_round_boundary config 3701)

let test_wall_time_record_before_ls () =
  let expected : Dth_engine.Clock.wall_time =
    { hours = 8; minutes = 59; seconds = 59 }
  in
  Alcotest.(check wall_time_testable)
    "before leap second has correct wall time record" expected
    (to_wall_time config 3599)

let test_wall_time_record_at_opening_start () =
  let expected : Dth_engine.Clock.wall_time =
    { hours = 8; minutes = 12; seconds = 0 }
  in
  Alcotest.(check wall_time_testable)
    "opening start has correct wall time record" expected
    (to_wall_time config config.time.opening_start_second)

let test_ls_record () =
  let expected : Dth_engine.Clock.wall_time =
    { hours = 8; minutes = 59; seconds = 60 }
  in
  Alcotest.(check wall_time_testable)
    "leap seconds has correct formatted wall time record" expected
    (to_wall_time config 3600)

let test_wall_time_record_after_ls () =
  let expected : Dth_engine.Clock.wall_time =
    { hours = 9; minutes = 0; seconds = 0 }
  in
  Alcotest.(check wall_time_testable)
    "after leap second has correct wall time record" expected
    (to_wall_time config 3601)

let test_format_wall_time () =
  Alcotest.(check string)
    "formats wall time with zero padding" "08:59:60"
    (format_wall_time { hours = 8; minutes = 59; seconds = 60 })

let tests =
  [
    Alcotest.test_case "in leap window before window" `Quick
      test_in_leap_window_before_window;
    Alcotest.test_case "in leap window at start" `Quick
      test_in_leap_window_at_window_start;
    Alcotest.test_case "in leap window at end" `Quick
      test_in_leap_window_at_window_end;
    Alcotest.test_case "in leap window after window" `Quick
      test_in_leap_window_after_window;
    Alcotest.test_case "turn duration outside leap window" `Quick
      test_turn_duration_normal_time;
    Alcotest.test_case "turn duration inside leap window" `Quick
      test_turn_duration_leap_window_time;
    Alcotest.test_case "advance clock" `Quick test_advance_clock;
    Alcotest.test_case "snap before leap second" `Quick test_snap_before_ls;
    Alcotest.test_case "snap during leap second" `Quick test_snap_in_ls;
    Alcotest.test_case "snap after leap second" `Quick test_snap_after_ls;
    Alcotest.test_case "before leap second has correct wall time record" `Quick
      test_wall_time_record_before_ls;
    Alcotest.test_case "opening start has correct wall time record" `Quick
      test_wall_time_record_at_opening_start;
    Alcotest.test_case "leap second has correct record" `Quick test_ls_record;
    Alcotest.test_case "after leap second has correct wall time record" `Quick
      test_wall_time_record_after_ls;
    Alcotest.test_case "formats wall time" `Quick test_format_wall_time;
  ]

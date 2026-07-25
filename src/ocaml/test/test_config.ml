let test_default_values () =
  let config = Dth_engine.Config.default () in
  Alcotest.(check int) "game start hour" 8 config.time.game_start_hour;
  Alcotest.(check int)
    "opening start second" (12 * 60) config.time.opening_start_second;
  Alcotest.(check int) "seconds per minute" 60 config.time.seconds_per_minute;
  Alcotest.(check int) "minutes per hour" 60 config.time.minutes_per_hour;
  Alcotest.(check int)
    "leap-second window start" (59 * 60) config.time.ls_window_start;
  Alcotest.(check int)
    "leap-second window end" (60 * 60) config.time.ls_window_end;
  Alcotest.(check int)
    "within-round overhead" 60 config.time.within_round_overhead;
  Alcotest.(check int) "normal turn duration" 60 config.turn.duration_normal;
  Alcotest.(check int) "leap turn duration" 61 config.turn.duration_leap;
  Alcotest.(check int)
    "failed check penalty" 60 config.turn.failed_check_penalty;
  Alcotest.(check int) "cylinder max" 300 config.cylinder.max;
  Alcotest.(check int)
    "death procedure overhead" 120 config.cylinder.death_procedure_overhead;
  Alcotest.(check (float 0.000001))
    "revival baseline" 0.80 config.survival.baseline;
  Alcotest.(check (float 0.000001))
    "TTD half life" 120.0 config.survival.ttd_half_life_seconds;
  Alcotest.(check (float 0.000001))
    "TTD exponent" 1.3 config.survival.ttd_curve_exponent;
  Alcotest.(check (float 0.000001))
    "effective referee decay" 0.88
    config.survival.effective_referee_decay;
  Alcotest.(check (float 0.000001))
    "effective referee floor" 0.4 config.survival.effective_referee_floor

let tests = [ Alcotest.test_case "default values" `Quick test_default_values ]

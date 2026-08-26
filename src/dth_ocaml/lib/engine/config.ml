type time_config = {
  game_start_hour : int;
  opening_start_second : int;
  seconds_per_minute : int;
  minutes_per_hour : int;
  ls_window_start : int;
  ls_window_end : int;
  within_round_overhead : int;
}

type turn_config = {
  duration_normal : int;
  duration_leap : int;
  failed_check_penalty : int;
}

type cylinder_config = {
  max : int;
  death_procedure_overhead : int;
}

type survival_config = {
  baseline : float;
  ttd_decay_per_minute : float;
}

type t = {
  time : time_config;
  turn : turn_config;
  cylinder : cylinder_config;
  survival : survival_config;
}

let default_config =
  {
    time =
      {
        game_start_hour = 8;
        opening_start_second = 12 * 60;
        seconds_per_minute = 60;
        minutes_per_hour = 60;
        ls_window_start = 59 * 60;
        ls_window_end = 60 * 60;
        within_round_overhead = 60;
      };
    turn =
      { duration_normal = 60; duration_leap = 61; failed_check_penalty = 60 };
    cylinder = { max = 300; death_procedure_overhead = 120 };
    survival = { baseline = 0.95; ttd_decay_per_minute = 0.75 };
  }

let default () = default_config

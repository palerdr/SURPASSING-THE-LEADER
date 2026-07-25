let in_leap_window (config : Config.t) (abs_game_second : int) : bool =
  config.time.ls_window_start <= abs_game_second
  && abs_game_second <= config.time.ls_window_end

let turn_duration_for_start_time (config : Config.t) (start_time : int) : int =
  if in_leap_window config start_time then config.turn.duration_leap
  else config.turn.duration_normal

let advance (abs_clock_time : int) (elapsed_time : int) : int =
  abs_clock_time + elapsed_time

let snap_to_60 (t : int) : int = ((t / 60) + 1) * 60

let snap_to_next_round_boundary (config : Config.t) (end_time : int) : int =
  if end_time < config.time.ls_window_start then snap_to_60 end_time
  else
    let pushed : int = config.time.ls_window_end + 1 in
    if in_leap_window config end_time then pushed
    else
      let elapsed : int = end_time - pushed in
      pushed + snap_to_60 elapsed

type wall_time = {
  hours : int;
  minutes : int;
  seconds : int;
}

let to_wall_time (config : Config.t) (gs : int) : wall_time =
  let start_hour : int = config.time.game_start_hour in
  let seconds_per_minute : int = config.time.seconds_per_minute in
  let seconds_per_hour : int =
    config.time.minutes_per_hour * config.time.seconds_per_minute
  in
  if gs < config.time.ls_window_end then
    {
      hours = start_hour + (gs / seconds_per_hour);
      minutes = gs mod seconds_per_hour / seconds_per_minute;
      seconds = gs mod seconds_per_minute;
    }
  else if gs = config.time.ls_window_end then
    { hours = start_hour; minutes = 59; seconds = 60 }
  else
    let shifted : int = gs - 1 in
    {
      hours = start_hour + (shifted / seconds_per_hour);
      minutes = shifted mod seconds_per_hour / seconds_per_minute;
      seconds = shifted mod seconds_per_minute;
    }

let format_wall_time (wt : wall_time) : string =
  Printf.sprintf "%02d:%02d:%02d" wt.hours wt.minutes wt.seconds

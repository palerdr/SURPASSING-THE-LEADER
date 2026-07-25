open Dth_engine

type timing = int

let range_1_to n = List.init n (fun i -> i + 1)

let legal_drop_times (game : Game.t) (player : Domain.player) : timing list =
  let normal = (Game.config game).Config.turn.duration_normal in
  let upper_bound =
    if
      player = Domain.Baku
      && Game.is_leap_second_turn game
    then Game.get_turn_duration game
    else normal
  in
  range_1_to upper_bound

let legal_check_times (game : Game.t) (_player : Domain.player) : timing list =
  range_1_to (Game.config game).Config.turn.duration_normal

let legal_times_for_role (game : Game.t) (player : Domain.player)
    (role : Domain.role) : timing list =
  match role with
  | Domain.Dropper -> legal_drop_times game player
  | Domain.Checker -> legal_check_times game player

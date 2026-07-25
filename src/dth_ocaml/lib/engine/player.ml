type player_state = {
  id : Domain.player;
  cylinder_seconds : int;
  ttd_seconds : int;
  deaths : int;
  status : Domain.life_state;
  death_history : int list;
}

type cylinder_report =
  | No_overflow
  | Overflow

let create_player (_config : Config.t) (id : Domain.player) : player_state =
  {
    id;
    cylinder_seconds = 0;
    ttd_seconds = 0;
    deaths = 0;
    status = Domain.Alive;
    death_history = [];
  }

let add_to_cylinder (p : player_state) (amt : int) : player_state =
  { p with cylinder_seconds = p.cylinder_seconds + amt }

let add_to_ttd (p : player_state) (amt : int) : player_state =
  { p with ttd_seconds = p.ttd_seconds + amt }

let get_cylinder (p : player_state) : int = p.cylinder_seconds
let get_ttd (p : player_state) : int = p.ttd_seconds

let remaining_cap (config : Config.t) (p : player_state) : int =
  let cylinder : Config.cylinder_config = config.cylinder in
  cylinder.max - p.cylinder_seconds

let can_absorb_injection (config : Config.t) (p : player_state) (amount : int) :
    bool =
  amount < remaining_cap config p

let atc_checked (config : Config.t) (p : player_state) (amt : int) :
    player_state * cylinder_report =
  let overflow_status : cylinder_report =
    if can_absorb_injection config p amt then No_overflow else Overflow
  in
  (add_to_cylinder p amt, overflow_status)

let reset_cylinder (p : player_state) : player_state =
  { p with cylinder_seconds = 0 }

let safe_checks_remaining (config : Config.t) (p : player_state) : int =
  let cylinder : Config.cylinder_config = config.cylinder in
  let turn : Config.turn_config = config.turn in
  max 0 ((cylinder.max - 1 - p.cylinder_seconds) / turn.duration_normal)

let on_death (p : player_state) (time_dead : int) : player_state =
  let p_with_ttd : player_state = add_to_ttd p time_dead in
  {
    p_with_ttd with
    deaths = p_with_ttd.deaths + 1;
    status = Domain.Dead;
    death_history = time_dead :: p_with_ttd.death_history;
  }

let on_revival (p : player_state) : player_state =
  { p with cylinder_seconds = 0; status = Domain.Alive }

let on_perm_death (p : player_state) : player_state =
  { p with status = Domain.Dead }

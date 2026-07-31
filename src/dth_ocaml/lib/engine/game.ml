type half_round_result =
  | Check_success
  | Check_fail_survived
  | Check_fail_died
  | Cylinder_overflow_survived
  | Cylinder_overflow_died

type half_round_record = {
  round_num : int;
  half : Domain.half_index;
  dropper : Domain.player;
  checker : Domain.player;
  drop_time : int;
  check_time : int;
  turn_duration : int;
  result : half_round_result;
  st_gained : int;
  death_duration : int;
  survived : bool option;
  game_clock_at_start : int;
  survival_probability : float option;
}

(* AF: A value [g : t] represents a complete game state of Drop The
   Handkerchief, namely the tuple (game_clock seconds since 8:00 AM, current
   round_number, current_half indicating which player is dropper this half,
   history of all resolved half-rounds in reverse chronological order, the
   player chosen as first dropper at game creation, Baku's and Hal's per-player
   state (cylinder, deaths, ttd, alive flag), winner and loser if the game is
   decided, whether the game is over, a deterministic RNG state from which all
   future survival rolls draw, the immutable Config.t supplying every tunable
   parameter).

   RI: - game_clock >= 0 and is monotonically non-decreasing as half-rounds
   resolve. - round_number >= 1. - current_half is the half about to be played
   (alternates First/Second per round; advancing past Second increments
   round_number and resets to First). - history.length equals the number of
   half-rounds already resolved; the head is the most recent record. -
   first_dropper is fixed at game creation and never mutated. - For each
   player_state ps in {baku_state, hal_state}: 0 <= ps.cylinder_seconds <=
   config.cylinder.max (overflow at max triggers a death sequence that resets it
   to 0). - game_over <=> (winner <> None && loser <> None). - winner = Some p
   iff loser = Some (the other player). - If a player_state has status = Dead
   (permanent), then game_over = true and that player is the loser. *)
type t = {
  config : Config.t;
  game_clock : int;
  round_number : int;
  current_half : Domain.half_index;
  history : half_round_record list;
  first_dropper : Domain.player;
  baku_state : Player.player_state;
  hal_state : Player.player_state;
  winner : Domain.player option;
  loser : Domain.player option;
  game_over : bool;
  rng : Random.State.t;
}

(* public access functions *)
let config (game : t) : Config.t = game.config
let game_clock (game : t) : int = game.game_clock
let round_num (game : t) : int = game.round_number
let current_half (game : t) : Domain.half_index = game.current_half
let history (game : t) : half_round_record list = game.history
let winner (game : t) : Domain.player option = game.winner
let loser (game : t) : Domain.player option = game.loser
let game_over (game : t) : bool = game.game_over

(* small note we are using convention first dropper (default Hal) is always the
   first *)
let get_player_state (game : t) (player_id : Domain.player) :
    Player.player_state =
  match player_id with
  | Domain.Hal -> game.hal_state
  | Domain.Baku -> game.baku_state

let get_safe_checks (game : t) (p : Domain.player) : int =
  if p = Domain.Hal then Player.safe_checks_remaining game.config game.hal_state
  else if p = Domain.Baku then
    Player.safe_checks_remaining game.config game.baku_state
  else raise (Failure "Invalid player id")

(* clock accessors using game as arg *)
let format_game_clock (game : t) : string =
  Clock.format_wall_time (Clock.to_wall_time (config game) (game_clock game))

let get_turn_duration (game : t) : int =
  Clock.turn_duration_for_start_time game.config game.game_clock

let snap_clock_to_next_minute (game : t) : t =
  {
    game with
    game_clock = Clock.snap_to_next_round_boundary game.config game.game_clock;
  }

let validate_drop_time (dropper : Domain.player) (drop_time : int)
    (turn_time : int) : unit =
  let upper_bound =
    match dropper with
    | Domain.Baku -> turn_time
    | Domain.Hal -> min turn_time 60
  in
  if 1 <= drop_time && drop_time <= upper_bound then ()
  else raise (Failure "Invalid drop-time")

let validate_check_time (check_time : int) (turn_time : int) : unit =
  if 1 <= check_time && check_time <= min turn_time 60 then ()
  else raise (Failure "Invalid checking-time")

let is_leap_second_turn (game : t) : bool =
  Clock.in_leap_window game.config game.game_clock

let advance_clock (game : t) (time_forward : int) : t =
  { game with game_clock = Clock.advance game.game_clock time_forward }

(* default version of a game *)
let create ?(config : Config.t = Config.default ()) ?(seed : int = 42)
    ?(first_dropper : Domain.player = Domain.Hal) () : t =
  {
    config;
    game_clock = config.time.opening_start_second;
    round_number = 1;
    current_half = First;
    history = [];
    first_dropper;
    baku_state = Player.create_player config Domain.Baku;
    hal_state = Player.create_player config Domain.Hal;
    winner = None;
    loser = None;
    game_over = false;
    rng = Random.State.make [| seed |];
  }

(* Helper functions for resolution/play *)
let get_roles_for_half (game : t) : Player.player_state * Player.player_state =
  let half : Domain.half_index = game.current_half in
  match half with
  | First ->
      if game.first_dropper = Domain.Hal then (game.hal_state, game.baku_state)
      else (game.baku_state, game.hal_state)
  | Second ->
      if game.first_dropper = Domain.Hal then (game.baku_state, game.hal_state)
      else (game.hal_state, game.baku_state)

let handle_death (game : t) (checker : Player.player_state) ~(st_in_vial : int)
    ~(death_duration : int) (survived_outcome : bool option) :
    Player.player_state * bool * float =
  let survival_prob : float =
    Referee.compute_revival_prob game.config ~st_in_vial
      ~ttd_accrued:checker.ttd_seconds
  in
  let survived =
    match survived_outcome with
    | None ->
        Referee.attempt_revival game.config ~st_in_vial
          ~ttd_accrued:checker.ttd_seconds game.rng
    | Some forced -> forced
  in
  let checker_died = Player.on_death checker death_duration in
  let final_checker =
    if survived then Player.on_revival checker_died
    else Player.on_perm_death checker_died
  in
  (final_checker, survived, survival_prob)

let resolve_check (game : t) (checker : Player.player_state) (drop_time : int)
    (check_time : int) (survived_outcome : bool option) :
    t * int * (int * bool * float) option =
  if check_time >= drop_time then
    let st : int = check_time - drop_time + 1 in
    let checker', injection_result =
      Player.atc_checked game.config checker st
    in
    match injection_result with
    | No_overflow ->
        let game' : t =
          if checker.id = Domain.Baku then { game with baku_state = checker' }
          else { game with hal_state = checker' }
        in
        (game', st, None)
    | Overflow ->
        let death_duration : int = game.config.cylinder.max in
        let fatal_st =
          game.config.cylinder.max - game.config.turn.failed_check_penalty
        in
        let new_player, survived, prob =
          handle_death game checker' ~st_in_vial:fatal_st ~death_duration
            survived_outcome
        in
        let game' : t =
          if checker.id = Domain.Baku then { game with baku_state = new_player }
          else { game with hal_state = new_player }
        in
        (game', st, Some (death_duration, survived, prob))
  else
    let penalty : int = game.config.turn.failed_check_penalty in
    let st_in_vial = checker.cylinder_seconds in
    let checker_after_penalty : Player.player_state =
      Player.add_to_cylinder checker penalty
    in
    let death_duration : int =
      min checker_after_penalty.cylinder_seconds game.config.cylinder.max
    in
    let new_player, survived, prob =
      handle_death game checker_after_penalty ~st_in_vial ~death_duration
        survived_outcome
    in
    let game' : t =
      if checker.id = Domain.Baku then { game with baku_state = new_player }
      else { game with hal_state = new_player }
    in
    (game', 0, Some (death_duration, survived, prob))

let construct_HRR (game : t) (dropper : Domain.player) (checker : Domain.player)
    (drop_time : int) (check_time : int) (turn_duration : int)
    (result : half_round_result) (st_gained : int) (death_duration : int)
    (survived : bool option) (game_clock_at_start : int)
    (survival_probability : float option) : half_round_record =
  {
    round_num = game.round_number;
    half = game.current_half;
    dropper;
    checker;
    check_time;
    drop_time;
    turn_duration;
    result;
    st_gained;
    death_duration;
    survived;
    game_clock_at_start;
    survival_probability;
  }

(* half round play logic *)
let resolve_half_round (game : t) (drop_time : int) (check_time : int)
    (survived_outcome : bool option) : t * half_round_record =
  if game.game_over then raise (Failure "Game is already over")
  else
    let game = { game with rng = Random.State.copy game.rng } in
    let clock_at_start = game.game_clock in
    let turn_duration = get_turn_duration game in
    let dropper_ps, checker_ps = get_roles_for_half game in
    validate_check_time check_time turn_duration;
    validate_drop_time dropper_ps.id drop_time turn_duration;
    let game, st_gained, death_details =
      resolve_check game checker_ps drop_time check_time survived_outcome
    in
    let game = advance_clock game turn_duration in
    let check_was_successful = check_time >= drop_time in
    let game, result, death_duration, survived_field, prob_field =
      match death_details with
      | None -> (game, Check_success, 0, None, None)
      | Some (dd, true, prob) ->
          let game =
            advance_clock game
              (dd + game.config.cylinder.death_procedure_overhead)
          in
          let result =
            if check_was_successful then Cylinder_overflow_survived
            else Check_fail_survived
          in
          (game, result, dd, Some true, Some prob)
      | Some (dd, false, prob) ->
          let game =
            advance_clock game
              (dd + game.config.cylinder.death_procedure_overhead)
          in
          let game =
            {
              game with
              game_over = true;
              winner = Some dropper_ps.id;
              loser = Some checker_ps.id;
            }
          in
          let result =
            if check_was_successful then Cylinder_overflow_died
            else Check_fail_died
          in
          (game, result, dd, Some false, Some prob)
    in
    let record =
      construct_HRR game dropper_ps.id checker_ps.id drop_time check_time
        turn_duration result st_gained death_duration survived_field
        clock_at_start prob_field
    in
    let game = { game with history = record :: game.history } in
    let game =
      if game.game_over then game
      else
        match game.current_half with
        | First ->
            let game =
              advance_clock game game.config.time.within_round_overhead
            in
            { game with current_half = Second }
        | Second ->
            let game = snap_clock_to_next_minute game in
            {
              game with
              current_half = First;
              round_number = game.round_number + 1;
            }
    in
    (game, record)

let play_half_round (game : t) (drop_time : int) (check_time : int) :
    t * half_round_record =
  resolve_half_round game drop_time check_time None

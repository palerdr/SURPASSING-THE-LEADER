open Dth_engine

type chance_mode =
  | Enumerate
  | Sample

type branch = {
  probability : float;
  game : Game.t;
  record : Game.half_round_record;
}

let baku_terminal_utility (game : Game.t) : float option =
  match Game.winner game with
  | Some Domain.Baku -> Some 1.0
  | Some Domain.Hal -> Some (-1.0)
  | None -> None

let clamp_probability p = if p <= 0.0 then 0.0 else if p >= 1.0 then 1.0 else p
let singleton probability game record = [ { probability; game; record } ]

let enumerate (game : Game.t) ~(drop_time : int) ~(check_time : int) :
    branch list =
  let survived_game, survived_record =
    Game.resolve_half_round game drop_time check_time (Some true)
  in
  match survived_record.survived with
  | None -> singleton 1.0 survived_game survived_record
  | Some true -> (
      match survived_record.survival_probability with
      | None -> singleton 1.0 survived_game survived_record
      | Some raw_p ->
          let p = clamp_probability raw_p in
          if p = 1.0 then singleton 1.0 survived_game survived_record
          else
            let died_game, died_record =
              Game.resolve_half_round game drop_time check_time (Some false)
            in
            if p = 0.0 then singleton 1.0 died_game died_record
            else
              [
                {
                  probability = p;
                  game = survived_game;
                  record = survived_record;
                };
                {
                  probability = 1.0 -. p;
                  game = died_game;
                  record = died_record;
                };
              ])
  | Some false ->
      (* A forced survival branch should never report death. Keep this case
         defensive so malformed future engine behavior fails closed. *)
      singleton 1.0 survived_game survived_record

let sample (game : Game.t) ~(drop_time : int) ~(check_time : int) : branch list
    =
  let game, record = Game.play_half_round game drop_time check_time in
  singleton 1.0 game record

let resolve ?(chance_mode = Enumerate) (game : Game.t) ~(drop_time : int)
    ~(check_time : int) : branch list =
  match chance_mode with
  | Enumerate -> enumerate game ~drop_time ~check_time
  | Sample -> sample game ~drop_time ~check_time

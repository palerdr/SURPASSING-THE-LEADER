let canonical : ((int * Domain.half_index) * int) list =
  [
    ((1, Domain.First), 60);
    ((1, Domain.Second), 25);
    ((2, Domain.First), 35);
    ((2, Domain.Second), 5);
    ((3, Domain.First), 56);
    ((3, Domain.Second), 60);
    ((4, Domain.First), 7);
    ((4, Domain.Second), 60);
    ((5, Domain.First), 1);
    ((5, Domain.Second), 16);
    ((6, Domain.First), 60);
    ((6, Domain.Second), 10);
    ((7, Domain.First), 1);
    ((7, Domain.Second), 1);
    ((8, Domain.First), 1);
    ((8, Domain.Second), 1);
    ((9, Domain.First), 1);
    ((9, Domain.Second), 60);
  ]

let lookup (game : Game.t) : int option =
  List.assoc_opt (Game.round_num game, Game.current_half game) canonical

let fallback (game : Game.t) : int =
  let dropper_ps, _ = Game.get_roles_for_half game in
  if dropper_ps.id = Domain.Hal then 1
  else (Game.config game).Config.turn.duration_normal

let choose_action (game : Game.t) : int =
  match lookup game with
  | Some a -> a
  | None -> fallback game

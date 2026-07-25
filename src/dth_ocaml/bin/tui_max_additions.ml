module A = Notty.A
module I = Notty.I
module Game = Dth_engine.Game
module Domain = Dth_engine.Domain

(* reuse the same colour helpers as tui_layout *)
let dim s = I.string A.(fg (Tui_theme.accent_dim ())) s
let gold s = I.string A.(fg (Tui_theme.accent_gold ()) ++ st bold) s
let alert s = I.string A.(fg (Tui_theme.accent_alert ()) ++ st bold) s
let text s = I.string A.(fg (Tui_theme.accent_text ())) s

(* 1 -- NND PROGRESS BAR *)
let nnd_bar ~bar_w ~current ~max =
  let filled =
    if max = 0 then 0
    else
      int_of_float
        (float_of_int current /. float_of_int max *. float_of_int bar_w)
  in
  let filled = min bar_w (Int.max 0 filled) in
  let empty = bar_w - filled in
  let bar_str = String.make filled '|' ^ String.make empty ' ' in

  (* colour the bar gold when healthy, alert when low (under 20%) *)
  let bar_colour =
    if filled <= bar_w / 5 then A.(fg (Tui_theme.accent_alert ()) ++ st bold)
    else A.(fg (Tui_theme.accent_gold ()))
  in

  let format_mmss s =
    let s = Int.max 0 s in
    Printf.sprintf "%02d:%02d" (s / 60) (s mod 60)
  in

  I.hcat
    [
      dim "[";
      I.string bar_colour bar_str;
      dim "] ";
      text (Printf.sprintf "%d / %s" current (format_mmss max));
    ]

(* 1 -- GAME ARCHIVE FEED *)
let player_name = function
  | Domain.Hal -> "Hal"
  | Domain.Baku -> "Baku"

let result_label = function
  | Game.Check_success -> "Check success"
  | Game.Check_fail_survived -> "Check failed — revived"
  | Game.Check_fail_died -> "Check failed — DIED"
  | Game.Cylinder_overflow_survived -> "Cylinder overflow — revived"
  | Game.Cylinder_overflow_died -> "Cylinder overflow — DIED"

let result_colour = function
  | Game.Check_success -> A.(fg (Tui_theme.accent_gold ()))
  | Game.Check_fail_died | Game.Cylinder_overflow_died ->
      A.(fg (Tui_theme.accent_alert ()) ++ st bold)
  | _ -> A.(fg (Tui_theme.accent_text ()))

let archive_feed ~n (game : Game.t) =
  let hist = Game.history game in

  (* take the n most recent records (history is newest-first) *)
  let rec take k lst acc =
    if k = 0 then List.rev acc
    else
      match lst with
      | [] -> List.rev acc
      | x :: rest -> take (k - 1) rest (x :: acc)
  in
  let recent = take n hist [] in

  let row (r : Game.half_round_record) =
    let half_s =
      match r.half with
      | Domain.First -> "1st"
      | Domain.Second -> "2nd"
    in
    let label = result_label r.result in
    I.hcat
      [
        dim (Printf.sprintf "Turn %d (%s):  " r.round_num half_s);
        text
          (Printf.sprintf "%s dropped @%ds, %s checked @%ds  →  "
             (player_name r.dropper) r.drop_time (player_name r.checker)
             r.check_time);
        I.string (result_colour r.result) label;
        dim (Printf.sprintf "  +%ds ST" r.st_gained);
      ]
  in

  match recent with
  | [] -> I.vcat [ gold "Game Archive"; dim "(no turns yet)" ]
  | rows -> I.vcat (gold "Game Archive" :: List.map row rows)

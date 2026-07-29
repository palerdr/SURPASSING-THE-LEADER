module A = Notty.A
module I = Notty.I
module Game = Dth_engine.Game
module Domain = Dth_engine.Domain
module Player = Dth_engine.Player
module Config = Dth_engine.Config
module Clock = Dth_engine.Clock
module Referee = Dth_engine.Referee

type image = Notty.image

let format_mmss seconds =
  let s = max 0 seconds in
  Printf.sprintf "%02d:%02d" (s / 60) (s mod 60)

let player_name = function
  | Domain.Hal -> "Hal"
  | Domain.Baku -> "Baku"

let result_label = function
  | Game.Check_success -> "Check success"
  | Game.Check_fail_survived -> "Check failed -- revived"
  | Game.Check_fail_died -> "Check failed -- DIED"
  | Game.Cylinder_overflow_survived -> "Cylinder overflow -- revived"
  | Game.Cylinder_overflow_died -> "Cylinder overflow -- DIED"

let half_label = function
  | Domain.First -> "First"
  | Domain.Second -> "Second"

let palette_for_player = function
  | Domain.Hal -> Tui_theme.palette_hal ()
  | Domain.Baku -> Tui_theme.palette_baku ()

let hrule ~w =
  I.uchar A.(fg (Tui_theme.accent_dim ())) (Uchar.of_int 0x2500) w 1

let dbl_rule ~w =
  I.uchar A.(fg (Tui_theme.accent_gold ())) (Uchar.of_int 0x2550) w 1

let text ?attr s =
  let attr =
    match attr with
    | Some attr -> attr
    | None -> A.fg (Tui_theme.accent_text ())
  in
  I.string attr s

let dim s = I.string A.(fg (Tui_theme.accent_dim ())) s
let alert s = I.string A.(fg (Tui_theme.accent_alert ()) ++ st bold) s
let gold s = I.string A.(fg (Tui_theme.accent_gold ()) ++ st bold) s
let pad_left n img = if n <= 0 then img else I.( <|> ) (I.void n 1) img
let pad_top n img = if n <= 0 then img else I.( <-> ) (I.void 1 n) img

let hcenter ~w img =
  let iw = I.width img in
  if iw >= w then img
  else
    let left = (w - iw) / 2 in
    pad_left left img

let player_panel (game : Game.t) (pid : Domain.player) =
  let config = Game.config game in
  let st = Game.get_player_state game pid in
  let palette = palette_for_player pid in
  let portrait = Tui_art.portrait ~palette in
  let vial =
    Tui_art.cylinder_vial ~palette
      ~fill:
        (float_of_int st.cylinder_seconds /. float_of_int config.cylinder.max)
  in
  let header = gold (String.uppercase_ascii (player_name pid)) in
  let status_line =
    match st.status with
    | Domain.Alive -> dim "alive"
    | Domain.Dead -> alert "dead"
  in
  let cylinder_line =
    I.hcat
      [
        text "Cylinder ";
        gold (format_mmss st.cylinder_seconds);
        dim (Printf.sprintf " / %s" (format_mmss config.cylinder.max));
      ]
  in
  let death_line =
    I.hcat
      [
        text (Printf.sprintf "Deaths %d   " st.deaths);
        text (Printf.sprintf "TTD %s" (format_mmss st.ttd_seconds));
      ]
  in
  I.vcat
    [
      header;
      status_line;
      I.void 0 1;
      portrait;
      I.void 0 1;
      cylinder_line;
      vial;
      I.void 0 1;
      death_line;
    ]

let role_line (game : Game.t) =
  let dropper, checker = Game.get_roles_for_half game in
  I.hcat
    [
      text "Dropper: ";
      gold (player_name dropper.id);
      text "    Checker: ";
      gold (player_name checker.id);
    ]

let header_bar ~w (game : Game.t) =
  let title = gold "DROP THE HANDKERCHIEF" in
  let round_info =
    I.string
      A.(fg (Tui_theme.accent_text ()))
      (Printf.sprintf "Round %d / %s" (Game.round_num game)
         (half_label (Game.current_half game)))
  in
  let clock =
    I.string
      A.(fg (Tui_theme.accent_gold ()) ++ st bold)
      (Game.format_game_clock game)
  in
  let left_w = I.width title in
  let right_w = I.width clock in
  let mid_w = I.width round_info in
  let pad_a = max 2 ((w - left_w - mid_w - right_w) / 2) in
  let pad_b = max 2 (w - left_w - pad_a - mid_w - right_w) in
  I.hcat [ title; I.void pad_a 1; round_info; I.void pad_b 1; clock ]

let footer_bar ~w ~(game : Game.t) ~hint =
  let hist = Game.history game in
  let recent =
    match hist with
    | [] -> dim "History: (none yet)"
    | _ ->
        let take3 lst =
          let rec go n acc = function
            | [] -> List.rev acc
            | _ when n = 0 -> List.rev acc
            | x :: xs -> go (n - 1) (x :: acc) xs
          in
          go 3 [] lst
        in
        let summary (r : Game.half_round_record) =
          Printf.sprintf "R%d%s %s ST%ds" r.round_num
            (match r.half with
            | Domain.First -> "F"
            | Domain.Second -> "S")
            (result_label r.result) r.st_gained
        in
        let items = List.map summary (take3 hist) in
        dim ("History: " ^ String.concat "  |  " items)
  in
  let turn_info =
    let leap = if Game.is_leap_second_turn game then " [LEAP]" else "" in
    dim
      (Printf.sprintf "Turn duration %ds%s   Revival state: vial ST + TTD"
         (Game.get_turn_duration game)
         leap)
  in
  let help = dim hint in
  I.vcat [ hrule ~w; turn_info; recent; I.void 0 1; help; dbl_rule ~w ]

let frame ~w ~(game : Game.t) ~center ~hint =
  let hal_panel = player_panel game Domain.Hal in
  let baku_panel = player_panel game Domain.Baku in
  let panel_w = max (I.width hal_panel) (I.width baku_panel) in
  let gap_w = max 2 (w - (panel_w * 2) - I.width center) in
  let gap_l = gap_w / 2 in
  let gap_r = gap_w - gap_l in
  let body =
    I.hcat [ hal_panel; I.void gap_l 1; center; I.void gap_r 1; baku_panel ]
  in
  I.vcat
    [
      dbl_rule ~w;
      header_bar ~w game;
      role_line game;
      hrule ~w;
      body;
      footer_bar ~w ~game ~hint;
    ]

let splash ~term_w ~term_h =
  let w = max 48 (min 100 term_w) in
  let cloth_h = max 12 (min 24 (term_h * 2 / 3)) in
  let cloth = Tui_art.handkerchief_strip ~w:(w - 8) ~h:cloth_h in
  let banner = Tui_art.title_banner ~w in
  let tagline = dim "A Kakerou Match in OCaml -- adapted from Usogui" in
  let call_to_action = gold "Press any key to begin.  (q to quit)" in
  let content =
    I.vcat
      [
        banner;
        I.void 0 1;
        hcenter ~w tagline;
        I.void 0 1;
        hcenter ~w cloth;
        I.void 0 2;
        hcenter ~w call_to_action;
      ]
  in
  let padding_top = max 1 ((term_h - I.height content) / 2) in
  pad_top padding_top content

let choice_prompt ~term_w ~term_h ~title ~options ~entered ~hint =
  let w = max 48 (min 100 term_w) in
  let cursor = I.string A.(fg (Tui_theme.accent_alert ()) ++ st bold) "_" in
  let option_line idx label =
    I.hcat [ gold (Printf.sprintf "%d" idx); text ". "; text label ]
  in
  let choices = List.mapi (fun i label -> option_line (i + 1) label) options in
  let entered_line = I.hcat [ gold "Choice> "; text entered; cursor ] in
  let content =
    I.vcat
      ([ dbl_rule ~w; I.void 0 1; hcenter ~w (gold title); I.void 0 1 ]
      @ List.map (hcenter ~w) choices
      @ [
          I.void 0 1;
          hcenter ~w entered_line;
          I.void 0 1;
          hcenter ~w (dim hint);
          I.void 0 1;
          dbl_rule ~w;
        ])
  in
  let padding_top = max 1 ((term_h - I.height content) / 2) in
  pad_top padding_top content

let handoff ~to_name ~reason =
  let w = 60 in
  let cloth = Tui_art.handkerchief_strip ~w:(w - 4) ~h:10 in
  let content =
    I.vcat
      [
        dbl_rule ~w;
        I.void 0 1;
        hcenter ~w (gold (Printf.sprintf ">> Pass keyboard to %s <<" to_name));
        I.void 0 1;
        hcenter ~w (text reason);
        I.void 0 1;
        hcenter ~w cloth;
        I.void 0 1;
        hcenter ~w (dim "Press Enter when ready.  (q to quit)");
        I.void 0 1;
        dbl_rule ~w;
      ]
  in
  content

let pre_turn ?(hint = "Press Enter to hand keyboard to the dropper. q quits.")
    (game : Game.t) =
  let w = 100 in
  let dropper, checker = Game.get_roles_for_half game in
  let msg =
    I.vcat
      [
        I.void 0 2;
        hcenter ~w:40
          (gold (Printf.sprintf "%s drops." (player_name dropper.id)));
        I.void 0 1;
        hcenter ~w:40
          (gold (Printf.sprintf "%s checks." (player_name checker.id)));
        I.void 0 2;
        hcenter ~w:40
          (text
             (Printf.sprintf "Turn duration: %d s"
                (Game.get_turn_duration game)));
        I.void 0 1;
        hcenter ~w:40
          (if Game.is_leap_second_turn game then alert "LEAP SECOND TURN"
           else dim "normal turn");
        I.void 0 2;
        hcenter ~w:40 (dim "Press Enter to continue.");
      ]
  in
  frame ~w ~game ~center:msg ~hint

let automated_action (game : Game.t) ~actor ~role_label =
  let w = 100 in
  let center =
    I.vcat
      [
        I.void 0 2;
        hcenter ~w:44
          (gold
             (Printf.sprintf "%s -- %s"
                (String.uppercase_ascii (player_name actor))
                role_label));
        I.void 0 1;
        hcenter ~w:44 (text "Canonical Hal selected this action.");
        I.void 0 1;
        hcenter ~w:44 (dim "The value stays hidden until resolution.");
        I.void 0 2;
        hcenter ~w:44 (dim "Press Enter to continue.");
      ]
  in
  frame ~w ~game ~center ~hint:"Press Enter to continue. q quits."

let input_prompt (game : Game.t) ~actor ~role_label ~prompt ~entered ~hidden
    ~max_value =
  let w = 100 in
  let shown =
    if hidden then String.make (String.length entered) '*' else entered
  in
  let cursor = I.string A.(fg (Tui_theme.accent_alert ()) ++ st bold) "_" in
  let input_line =
    I.hcat [ gold (Printf.sprintf "%s> " role_label); text shown; cursor ]
  in
  let banner =
    I.vcat
      [
        I.void 0 2;
        hcenter ~w:40
          (gold
             (Printf.sprintf "%s -- %s"
                (String.uppercase_ascii (player_name actor))
                role_label));
        I.void 0 1;
        hcenter ~w:40 (text prompt);
        I.void 0 1;
        hcenter ~w:40 (dim (Printf.sprintf "Legal range: 1 .. %d" max_value));
        I.void 0 2;
        hcenter ~w:40 input_line;
      ]
  in
  let hint =
    if hidden then
      "Digits type, Backspace erases, Enter confirms. Entry hidden from the \
       other player. q quits."
    else "Digits type, Backspace erases, Enter confirms. q quits."
  in
  frame ~w ~game ~center:banner ~hint

let resolution (game : Game.t) (r : Game.half_round_record) =
  let w = 100 in
  let result_attr =
    match r.result with
    | Game.Check_success -> A.(fg (Tui_theme.accent_gold ()) ++ st bold)
    | Game.Cylinder_overflow_survived | Game.Check_fail_survived ->
        A.(fg (Tui_theme.accent_alert ()) ++ st bold)
    | Game.Check_fail_died | Game.Cylinder_overflow_died ->
        A.(fg (Tui_theme.accent_alert ()) ++ st bold ++ st blink)
  in
  let survival_line =
    match (r.survived, r.survival_probability) with
    | Some true, Some p ->
        dim (Printf.sprintf "Survived revival roll (p = %.1f%%)" (p *. 100.0))
    | Some false, Some p ->
        alert (Printf.sprintf "Revival FAILED (p = %.1f%%)" (p *. 100.0))
    | _ -> dim "No death this half-round"
  in
  let center =
    I.vcat
      [
        I.void 0 1;
        hcenter ~w:52 (I.string result_attr (result_label r.result));
        I.void 0 1;
        hcenter ~w:52
          (text
             (Printf.sprintf "%s dropped at %ds   %s checked at %ds"
                (player_name r.dropper) r.drop_time (player_name r.checker)
                r.check_time));
        I.void 0 1;
        hcenter ~w:52
          (text
             (Printf.sprintf "ST gained: %ds     Death duration: %ds"
                r.st_gained r.death_duration));
        I.void 0 1;
        hcenter ~w:52 survival_line;
        I.void 0 1;
        hcenter ~w:52
          (dim
             (Printf.sprintf "Clock at half start: %s"
                (Clock.format_wall_time
                   (Clock.to_wall_time (Game.config game) r.game_clock_at_start))));
        I.void 0 2;
        hcenter ~w:52 (dim "Press Enter for next half.");
      ]
  in
  frame ~w ~game ~center ~hint:"Press Enter to continue. q quits."

let ending (game : Game.t) =
  let w = 100 in
  let winner = Game.winner game in
  let loser = Game.loser game in
  let card =
    match (winner, loser) with
    | Some w, Some l ->
        Tui_art.end_card ~winner_name:(player_name w)
          ~loser_name:(player_name l)
    | _ ->
        I.vcat
          [
            alert "Match ended without a resolved winner.";
            dim "This should not happen under normal play.";
          ]
  in
  let body =
    I.vcat
      [
        I.void 0 2;
        hcenter ~w:70 card;
        I.void 0 2;
        hcenter ~w:70 (dim "Press q to exit.");
      ]
  in
  I.vcat [ dbl_rule ~w; header_bar ~w game; hrule ~w; body; dbl_rule ~w ]

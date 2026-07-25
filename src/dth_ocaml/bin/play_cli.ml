module Game = Dth_engine.Game

type game_mode =
  | Two_player
  | Single_player_vs_hal

let other_color_mode = function
  | Tui_theme.Color -> Tui_theme.Safe
  | Tui_theme.Safe -> Tui_theme.Color

let color_mode_label = function
  | Tui_theme.Color -> "Color"
  | Tui_theme.Safe -> "Safe / monochrome"

let read_choice term ~title ~options ~hint =
  let term_w, term_h = Notty_unix.Term.size term in
  Tui_input.read_int_in_range term
    ~render:(fun entered ->
      Tui_layout.choice_prompt ~term_w ~term_h ~title ~options ~entered ~hint)
    ~lo:1 ~hi:(List.length options)

let choose_display_mode term =
  let recommended = Tui_theme.suggested_color_mode () in
  let fallback = other_color_mode recommended in
  let choice =
    read_choice term ~title:"Display"
      ~options:
        [
          Printf.sprintf "%s (recommended)" (color_mode_label recommended);
          color_mode_label fallback;
        ]
      ~hint:"Type 1 or 2, then Enter. q quits."
  in
  if choice = 1 then recommended else fallback

let choose_game_mode term =
  match
    read_choice term ~title:"Game Mode"
      ~options:[ "Two Player"; "Single Player vs Hal" ]
      ~hint:"Type 1 or 2, then Enter. q quits."
  with
  | 1 -> Two_player
  | _ -> Single_player_vs_hal

let hal_controlled mode actor =
  match mode with
  | Two_player -> false
  | Single_player_vs_hal -> actor = Dth_engine.Domain.Hal

let read_human_action term game ~actor ~role_label ~prompt ~hidden ~max_value =
  Tui_input.read_int_in_range term
    ~render:(fun entered ->
      Tui_layout.input_prompt game ~actor ~role_label ~prompt ~entered ~hidden
        ~max_value)
    ~lo:1 ~hi:max_value

let read_action term mode game ~actor ~role_label ~prompt ~hidden ~max_value =
  if hal_controlled mode actor then (
    Notty_unix.Term.image term
      (Tui_layout.automated_action game ~actor ~role_label);
    Tui_input.wait_for_enter term;
    Dth_engine.Hal.choose_action game)
  else read_human_action term game ~actor ~role_label ~prompt ~hidden ~max_value

let maybe_handoff term mode ~to_name ~reason =
  match mode with
  | Two_player ->
      Notty_unix.Term.image term (Tui_layout.handoff ~to_name ~reason);
      Tui_input.wait_for_enter term
  | Single_player_vs_hal -> ()

let pre_turn_hint = function
  | Two_player -> "Press Enter to hand keyboard to the dropper. q quits."
  | Single_player_vs_hal ->
      "Press Enter to continue. Baku is human; Hal is automatic. q quits."

let play_match term mode =
  let game = ref (Game.create ()) in
  while not (Game.game_over !game) do
    let dropper, checker = Game.get_roles_for_half !game in
    let dropper_name = Tui_layout.player_name dropper.Dth_engine.Player.id in
    let checker_name = Tui_layout.player_name checker.Dth_engine.Player.id in
    let last_action actions = List.hd (List.rev actions) in
    let drop_max =
      Dth_solver.Solver_actions.legal_drop_times !game
        dropper.Dth_engine.Player.id
      |> last_action
    in
    let check_max =
      Dth_solver.Solver_actions.legal_check_times !game
        checker.Dth_engine.Player.id
      |> last_action
    in

    Notty_unix.Term.image term
      (Tui_layout.pre_turn ~hint:(pre_turn_hint mode) !game);
    Tui_input.wait_for_enter term;

    maybe_handoff term mode ~to_name:dropper_name
      ~reason:
        (Printf.sprintf
           "%s, look away. %s will commit a drop time hidden from you."
           checker_name dropper_name);

    let drop_time =
      read_action term mode !game ~actor:dropper.Dth_engine.Player.id
        ~role_label:"DROP TIME"
        ~prompt:"When does the dropper release the handkerchief?" ~hidden:false
        ~max_value:drop_max
    in

    maybe_handoff term mode ~to_name:checker_name
      ~reason:
        (Printf.sprintf
           "%s, look away. %s will commit a check time hidden from you."
           dropper_name checker_name);

    let check_time =
      read_action term mode !game ~actor:checker.Dth_engine.Player.id
        ~role_label:"CHECK TIME" ~prompt:"When does the checker turn around?"
        ~hidden:false ~max_value:check_max
    in

    let game', record = Game.play_half_round !game drop_time check_time in
    game := game';

    let hal_st = Game.get_player_state !game Dth_engine.Domain.Hal in
    let cfg = Game.config !game in
    Notty_unix.Term.image term
      (Tui_max_additions.nnd_bar ~bar_w:20
         ~current:hal_st.Dth_engine.Player.cylinder_seconds
         ~max:cfg.Dth_engine.Config.cylinder.max);
    Tui_input.wait_for_enter term;
    Notty_unix.Term.image term (Tui_max_additions.archive_feed ~n:3 !game);
    Tui_input.wait_for_enter term;

    Notty_unix.Term.image term (Tui_layout.resolution !game record);
    Tui_input.wait_for_enter term
  done;
  Notty_unix.Term.image term (Tui_layout.ending !game);
  Tui_input.wait_for_any_key term

let () =
  Tui_theme.set_color_mode (Tui_theme.suggested_color_mode ());
  let term = Notty_unix.Term.create () in
  let release () = try Notty_unix.Term.release term with _ -> () in
  Fun.protect ~finally:release (fun () ->
      try
        let w, h = Notty_unix.Term.size term in
        Notty_unix.Term.image term (Tui_layout.splash ~term_w:w ~term_h:h);
        Tui_input.wait_for_any_key term;
        Tui_theme.set_color_mode (choose_display_mode term);
        play_match term (choose_game_mode term)
      with Tui_input.Quit -> ())

(** Frame composition for the Drop The Handkerchief TUI.

    Every public function returns a complete [Notty.image] sized for the current
    terminal. The layout never writes to stdout directly; the caller hands the
    image to a [Notty_unix.Term] for rendering. *)

type image = Notty.image

(** Render a non-negative second count as ["mm:ss"]. *)
val format_mmss : int -> string

(** Human-readable name for a player ([Hal] / [Baku]). *)
val player_name : Dth_engine.Domain.player -> string

(** Short, colourable label describing a resolved half-round outcome. *)
val result_label : Dth_engine.Game.half_round_result -> string

(** Opening screen: title banner, dithered cloth, and a "press any key" hint. *)
val splash : term_w:int -> term_h:int -> image

(** Startup choice screen for mode and display options. *)
val choice_prompt :
  term_w:int ->
  term_h:int ->
  title:string ->
  options:string list ->
  entered:string ->
  hint:string ->
  image

(** Pass-the-keyboard screen blanking player state while the keyboard changes
    hands. *)
val handoff : to_name:string -> reason:string -> image

(** Scoreboard drawn before the dropper's input. Shows both players, the
    upcoming roles, the wall clock, and the round/half. *)
val pre_turn : ?hint:string -> Dth_engine.Game.t -> image

(** Informational screen shown when Hal's canonical policy chooses an action. *)
val automated_action :
  Dth_engine.Game.t ->
  actor:Dth_engine.Domain.player ->
  role_label:string ->
  image

(** Input screen for a single numeric entry. The dropper's digits are [hidden]
    so the next player cannot peek; the checker's are shown plainly because the
    dropper has already committed. *)
val input_prompt :
  Dth_engine.Game.t ->
  actor:Dth_engine.Domain.player ->
  role_label:string ->
  prompt:string ->
  entered:string ->
  hidden:bool ->
  max_value:int ->
  image

(** Reveal screen after a half-round resolves. Surfaces drop/check times,
    result, ST gained, death duration, and survival probability. *)
val resolution : Dth_engine.Game.t -> Dth_engine.Game.half_round_record -> image

(** End-of-match hero screen driven by [Game.winner] / [Game.loser]. *)
val ending : Dth_engine.Game.t -> image

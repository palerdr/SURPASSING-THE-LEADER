(** Procedurally generated dithered art.

    All images are rendered as half-block cells via [Tui_theme]. No external
    image assets are required. *)

type image = Notty.image

(** A flowing dithered "cloth" band [w] pixels wide by [h] pixels tall (yielding
    a [w x (h/2)] cell image). Used on the splash and resolution cards. *)
val handkerchief_strip : w:int -> h:int -> image

(** The DTH title with a dithered gradient backdrop, fit to [w] cells wide. *)
val title_banner : w:int -> image

(** A 10 x 15 cell vial whose fluid height is [fill] (clamped to [0, 1]) using
    the given palette. *)
val cylinder_vial : palette:Tui_theme.palette -> fill:float -> image

(** A 24 x 12 cell stylised portrait silhouette (head + shoulders + vignette).
*)
val portrait : palette:Tui_theme.palette -> image

(** End-of-match hero image with the winner's name dominant. *)
val end_card : winner_name:string -> loser_name:string -> image

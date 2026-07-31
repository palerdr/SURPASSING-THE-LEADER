(** Palette, ordered dithering, and the half-block rendering primitive.

    All dithered art in the TUI is composed of upper-half-block cells ([U+2580])
    whose foreground is the top pixel's colour and background is the bottom
    pixel's colour, doubling the vertical resolution of the terminal grid. *)

type color = Notty.A.color
type image = Notty.image
type palette = color array

type color_mode =
  | Color
  | Safe

(** Set the active terminal colour strategy. [Safe] sticks to ANSI colours for
    terminals that misrender truecolor output. *)
val set_color_mode : color_mode -> unit

(** The currently active terminal colour strategy. *)
val color_mode : unit -> color_mode

(** Conservative colour-mode default derived from terminal environment
    variables. *)
val suggested_color_mode : unit -> color_mode

(** Grayscale ramp from deep black to off-white, six stops. *)
val palette_mono : unit -> palette

(** Cool blue-gray ramp used for Hal's art. *)
val palette_hal : unit -> palette

(** Warm amber ramp used for Baku's art. *)
val palette_baku : unit -> palette

(** Primary foreground for headings and prompts. *)
val accent_text : unit -> color

(** Secondary foreground for footers and help text. *)
val accent_dim : unit -> color

(** Warning / injection red. *)
val accent_alert : unit -> color

(** Highlight used for clocks and winners. *)
val accent_gold : unit -> color

(** Global background for the frame. *)
val bg : unit -> color

(** [bayer x y] is the normalised 8x8 Bayer threshold between 0 inclusive and 1
    exclusive at the given cell. *)
val bayer : int -> int -> float

(** [dither palette ~x ~y ~v] picks a palette entry for a pixel whose intensity
    is [v] in [0, 1], using the ordered Bayer matrix at [(x, y)]. *)
val dither : palette -> x:int -> y:int -> v:float -> color

(** A single cell rendered as upper-half-block with the given top and bottom
    pixel colours. *)
val halfblock_cell : top:color -> bot:color -> image

(** [render_field palette ~w ~h f] renders a [w x h] pixel field as a
    [w x (h/2)] cell image, sampling intensity via [f x y] for each pixel and
    dithering it through [palette]. [h] should be even; if odd the last pixel
    row is ignored. *)
val render_field : palette -> w:int -> h:int -> (int -> int -> float) -> image

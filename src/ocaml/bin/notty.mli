(** A small, project-local terminal image interface used by the restored TUI.

    The original TUI was written against Notty. Native Notty currently cannot be
    built by the Windows OCaml switch because its Unix ioctl stub is not
    available there, so the executable keeps the useful image API locally and
    renders ANSI text through [Notty_unix.Term]. *)

module A : sig
  type color =
    | Basic of int
    | Rgb of int * int * int

  type style =
    | Bold
    | Blink

  type attr = {
    fg : color option;
    bg : color option;
    styles : style list;
  }

  val empty : attr
  val black : color
  val blue : color
  val cyan : color
  val lightblack : color
  val lightcyan : color
  val lightred : color
  val lightwhite : color
  val lightyellow : color
  val white : color
  val yellow : color
  val rgb_888 : r:int -> g:int -> b:int -> color
  val fg : color -> attr
  val bg : color -> attr
  val st : style -> attr
  val bold : style
  val blink : style
  val ( ++ ) : attr -> attr -> attr
end

type image

module I : sig
  val empty : image
  val string : A.attr -> string -> image
  val uchar : A.attr -> Uchar.t -> int -> int -> image
  val void : int -> int -> image
  val hcat : image list -> image
  val vcat : image list -> image
  val width : image -> int
  val height : image -> int
  val hcrop : int -> int -> image -> image
  val ( <|> ) : image -> image -> image
  val ( <-> ) : image -> image -> image
  val ( </> ) : image -> image -> image
end

(** Render an image as ANSI text for the local terminal backend. *)
val render_ansi : image -> string

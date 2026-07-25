(** Blocking input helpers built on top of [Notty_unix.Term].

    Every helper treats the uppercase or lowercase letter [q] and the Escape key
    as a quit request and raises [Quit]; callers are expected to catch it at the
    top level and release the terminal. *)

exception Quit

(** Block until any key is pressed. Resize events are swallowed. *)
val wait_for_any_key : Notty_unix.Term.t -> unit

(** Block until the user presses Enter. Non-Enter keys other than [q]/[Esc] are
    ignored. *)
val wait_for_enter : Notty_unix.Term.t -> unit

(** [read_int_in_range term ~render ~lo ~hi] reads a decimal integer in the
    inclusive range [[lo, hi]]. After every keystroke it calls [render entered]
    and hands the image to the terminal so the caller can redraw the prompt.
    Invalid Enter events (empty buffer, out-of-range, unparseable) are silently
    ignored. *)
val read_int_in_range :
  Notty_unix.Term.t -> render:(string -> Notty.image) -> lo:int -> hi:int -> int

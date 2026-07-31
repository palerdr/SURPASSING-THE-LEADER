(** Terminal input/output adapter for the project-local [Notty] image type. *)

module Term : sig
  type t

  type key =
    [ `ASCII of char
    | `Uchar of Uchar.t
    | `Escape
    | `Enter
    | `Backspace
    | `Delete
    ]

  type modifier =
    [ `Alt
    | `Ctrl
    | `Meta
    | `Shift
    ]

  type event =
    [ `End
    | `Resize of int * int
    | `Key of key * modifier list
    | `Other
    ]

  val create : unit -> t
  val release : t -> unit
  val size : t -> int * int
  val event : t -> event
  val image : t -> Notty.image -> unit
end

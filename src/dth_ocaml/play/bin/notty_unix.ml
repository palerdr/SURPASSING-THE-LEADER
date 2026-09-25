module Term = struct
  type t = {
    input : in_channel;
    output : out_channel;
    mutable size : int * int;
  }

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

  let env_int name fallback =
    match Sys.getenv_opt name with
    | Some value -> (
        match int_of_string_opt value with
        | Some n when n > 0 -> n
        | _ -> fallback)
    | None -> fallback

  let create () =
    {
      input = stdin;
      output = stdout;
      size = (env_int "COLUMNS" 100, env_int "LINES" 40);
    }

  let release _term = ()
  let size term = term.size

  let event term =
    try
      match input_char term.input with
      | '\003' as c -> `Key (`ASCII c, [ `Ctrl ])
      | '\r' | '\n' -> `Key (`Enter, [])
      | '\027' -> `Key (`Escape, [])
      | '\008' -> `Key (`Backspace, [])
      | '\127' -> `Key (`Delete, [])
      | c -> `Key (`ASCII c, [])
    with End_of_file -> `End

  let image term image =
    output_string term.output "\027[2J\027[H";
    output_string term.output (Notty.render_ansi image);
    flush term.output
end

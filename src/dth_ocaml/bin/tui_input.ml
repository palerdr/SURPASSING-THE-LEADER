exception Quit

let is_quit_char c = c = Char.code 'q' || c = Char.code 'Q' || c = 3

let key_char = function
  | `ASCII c -> Some (Char.code c)
  | `Uchar u ->
      let i = Uchar.to_int u in
      if i < 128 then Some i else None

let rec wait_for_any_key term =
  match Notty_unix.Term.event term with
  | `End -> raise Quit
  | `Resize _ -> wait_for_any_key term
  | `Key (((`ASCII _ | `Uchar _) as k), _)
    when match key_char k with
         | Some c -> is_quit_char c
         | None -> false -> raise Quit
  | `Key (`Escape, _) -> raise Quit
  | `Key _ -> ()
  | _ -> wait_for_any_key term

let rec wait_for_enter term =
  match Notty_unix.Term.event term with
  | `End -> raise Quit
  | `Resize _ -> wait_for_enter term
  | `Key (((`ASCII _ | `Uchar _) as k), _)
    when match key_char k with
         | Some c -> is_quit_char c
         | None -> false -> raise Quit
  | `Key (`Escape, _) -> raise Quit
  | `Key (`Enter, _) -> ()
  | _ -> wait_for_enter term

let read_int_in_range term ~render ~lo ~hi =
  let redraw entered = Notty_unix.Term.image term (render entered) in
  let rec loop entered =
    redraw entered;
    match Notty_unix.Term.event term with
    | `End -> raise Quit
    | `Resize _ -> loop entered
    | `Key (`Escape, _) -> raise Quit
    | `Key (((`ASCII _ | `Uchar _) as k), _) -> (
        match key_char k with
        | None -> loop entered
        | Some c ->
            if is_quit_char c then raise Quit
            else if c >= Char.code '0' && c <= Char.code '9' then
              let next = entered ^ String.make 1 (Char.chr c) in
              if String.length next > 4 then loop entered else loop next
            else loop entered)
    | `Key (`Backspace, _) ->
        let n = String.length entered in
        if n = 0 then loop entered else loop (String.sub entered 0 (n - 1))
    | `Key (`Delete, _) ->
        let n = String.length entered in
        if n = 0 then loop entered else loop (String.sub entered 0 (n - 1))
    | `Key (`Enter, _) -> (
        if entered = "" then loop entered
        else
          match int_of_string_opt entered with
          | Some v when v >= lo && v <= hi -> v
          | _ -> loop entered)
    | _ -> loop entered
  in
  loop ""

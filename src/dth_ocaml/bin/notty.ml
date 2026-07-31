module A = struct
  type color =
    | Basic of int
    | Rgb of int * int * int

  type style =
    | Bold
    | Blink

  let bold = Bold
  let blink = Blink

  type attr = {
    fg : color option;
    bg : color option;
    styles : style list;
  }

  let empty = { fg = None; bg = None; styles = [] }
  let black = Basic 0
  let blue = Basic 4
  let cyan = Basic 6
  let lightblack = Basic 8
  let lightcyan = Basic 14
  let lightred = Basic 9
  let lightwhite = Basic 15
  let lightyellow = Basic 11
  let white = Basic 7
  let yellow = Basic 3
  let clamp_channel n = max 0 (min 255 n)
  let rgb_888 ~r ~g ~b = Rgb (clamp_channel r, clamp_channel g, clamp_channel b)
  let fg color = { empty with fg = Some color }
  let bg color = { empty with bg = Some color }
  let st style = { empty with styles = [ style ] }

  let ( ++ ) left right =
    {
      fg =
        (match right.fg with
        | Some _ -> right.fg
        | None -> left.fg);
      bg =
        (match right.bg with
        | Some _ -> right.bg
        | None -> left.bg);
      styles = left.styles @ right.styles;
    }
end

type cell = {
  text : string;
  attr : A.attr;
  opaque : bool;
}

type image = {
  width : int;
  height : int;
  rows : cell array array;
}

let transparent = { text = ""; attr = A.empty; opaque = false }
let blank attr = { text = " "; attr; opaque = true }

let make_image width height f =
  let width = max 0 width in
  let height = max 0 height in
  {
    width;
    height;
    rows = Array.init height (fun y -> Array.init width (fun x -> f x y));
  }

let copy_into ~dst ~src ~x0 ~y0 =
  for y = 0 to src.height - 1 do
    for x = 0 to src.width - 1 do
      let dx = x0 + x in
      let dy = y0 + y in
      if dx >= 0 && dx < dst.width && dy >= 0 && dy < dst.height then
        dst.rows.(dy).(dx) <- src.rows.(y).(x)
    done
  done

let utf8_next s i =
  let len = String.length s in
  if i >= len then len
  else
    let b = Char.code s.[i] in
    if b land 0x80 = 0 then i + 1
    else if b land 0xe0 = 0xc0 then min len (i + 2)
    else if b land 0xf0 = 0xe0 then min len (i + 3)
    else if b land 0xf8 = 0xf0 then min len (i + 4)
    else i + 1

let string_cells attr text =
  let rec loop i acc =
    if i >= String.length text then Array.of_list (List.rev acc)
    else
      let next = utf8_next text i in
      let glyph = String.sub text i (next - i) in
      loop next ({ text = glyph; attr; opaque = true } :: acc)
  in
  loop 0 []

let image_of_cells cells =
  { width = Array.length cells; height = 1; rows = [| cells |] }

module I = struct
  let empty = { width = 0; height = 0; rows = [||] }
  let string attr text = image_of_cells (string_cells attr text)

  let uchar attr uchar width height =
    let code = Uchar.to_int uchar in
    let glyph =
      if code <= 0x7f then String.make 1 (Char.chr code)
      else if code <= 0x7ff then
        String.init 2 (function
          | 0 -> Char.chr (0xc0 lor (code lsr 6))
          | _ -> Char.chr (0x80 lor (code land 0x3f)))
      else if code <= 0xffff then
        String.init 3 (function
          | 0 -> Char.chr (0xe0 lor (code lsr 12))
          | 1 -> Char.chr (0x80 lor ((code lsr 6) land 0x3f))
          | _ -> Char.chr (0x80 lor (code land 0x3f)))
      else
        String.init 4 (function
          | 0 -> Char.chr (0xf0 lor (code lsr 18))
          | 1 -> Char.chr (0x80 lor ((code lsr 12) land 0x3f))
          | 2 -> Char.chr (0x80 lor ((code lsr 6) land 0x3f))
          | _ -> Char.chr (0x80 lor (code land 0x3f)))
    in
    make_image width height (fun _ _ -> { text = glyph; attr; opaque = true })

  let void width height = make_image width height (fun _ _ -> transparent)

  let hcat images =
    let width = List.fold_left (fun total img -> total + img.width) 0 images in
    let height =
      List.fold_left (fun current img -> max current img.height) 0 images
    in
    let result = make_image width height (fun _ _ -> transparent) in
    let x = ref 0 in
    List.iter
      (fun img ->
        copy_into ~dst:result ~src:img ~x0:!x ~y0:0;
        x := !x + img.width)
      images;
    result

  let vcat images =
    let width =
      List.fold_left (fun current img -> max current img.width) 0 images
    in
    let height =
      List.fold_left (fun total img -> total + img.height) 0 images
    in
    let result = make_image width height (fun _ _ -> transparent) in
    let y = ref 0 in
    List.iter
      (fun img ->
        copy_into ~dst:result ~src:img ~x0:0 ~y0:!y;
        y := !y + img.height)
      images;
    result

  let width img = img.width
  let height img = img.height

  let hcrop left width img =
    let left = max 0 left in
    let width = max 0 width in
    make_image width img.height (fun x y ->
        let source_x = left + x in
        if source_x < img.width then img.rows.(y).(source_x) else transparent)

  let ( <|> ) left right = hcat [ left; right ]
  let ( <-> ) top bottom = vcat [ top; bottom ]

  let ( </> ) top bottom =
    let width = max top.width bottom.width in
    let height = max top.height bottom.height in
    make_image width height (fun x y ->
        let top_cell =
          if x < top.width && y < top.height then Some top.rows.(y).(x)
          else None
        in
        match top_cell with
        | Some cell when cell.opaque -> cell
        | _ ->
            if x < bottom.width && y < bottom.height then bottom.rows.(y).(x)
            else transparent)
end

let ansi_color ~background = function
  | A.Basic n ->
      let base = if n >= 8 then 90 else 30 in
      let code = base + (n mod 8) in
      Printf.sprintf "%d" (if background then code + 10 else code)
  | A.Rgb (r, g, b) ->
      Printf.sprintf "%d;2;%d;%d;%d" (if background then 48 else 38) r g b

let ansi_attr attr =
  let codes = ref [ "0" ] in
  (match attr.A.fg with
  | Some color -> codes := !codes @ [ ansi_color ~background:false color ]
  | None -> ());
  (match attr.A.bg with
  | Some color -> codes := !codes @ [ ansi_color ~background:true color ]
  | None -> ());
  List.iter
    (fun style ->
      codes :=
        !codes
        @ [
            (match style with
            | A.Bold -> "1"
            | A.Blink -> "5");
          ])
    attr.A.styles;
  "\027[" ^ String.concat ";" !codes ^ "m"

let render_ansi img =
  let buffer = Buffer.create (max 16 (img.width * max 1 img.height * 2)) in
  Array.iter
    (fun row ->
      Array.iter
        (fun cell ->
          if cell.opaque then (
            Buffer.add_string buffer (ansi_attr cell.attr);
            Buffer.add_string buffer cell.text)
          else Buffer.add_char buffer ' ')
        row;
      Buffer.add_string buffer "\027[0m\n")
    img.rows;
  Buffer.contents buffer

module A = Notty.A
module I = Notty.I

type color = Notty.A.color
type image = Notty.image
type palette = color array

type color_mode =
  | Color
  | Safe

let current_mode = ref Color
let set_color_mode mode = current_mode := mode
let color_mode () = !current_mode

let string_contains haystack needle =
  let h_len = String.length haystack in
  let n_len = String.length needle in
  let rec loop i =
    if i + n_len > h_len then false
    else if String.sub haystack i n_len = needle then true
    else loop (i + 1)
  in
  n_len = 0 || loop 0

let suggested_color_mode () =
  match (Sys.getenv_opt "TERM", Sys.getenv_opt "COLORTERM") with
  | None, _ | Some "dumb", _ -> Safe
  | _, Some c ->
      let c = String.lowercase_ascii c in
      if string_contains c "truecolor" || string_contains c "24bit" then Color
      else Safe
  | _ -> Safe

let rgb ~r ~g ~b = A.rgb_888 ~r ~g ~b

let color_palette_mono =
  [|
    rgb ~r:10 ~g:10 ~b:12;
    rgb ~r:30 ~g:30 ~b:34;
    rgb ~r:60 ~g:60 ~b:66;
    rgb ~r:105 ~g:105 ~b:112;
    rgb ~r:165 ~g:165 ~b:170;
    rgb ~r:230 ~g:230 ~b:232;
  |]

let safe_palette_mono =
  [| A.black; A.black; A.lightblack; A.white; A.lightwhite; A.lightwhite |]

let color_palette_hal =
  [|
    rgb ~r:8 ~g:10 ~b:16;
    rgb ~r:20 ~g:28 ~b:44;
    rgb ~r:36 ~g:54 ~b:82;
    rgb ~r:70 ~g:100 ~b:140;
    rgb ~r:130 ~g:170 ~b:210;
    rgb ~r:215 ~g:230 ~b:245;
  |]

let safe_palette_hal =
  [| A.black; A.black; A.blue; A.cyan; A.lightcyan; A.lightwhite |]

let color_palette_baku =
  [|
    rgb ~r:12 ~g:8 ~b:6;
    rgb ~r:44 ~g:22 ~b:14;
    rgb ~r:90 ~g:44 ~b:20;
    rgb ~r:160 ~g:90 ~b:34;
    rgb ~r:215 ~g:150 ~b:60;
    rgb ~r:245 ~g:225 ~b:190;
  |]

let safe_palette_baku =
  [| A.black; A.black; A.yellow; A.lightyellow; A.white; A.lightwhite |]

let palette_mono () =
  match !current_mode with
  | Color -> color_palette_mono
  | Safe -> safe_palette_mono

let palette_hal () =
  match !current_mode with
  | Color -> color_palette_hal
  | Safe -> safe_palette_hal

let palette_baku () =
  match !current_mode with
  | Color -> color_palette_baku
  | Safe -> safe_palette_baku

let accent_text () =
  match !current_mode with
  | Color -> rgb ~r:232 ~g:228 ~b:215
  | Safe -> A.lightwhite

let accent_dim () =
  match !current_mode with
  | Color -> rgb ~r:120 ~g:118 ~b:110
  | Safe -> A.white

let accent_alert () =
  match !current_mode with
  | Color -> rgb ~r:200 ~g:60 ~b:54
  | Safe -> A.lightred

let accent_gold () =
  match !current_mode with
  | Color -> rgb ~r:225 ~g:180 ~b:80
  | Safe -> A.lightyellow

let bg () =
  match !current_mode with
  | Color -> color_palette_mono.(0)
  | Safe -> A.black

(* Normalised 8x8 Bayer matrix. Values in [0, 63], divided by 64 below. *)
let bayer_matrix =
  [|
    [| 0; 32; 8; 40; 2; 34; 10; 42 |];
    [| 48; 16; 56; 24; 50; 18; 58; 26 |];
    [| 12; 44; 4; 36; 14; 46; 6; 38 |];
    [| 60; 28; 52; 20; 62; 30; 54; 22 |];
    [| 3; 35; 11; 43; 1; 33; 9; 41 |];
    [| 51; 19; 59; 27; 49; 17; 57; 25 |];
    [| 15; 47; 7; 39; 13; 45; 5; 37 |];
    [| 63; 31; 55; 23; 61; 29; 53; 21 |];
  |]

let bayer x y =
  let i = ((x mod 8) + 8) mod 8 in
  let j = ((y mod 8) + 8) mod 8 in
  (float_of_int bayer_matrix.(j).(i) +. 0.5) /. 64.0

let dither palette ~x ~y ~v =
  let n = Array.length palette in
  let v = if v < 0.0 then 0.0 else if v > 1.0 then 1.0 else v in
  let scaled = v *. float_of_int (n - 1) in
  let lo = int_of_float scaled in
  let lo = if lo < 0 then 0 else if lo > n - 1 then n - 1 else lo in
  let hi = if lo + 1 > n - 1 then n - 1 else lo + 1 in
  let frac = scaled -. float_of_int lo in
  if frac > bayer x y then palette.(hi) else palette.(lo)

let halfblock_cell ~top ~bot = I.string A.(fg top ++ A.bg bot) "\xe2\x96\x80"
(* U+2580 UPPER HALF BLOCK in UTF-8 *)

let render_field palette ~w ~h f =
  let rows = h / 2 in
  let row_image cy =
    let cells =
      Array.init w (fun cx ->
          let py_top = cy * 2 in
          let py_bot = (cy * 2) + 1 in
          let top = dither palette ~x:cx ~y:py_top ~v:(f cx py_top) in
          let bot = dither palette ~x:cx ~y:py_bot ~v:(f cx py_bot) in
          halfblock_cell ~top ~bot)
    in
    Array.fold_left I.( <|> ) I.empty cells
  in
  let rows_images = Array.init rows row_image in
  Array.fold_left I.( <-> ) I.empty rows_images

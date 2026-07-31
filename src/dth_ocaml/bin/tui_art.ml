module A = Notty.A
module I = Notty.I

type image = Notty.image

let clamp01 v = if v < 0.0 then 0.0 else if v > 1.0 then 1.0 else v

let handkerchief_strip ~w ~h =
  let sample x y =
    let fx = float_of_int x in
    let fy = float_of_int y in
    let fh = float_of_int h in
    (* A gentle double-sine fold, darker toward the edges. *)
    let fold =
      0.5
      +. (0.30 *. sin ((fx /. 9.0) +. (fy /. 5.0)))
      +. (0.18 *. sin ((fx /. 3.5) -. (fy /. 7.0)))
    in
    let edge_falloff = 1.0 -. (abs_float ((fy /. fh) -. 0.5) *. 1.6) in
    clamp01 (fold *. clamp01 edge_falloff)
  in
  Tui_theme.render_field (Tui_theme.palette_mono ()) ~w ~h sample

let title_banner ~w =
  let text = "DROP  THE  HANDKERCHIEF" in
  let text_len = String.length text in
  let pad_total = max 0 (w - text_len) in
  let pad_l = pad_total / 2 in
  let pad_r = pad_total - pad_l in
  let backdrop =
    handkerchief_strip ~w:(max 40 (w * 2)) ~h:8 |> fun img ->
    I.hcrop 0 (max 0 (I.width img - w)) img
  in
  let title_line = I.string A.(fg (Tui_theme.accent_gold ()) ++ st bold) text in
  let padded_title =
    I.hcat
      [
        I.string A.(fg (Tui_theme.accent_dim ())) (String.make pad_l ' ');
        title_line;
        I.string A.(fg (Tui_theme.accent_dim ())) (String.make pad_r ' ');
      ]
  in
  I.( </> ) padded_title backdrop

let cylinder_vial ~palette ~fill =
  let w = 10 in
  let h = 30 in
  let fill = clamp01 fill in
  let fill_h = int_of_float ((float_of_int h *. fill) +. 0.5) in
  let fluid_top = h - fill_h in
  let sample x y =
    let is_rim = x = 0 || x = w - 1 in
    if is_rim then 0.35
    else if y < fluid_top then
      (* empty glass: faint reflection pattern *)
      0.10 +. (0.05 *. (0.5 +. (0.5 *. sin (float_of_int y /. 3.0))))
    else
      (* fluid: bright at surface, dims toward bottom *)
      let dist_from_surface =
        if fill_h = 0 then 0.0
        else float_of_int (y - fluid_top) /. float_of_int fill_h
      in
      clamp01 (0.95 -. (dist_from_surface *. 0.45))
  in
  Tui_theme.render_field palette ~w ~h sample

let portrait ~palette =
  let w = 24 in
  let h = 24 in
  let cx = 11.5 in
  let cy_head = 8.5 in
  let r_head = 6.0 in
  let sample x y =
    let fx = float_of_int x in
    let fy = float_of_int y in
    let dx = fx -. cx in
    let dy_head = fy -. cy_head in
    let d_head = sqrt ((dx *. dx) +. (dy_head *. dy_head)) in
    let head =
      if d_head <= r_head then 0.85 -. (d_head /. r_head *. 0.15) else 0.0
    in
    (* Shoulders: widening trapezoid below the head. *)
    let shoulder_top = 16.0 in
    let shoulders =
      if fy >= shoulder_top then
        let half_w = 4.0 +. ((fy -. shoulder_top) *. 1.2) in
        if abs_float dx <= half_w then
          0.7 -. ((fy -. shoulder_top) /. 10.0 *. 0.2)
        else 0.0
      else 0.0
    in
    (* Radial vignette around the subject. *)
    let d_center = sqrt ((dx *. dx) +. (((fy -. 12.0) *. 0.8) ** 2.0)) in
    let vignette = clamp01 (0.35 -. (d_center /. 20.0)) in
    clamp01 (max (max head shoulders) vignette)
  in
  Tui_theme.render_field palette ~w ~h sample

let end_card ~winner_name ~loser_name =
  let header =
    I.string
      A.(fg (Tui_theme.accent_gold ()) ++ st bold)
      (Printf.sprintf "  %s SURVIVES  " (String.uppercase_ascii winner_name))
  in
  let sub =
    I.string
      A.(fg (Tui_theme.accent_alert ()))
      (Printf.sprintf "  %s has fallen  " (String.uppercase_ascii loser_name))
  in
  let hero = handkerchief_strip ~w:48 ~h:12 in
  I.vcat [ header; sub; I.void 0 1; hero ]

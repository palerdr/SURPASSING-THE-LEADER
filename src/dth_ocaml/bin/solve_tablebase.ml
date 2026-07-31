let default_output_path = "outputs/ocaml_value_table.bin"

let output_path_from_arguments () =
  match Array.length Sys.argv with
  | 1 -> default_output_path
  | 2 -> Sys.argv.(1)
  | _ -> invalid_arg "usage: dth-solve-tablebase [output-path]"

let ensure_parent_directory path =
  let parent = Filename.dirname path in
  if parent <> "." && not (Sys.file_exists parent) then Unix.mkdir parent 0o755

let () =
  let output_path = output_path_from_arguments () in
  ensure_parent_directory output_path;
  Printf.eprintf "Solving the packed DTH tablebase...\n%!";
  Dth_solver.Exact.solve_dth ();
  Dth_solver.Exact.write_value_table output_path;
  Printf.printf "Wrote %s\n%!" output_path

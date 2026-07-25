let () =
  let baselines = [ 0.75; 0.80; 0.85 ] in
  Printf.printf "Configured %d preliminary revival-baseline sweep points.\n"
    (List.length baselines)

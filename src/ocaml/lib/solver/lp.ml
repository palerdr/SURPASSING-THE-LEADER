type matrix = float array array

(* The package's public module is also named [Lp]. Because this project now has
   its own [Lp] module, use the package's internal alias here. *)
module Backend = struct
  module Poly = Lp__.Poly
  module Cnstr = Lp__.Cnstr
  module Objective = Lp__.Objective
  module Problem = Lp__.Problem

  let c = Poly.c
  let var = Poly.var
  let expand = Poly.expand
  let concat = Poly.concat
  let eq = Cnstr.eq
  let lt = Cnstr.lt
  let maximize = Objective.maximize
  let make = Problem.make
  let validate = Problem.validate
end

let solve (matrix : matrix) : float =
  if
    Array.length matrix <> 60
    || Array.exists (fun row -> Array.length row <> 60) matrix
  then invalid_arg "payoff matrix must be 60x60";
  let probability =
    Array.init 60 (fun row ->
        Backend.var ~lb:0.0 ~ub:1.0 (Printf.sprintf "drop_%d" (row + 1)))
  in
  let guaranteed_value = Backend.var ~lb:(-1.0) ~ub:1.0 "guaranteed_value" in
  let expected_payoff column =
    Array.init 60 (fun row ->
        Backend.expand (Backend.c matrix.(row).(column)) probability.(row))
    |> Backend.concat
  in
  let probability_sum =
    Backend.eq (Backend.concat probability) (Backend.c 1.0)
  in
  let guarantees =
    Array.to_list
      (Array.init 60 (fun column ->
           Backend.lt
             ~name:(Printf.sprintf "checker_%d" (column + 1))
             guaranteed_value (expected_payoff column)))
  in
  let problem =
    Backend.make ~name:"dropper_maximin"
      (Backend.maximize guaranteed_value)
      (probability_sum :: guarantees)
  in
  if not (Backend.validate problem) then invalid_arg "invalid matrix-game LP";
  match Lp_highs.solve ~msg:false problem with
  | Ok (objective, _) -> objective
  | Error message -> failwith ("HiGHS failed: " ^ message)

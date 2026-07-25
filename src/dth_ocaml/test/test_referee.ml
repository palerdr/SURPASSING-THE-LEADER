open Dth_engine.Referee

let config = Dth_engine.Config.default ()
let float_eps = Alcotest.float 0.000001

let probability st ttd =
  compute_revival_prob config ~st_in_vial:st ~ttd_accrued:ttd

let test_fresh_empty_vial_is_eighty_percent () =
  Alcotest.(check float_eps) "P(0,0)" 0.80 (probability 0 0)

let test_st_is_linear_at_fresh_ttd () =
  Alcotest.(check float_eps) "P(60,0)" 0.60 (probability 60 0);
  Alcotest.(check float_eps) "P(120,0)" 0.40 (probability 120 0);
  Alcotest.(check float_eps) "P(180,0)" 0.20 (probability 180 0)

let test_current_dose_boundary_is_fatal () =
  Alcotest.(check float_eps) "240 ST plus 60 dose" 0.0 (probability 240 0)

let test_strict_cumulative_boundary () =
  Alcotest.(check bool)
    "exactly 300 cumulative seconds remains eligible" true
    (probability 0 240 > 0.0);
  Alcotest.(check float_eps)
    "more than 300 cumulative seconds is fatal" 0.0 (probability 0 241)

let test_ttd_reduces_same_vial_state () =
  Alcotest.(check bool)
    "TTD weakens current dose potency" true
    (probability 60 120 < probability 60 60
    && probability 60 60 < probability 60 0)

let test_identity_is_not_an_input () =
  let first = probability 25 60 in
  let second = probability 25 60 in
  Alcotest.(check float_eps) "same physical state, same probability" first second

let test_attempt_revival_obeys_fatal_guard () =
  let rng = Random.State.make [| 42 |] in
  Alcotest.(check bool)
    "fatal state cannot revive" false
    (attempt_revival config ~st_in_vial:240 ~ttd_accrued:0 rng)

let tests =
  [
    Alcotest.test_case "fresh empty vial starts at 80%" `Quick
      test_fresh_empty_vial_is_eighty_percent;
    Alcotest.test_case "fresh TTD is linear in ST" `Quick
      test_st_is_linear_at_fresh_ttd;
    Alcotest.test_case "current dose boundary is fatal" `Quick
      test_current_dose_boundary_is_fatal;
    Alcotest.test_case "cumulative boundary is strict" `Quick
      test_strict_cumulative_boundary;
    Alcotest.test_case "TTD is monotonically harsher" `Quick
      test_ttd_reduces_same_vial_state;
    Alcotest.test_case "identity is not a state variable" `Quick
      test_identity_is_not_an_input;
    Alcotest.test_case "fatal guard dominates sampling" `Quick
      test_attempt_revival_obeys_fatal_guard;
  ]

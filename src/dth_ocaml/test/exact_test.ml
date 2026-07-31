module E = Dth_solver.Exact
module M = Dth_solver.Matrix_game

let fail name detail = failwith (name ^ ": " ^ detail)
let check name condition = if not condition then fail name "condition failed"

let approx ?(tolerance = 1e-9) expected actual =
  abs_float (expected -. actual) <= tolerance

let check_approx name expected actual = check name (approx expected actual)

let expect_exception name exception_ thunk =
  try
    thunk ();
    fail name "expected exception was not raised"
  with
  | raised when raised = exception_ -> ()
  | _ -> fail name "wrong exception was raised"

let branch_mass (branches : E.transition_distribution) =
  Array.fold_left
    (fun total (branch : E.branch) -> total +. branch.probability)
    0.0 branches

let only_branch name branches =
  if Array.length branches <> 1 then fail name "expected one branch";
  branches.(0)

let expect_live name expected_state (branch : E.branch) =
  match branch.outcome with
  | E.Live actual -> check name (actual = expected_state)
  | E.Terminal _ -> fail name "expected a live branch"

let expect_terminal name expected_value (branch : E.branch) =
  match branch.outcome with
  | E.Terminal actual -> check_approx name expected_value actual
  | E.Live _ -> fail name "expected a terminal branch"

let test_revival_surface () =
  check_approx "revival at origin" 0.95 (E.revival_probability 0 0);
  check_approx "dose boundary is ineligible" 0.0 (E.revival_probability 240 0);
  check "total load 300 remains eligible" (E.revival_probability 1 239 > 0.0);
  check_approx "total load above 300 is fatal" 0.0 (E.revival_probability 1 240);
  check "positive ttd reduces revival"
    (E.revival_probability 0 60 < E.revival_probability 0 0)

let test_action_primitives () =
  check "inclusive squandered time" (E.squandered_time (10, 20) = 11);
  check "successful check" (E.is_check_success (10, 20));
  check "failed check" (not (E.is_check_success (20, 10)));
  check "overflow boundary" (E.is_overflow 299 1);
  check "below overflow" (not (E.is_overflow 298 1));
  expect_exception "drop action zero" E.Invalid_Joint_Action (fun () ->
      E.validate_join_action (0, 1));
  expect_exception "check action above cap" E.Invalid_Joint_Action (fun () ->
      E.validate_join_action (1, 61))

(* States are (checker_load, checker_ttd, dropper_load, dropper_ttd). *)

let test_successful_transition () =
  (* Checker (10, 20), Dropper (30, 40); drop 10, check 20 squanders 11. *)
  let branches = E.successful_check_branches (10, 20, 30, 40) (10, 20) in
  check_approx "successful branch mass" 1.0 (branch_mass branches);
  let branch = only_branch "successful transition" branches in
  check_approx "successful branch probability" 1.0 branch.probability;
  (* The seats swap: the old Dropper checks next, and the old Checker drops
     carrying its new load of 10 + 11. *)
  expect_live "successful successor" (30, 40, 21, 20) branch

let test_failed_transition () =
  (* Checker (0, 0) takes the injection, so revival is the surface maximum. *)
  let branches = E.failed_check_branches (0, 0, 10, 20) (20, 10) in
  check_approx "failed branch mass" 1.0 (branch_mass branches);
  check "failed transition has revival and death branches"
    (Array.length branches = 2);
  let live, dead = (branches.(0), branches.(1)) in
  check_approx "revival branch probability" 0.95 live.probability;
  (* The revived Checker drops next at load 0, carrying the 60-second dose. *)
  expect_live "revival successor" (10, 20, 0, 60) live;
  check_approx "death branch probability" 0.05 dead.probability;
  expect_terminal "death payoff" 1.0 dead;
  let fatal = E.failed_check_branches (240, 0, 10, 20) (20, 10) in
  expect_terminal "ineligible failed check" 1.0 (only_branch "fatal" fatal)

let test_payoff_matrix_and_lp () =
  (* Checker load 299: every successful check overflows the cylinder and every
     failed one is an ineligible dose, so the whole matrix is a Dropper win. *)
  let terminal_state = (299, 0, 10, 20) in
  let matrix = E.payoff_matrix terminal_state in
  check "matrix has 60 rows" (Array.length matrix = 60);
  check "matrix has 60 columns"
    (Array.for_all (fun row -> Array.length row = 60) matrix);
  check "terminal matrix is all winning payoffs"
    (Array.for_all (Array.for_all (fun payoff -> approx 1.0 payoff)) matrix);
  check_approx "terminal state value" 1.0 (E.value terminal_state);
  let matching_identity = Array.make_matrix 60 60 (-1.0) in
  for index = 0 to 59 do
    matching_identity.(index).(index) <- 1.0
  done;
  check_approx "nontrivial LP value" (-58.0 /. 60.0) (M.solve matching_identity);
  let rock_paper_scissors =
    Array.init 60 (fun row ->
        Array.init 60 (fun column ->
            match (row - column + 3) mod 3 with
            | 0 -> 0.0
            | 1 -> 1.0
            | _ -> -1.0))
  in
  check_approx "cyclic game value" 0.0 (M.solve rock_paper_scissors);
  expect_exception "invalid LP matrix"
    (Invalid_argument "payoff matrix must be 60x60") (fun () ->
      M.solve (Array.make_matrix 59 60 0.0));
  let invalid_payoffs = Array.make_matrix 60 60 0.0 in
  invalid_payoffs.(0).(0) <- 2.0;
  expect_exception "payoff outside game range"
    (Invalid_argument "matrix payoffs must be finite values in [-1, 1]")
    (fun () -> M.solve invalid_payoffs)

(* Both games above are symmetric and solved by the uniform mixture, so they
   pass even for a solver that stops at the wrong vertex. These do not: their
   values are fixed by antisymmetry and by a saddle point, and each needs a
   pivot the previous hand-rolled tableau could not make. *)
let test_asymmetric_known_values () =
  let raw i j =
    (float_of_int (((i * 37) + (j * 11)) mod 101) /. 101.0) -. 0.5
  in
  let skew_symmetric =
    Array.init 60 (fun i ->
        Array.init 60 (fun j -> (raw i j -. raw j i) /. 2.0))
  in
  (* A = -A^T means either player can steal the other's mixture, so the value is
     exactly zero. *)
  check_approx "skew-symmetric game is fair" 0.0 (M.solve skew_symmetric);
  (* [[1; 0]; [0; -1]] has a pure saddle at (row 0, column 1) worth 0. The
     padding rows are strictly dominated and the padding columns hand the
     Dropper 1, so the padded game keeps that value. *)
  let padded_saddle =
    Array.init 60 (fun i ->
        Array.init 60 (fun j ->
            if i >= 2 then -1.0
            else if j >= 2 then 1.0
            else if i = 0 && j = 0 then 1.0
            else if i = 1 && j = 1 then -1.0
            else 0.0))
  in
  check_approx "padded saddle point" 0.0 (M.solve padded_saddle)

(* A matrix with the shape Exact.payoff_matrix actually produces: a constant
   failed-check block below the diagonal, and a value that depends only on
   inclusive squandered time on and above it. *)
let transition_class_matrix failed successes =
  Array.init 60 (fun drop ->
      Array.init 60 (fun check ->
          if check >= drop then successes.(check - drop + 1) else failed))

let test_certificate () =
  let successes =
    Array.init 61 (fun st -> cos (float_of_int st /. 7.0) *. 0.9)
  in
  let matrix = transition_class_matrix (-0.25) successes in
  let solution = M.solve_certified matrix in
  check "saddle gap within the gate"
    (solution.M.saddle_gap <= M.saddle_gap_tolerance);
  let mass policy = Array.fold_left ( +. ) 0.0 policy in
  check_approx "dropper policy is a distribution" 1.0 (mass solution.M.dropper);
  check_approx "checker policy is a distribution" 1.0 (mass solution.M.checker);
  check "policies are nonnegative"
    (Array.for_all (fun p -> p >= 0.0) solution.M.dropper
    && Array.for_all (fun p -> p >= 0.0) solution.M.checker);
  (* The certified value must be bracketed by what each policy achieves
     unilaterally, which is what makes it a certificate rather than a report. *)
  let lower =
    Array.fold_left min infinity
      (Array.init 60 (fun check ->
           let total = ref 0.0 in
           for drop = 0 to 59 do
             total :=
               !total +. (solution.M.dropper.(drop) *. matrix.(drop).(check))
           done;
           !total))
  in
  let upper =
    Array.fold_left max neg_infinity
      (Array.init 60 (fun drop ->
           let total = ref 0.0 in
           for check = 0 to 59 do
             total :=
               !total +. (solution.M.checker.(check) *. matrix.(drop).(check))
           done;
           !total))
  in
  check "value is bracketed by both policies"
    (lower -. 1e-9 <= solution.M.value && solution.M.value <= upper +. 1e-9)

(* Cross-implementation parity. These are complete-game Dropper-relative values
   read from the Python authority's certified artifact (backup_full_v1, every
   class certified to a 1e-6 saddle gap). The states are deep enough in the
   endgame that the direct recursion closes quickly, and they are the check
   that this project agrees with the peer solver rather than merely with
   itself -- in particular that a live child's value is negated across the
   seat swap. *)
let test_python_authority_parity () =
  let expect name state expected =
    check name (approx ~tolerance:1e-9 expected (E.value state))
  in
  expect "parity (299,0,299,0)" (299, 0, 299, 0) 1.0;
  expect "parity (299,0,0,0)" (299, 0, 0, 0) 1.0;
  expect "parity (290,0,299,0)" (290, 0, 299, 0) 0.714285714285714;
  expect "parity (295,0,290,0)" (295, 0, 290, 0) 0.885678349388892;
  expect "parity (280,0,285,0)" (280, 0, 285, 0) 0.579496748151480

let () =
  test_revival_surface ();
  test_action_primitives ();
  test_successful_transition ();
  test_failed_transition ();
  test_payoff_matrix_and_lp ();
  test_asymmetric_known_values ();
  test_certificate ();
  test_python_authority_parity ()

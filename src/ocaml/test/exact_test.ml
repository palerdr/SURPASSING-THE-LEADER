module E = Dth_solver.Exact
module L = Dth_solver.Lp

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

let test_successful_transition () =
  let branches = E.successful_check_branches (10, 20, 30, 40) (10, 20) in
  check_approx "successful branch mass" 1.0 (branch_mass branches);
  let branch = only_branch "successful transition" branches in
  check_approx "successful branch probability" 1.0 branch.probability;
  expect_live "successful successor" (41, 40, 10, 20) branch

let test_failed_transition () =
  let branches = E.failed_check_branches (10, 20, 0, 0) (20, 10) in
  check_approx "failed branch mass" 1.0 (branch_mass branches);
  check "failed transition has revival and death branches"
    (Array.length branches = 2);
  let live, dead = (branches.(0), branches.(1)) in
  check_approx "revival branch probability" 0.95 live.probability;
  expect_live "revival successor" (0, 60, 10, 20) live;
  check_approx "death branch probability" 0.05 dead.probability;
  expect_terminal "death payoff" 1.0 dead;
  let fatal = E.failed_check_branches (10, 20, 240, 0) (20, 10) in
  expect_terminal "ineligible failed check" 1.0 (only_branch "fatal" fatal)

let test_payoff_matrix_and_lp () =
  let terminal_state = (10, 20, 299, 0) in
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
  check_approx "nontrivial LP value" (-58.0 /. 60.0) (L.solve matching_identity);
  expect_exception "invalid LP matrix"
    (Invalid_argument "payoff matrix must be 60x60") (fun () ->
      L.solve (Array.make_matrix 59 60 0.0))

let () =
  test_revival_surface ();
  test_action_primitives ();
  test_successful_transition ();
  test_failed_transition ();
  test_payoff_matrix_and_lp ()

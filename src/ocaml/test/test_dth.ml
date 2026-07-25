let () =
  Alcotest.run "dth"
    [
      ("clock", Test_clock.tests);
      ("config", Test_config.tests);
      ("player", Test_player.tests);
      ("referee", Test_referee.tests);
      ("game", Test_game.tests);
      ("hal", Test_hal.tests);
      ("solver_actions", Test_solver_actions.tests);
      ("solver_policy", Test_solver_policy.tests);
      ("solver_transition", Test_solver_transition.tests);
    ]

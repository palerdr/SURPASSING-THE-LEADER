type outcome =
  | Terminal of float
  | Live of State.t

type branch = {
  probability : float;
  outcome : outcome;
}

let revival_probability st ttd =
  let dose = st + 60 in
  if dose >= 300 || ttd + dose > 300 then 0.0
  else
    let st_factor = 1.0 -. (float_of_int st /. 240.0) in
    let ttd_factor = 2.0 ** -.((float_of_int ttd /. 120.0) ** 1.3) in
    let referee_factor = max 0.4 (0.88 ** (float_of_int ttd /. 60.0)) in
    0.8 *. st_factor *. ttd_factor *. referee_factor

let transition (state : State.t) drop check =
  if drop < 1 || drop > 60 || check < 1 || check > 60 then
    invalid_arg "actions must be in 1..60";
  if check >= drop then
    let checker_st = state.checker_st + check - drop + 1 in
    if checker_st >= 300 then [ { probability = 1.0; outcome = Terminal 1.0 } ]
    else
      [
        {
          probability = 1.0;
          outcome =
            Live
              {
                dropper_st = checker_st;
                dropper_ttd = state.checker_ttd;
                checker_st = state.dropper_st;
                checker_ttd = state.dropper_ttd;
              };
        };
      ]
  else
    let dose = state.checker_st + 60 in
    let p = revival_probability state.checker_st state.checker_ttd in
    let death = { probability = 1.0 -. p; outcome = Terminal 1.0 } in
    if p = 0.0 then [ death ]
    else
      [
        {
          probability = p;
          outcome =
            Live
              {
                dropper_st = 0;
                dropper_ttd = state.checker_ttd + dose;
                checker_st = state.dropper_st;
                checker_ttd = state.dropper_ttd;
              };
        };
        death;
      ]

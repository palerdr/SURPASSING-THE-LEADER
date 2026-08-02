(* Drop The Handkerchief — unified revival model.

   The revival probability is deliberately a function of exactly two game-state
   variables:

   - [st_in_vial]: ST accumulated before the failed-check 60-second dose; -
   [ttd_accrued]: total time dead before the current injection.

   CPR count and identity-specific physicality are not state inputs. The referee
   burden is represented by a TTD-derived factor, so equivalent physical states
   receive equivalent odds regardless of player identity or history encoding. *)

type rng = Random.State.t

let clamp_probability probability = min 1.0 (max 0.0 probability)

let compute_revival_prob (config : Config.t) ~(st_in_vial : int)
    ~(ttd_accrued : int) : float =
  if st_in_vial < 0 then invalid_arg "st_in_vial must be nonnegative"
  else if ttd_accrued < 0 then invalid_arg "ttd_accrued must be nonnegative"
  else
    let injected_dose = st_in_vial + config.turn.failed_check_penalty in
    if
      injected_dose >= config.cylinder.max
      || ttd_accrued + injected_dose > config.cylinder.max
    then 0.0
    else
      let survival = config.survival in
      let survivable_st_span =
        config.cylinder.max - config.turn.failed_check_penalty
      in
      let st_factor =
        1.0 -. (float_of_int st_in_vial /. float_of_int survivable_st_span)
      in
      let death_minutes =
        float_of_int ttd_accrued
        /. float_of_int config.turn.failed_check_penalty
      in
      let ttd_factor = survival.ttd_decay_per_minute ** death_minutes in
      clamp_probability
        (survival.baseline *. st_factor *. ttd_factor)

let attempt_revival (config : Config.t) ~(st_in_vial : int) ~(ttd_accrued : int)
    (rng : rng) : bool =
  let revival_prob = compute_revival_prob config ~st_in_vial ~ttd_accrued in
  Random.State.float rng 1.0 < revival_prob

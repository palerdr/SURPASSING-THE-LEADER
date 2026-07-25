open Dth_engine

(** Solver-facing policies and timing-action distributions.

    This module is the boundary between game rules and solver behavior. The
    engine says which concrete timing actions can be resolved; this module says
    how a baseline, search routine, tabular strategy, or future policy network
    represents a mixed strategy over those legal actions.

    Public distributions are sparse lists because the legal action set is
    state-dependent: normal turns have 60 seconds, and only Baku as Dropper in
    the leap window may use second 61. Dense arrays are still a reasonable
    internal representation for regrets, matrices, or neural logits. *)

(** Raised when a distribution is empty, has illegal actions, contains invalid
    probabilities, or cannot be normalized. *)
exception Invalid_distribution of string

(** One entry in a sparse mixed strategy. *)
type action_prob = {
  action : Solver_actions.timing;
      (** Concrete timing action, measured in seconds from the start of the
          current half-round. *)
  prob : float;  (** Probability mass assigned to [action]. *)
}

(** Sparse probability distribution over timing actions.

    A valid distribution has at least one entry, no duplicate actions, only
    legal actions for the queried state/player/role, finite nonnegative
    probabilities, and total mass approximately equal to 1. *)
type distribution = action_prob list

(** A policy maps the current game state, acting player, and current role to a
    mixed strategy over that player's legal timing actions. *)
type t = Game.t -> Domain.player -> Domain.role -> distribution

(** [legal_actions game player role] returns the frozen legal timing actions for
    [player] in [role]. *)
val legal_actions :
  Game.t ->
  Domain.player ->
  Domain.role ->
  Solver_actions.timing list

(** [validate game player role distribution] raises
    {!Invalid_distribution} unless [distribution] is a normalized distribution
    over actions legal for [(game, player, role)]. *)
val validate :
  Game.t ->
  Domain.player ->
  Domain.role ->
  distribution ->
  unit

(** [normalize distribution] rescales nonnegative probability masses so they sum
    to 1. Tiny negative values are rejected, not silently repaired. *)
val normalize : distribution -> distribution

(** [normalize_for game player role distribution] normalizes
    [distribution] and then validates the result against the legal action set
    for [(game, player, role)]. *)
val normalize_for :
  Game.t ->
  Domain.player ->
  Domain.role ->
  distribution ->
  distribution

(** [uniform game player role] is the uniform mixed strategy over all
    legal timing actions. *)
val uniform :
  Game.t ->
  Domain.player ->
  Domain.role ->
  distribution

(** [deterministic action] gives all mass to [action] without checking whether
    it is legal in any particular state. Use {!deterministic_legal} when the
    state is available. *)
val deterministic : Solver_actions.timing -> distribution

(** [deterministic_legal game player role action] gives all mass to
    [action] and validates that the action is legal for the current decision. *)
val deterministic_legal :
  Game.t ->
  Domain.player ->
  Domain.role ->
  Solver_actions.timing ->
  distribution

(** [sample rng distribution] draws one timing action from [distribution]. The
    distribution is normalized before sampling, so callers may pass unnormalized
    positive weights. *)
val sample : Random.State.t -> distribution -> Solver_actions.timing

(** [instant] deterministically chooses the first legal second. *)
val instant : t

(** [safe] chooses the first legal second for droppers and the last legal second
    for checkers. This is a deliberately simple baseline, not an equilibrium
    claim. *)
val safe : t

(** [uniform_policy] is {!uniform} packaged as a policy value. *)
val uniform_policy : t

(** [canonical_hal] uses {!Dth_engine.Hal.choose_action} when [player] is Hal
    and falls back to uniform play for Baku. If the scripted Hal action is not
    legal under the frozen action rules, the nearest conservative fallback used
    here is the last legal action. *)
val canonical_hal : t

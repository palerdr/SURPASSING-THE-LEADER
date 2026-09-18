// Mirrors arena/web/schema.py. Kept by hand; a Python test asserts field names,
// types, nullability, and requiredness so drift fails before browser runtime.

export type Phase = "rules" | "awaiting_action" | "awaiting_ack" | "game_over";
export type OutcomeResult =
  | "check_success"
  | "check_fail_survived"
  | "check_fail_died"
  | "overflow_survived"
  | "overflow_died";

export interface PlayerView {
  name: string;
  character: "hal" | "baku";
  role: "dropper" | "checker";
  cylinder_seconds: number;
  ttd_seconds: number;
  deaths: number;
  is_human: boolean;
}

export interface OutcomeView {
  round: number;
  half: number;
  dropper: string;
  checker: string;
  drop_time: number;
  check_time: number;
  result: OutcomeResult;
  st_gained: number;
  death_duration: number;
  survived: boolean | null;
  survival_probability: number | null;
  game_over: boolean;
  session_ending: boolean;
  winner_name: string | null;
}

export interface Snapshot {
  sequence: number;
  phase: Phase;
  game_index: number;
  pure_dth: boolean;
  human_name: string;
  clock_display: string;
  clock_seconds: number;
  round: number;
  half: number;
  turn_duration: number;
  leap_window: boolean;
  dropper_name: string;
  checker_name: string;
  human_role: "dropper" | "checker";
  legal_seconds: number[];
  players: PlayerView[];
  cylinder_max: number;
  ttd_max: number;
  half_rounds: number;
  last_outcome: OutcomeView | null;
  winner_name: string | null;
  winner_is_human: boolean | null;
  stopped: boolean;
}

export interface Rules {
  human_name: string;
  hal_label: string;
  pure_dth: boolean;
  lines: string[];
}

/** One resolved half-round as the public transcript records it. */
export interface HistoryEntry {
  public_state_before: {
    clock_seconds: number;
    clock_display: string;
    round: number;
    half: number;
    turn_duration: number;
  };
  dropper: string;
  checker: string;
  drop_second: number;
  check_second: number;
  result: OutcomeResult;
  squandered_seconds: number;
  death_duration_seconds: number;
  survived: boolean | null;
  survival_probability: number | null;
}

export interface FinishedGame {
  game_index: number;
  seed: number | null;
  start_clock: number;
  winner: string | null;
  stopped: boolean;
  half_rounds: number;
  public_history: HistoryEntry[];
}

export interface Tally {
  human_wins: number;
  hal_wins: number;
  no_winner: number;
  stopped: number;
}

/** GET /api/transcript: the CLI's play-session transcript plus the live game. */
export interface Transcript {
  schema_version: string;
  hal_agent: string;
  public_hal_label: string | null;
  human_name: string;
  base_seed: number | null;
  start_clock: number;
  pure_dth: boolean;
  games: FinishedGame[];
  tally: Tally;
  current_game: {
    game_index: number;
    seed: number | null;
    start_clock: number;
    phase: Phase;
    half_rounds: number;
    public_history: HistoryEntry[];
  };
  hal_summary?: string;
}

/** Fields a new game may override; everything else is fixed by the server. */
export interface NewGameOptions {
  human_name?: string;
  seed?: number;
  start_clock?: number;
  max_half_rounds?: number;
}

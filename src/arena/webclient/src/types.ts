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
  lines: string[];
}

export const RESULT_TEXT: Record<string, string> = {
  check_success: "Check successful",
  check_fail_survived: "Check failed — revived",
  check_fail_died: "Check failed — died",
  overflow_survived: "Cylinder overflow — revived",
  overflow_died: "Cylinder overflow — died",
};

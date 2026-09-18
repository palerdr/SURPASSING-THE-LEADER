// Player-facing wording, kept close to the terminal interface so a player
// moving between the two surfaces reads the same game. Each line opens with
// a capital, since every line stands alone on the result screen. Pure
// functions only; every number here arrives from the server-owned referee.

import type { OutcomeResult, OutcomeView, Tally } from "../types";

/**
 * The terminal's `_RESULT_TEXT` in src/arena/tui.py, with a comma where the
 * terminal sets a dash: "CHECK FAILED, died, revived" reads cleaner.
 */
export const RESULT_TEXT: Record<OutcomeResult, string> = {
  check_success: "CHECK SUCCESS",
  check_fail_survived: "CHECK FAILED, died, revived",
  check_fail_died: "CHECK FAILED, died permanently",
  overflow_survived: "ST OVERFLOW, died, revived",
  overflow_died: "ST OVERFLOW, died permanently",
};

export function resultText(result: OutcomeResult): string {
  return RESULT_TEXT[result] ?? result;
}

/** "1–60", or the explicit list when the legal set is not contiguous. */
export function legalRange(legal: readonly number[]): string {
  const first = legal[0];
  const last = legal[legal.length - 1];
  if (first === undefined || last === undefined) return "none";
  const contiguous = legal.length === last - first + 1;
  return contiguous ? `${first}–${last}` : legal.join(", ");
}

/** The squandered time, just the number: `Squandered time 10s into <checker>'s ST`. */
export function squanderedTimeLine(outcome: Pick<OutcomeView, "st_gained" | "checker">): string {
  return `Squandered time ${outcome.st_gained.toFixed(0)}s into ${outcome.checker}'s ST`;
}

/**
 * The death block: the injected dose only. The revival chance and the
 * roll's verdict are not shown; the result line above already says died
 * or revived.
 */
export function deathLines(outcome: Pick<OutcomeView, "death_duration">): string[] {
  if (outcome.death_duration <= 0) return [];
  return [`Injected dose ${outcome.death_duration.toFixed(0)}s`];
}

/** "3–1" style series score from the human's point of view. */
export function tallyText(tally: Tally): string {
  const parts = [`${tally.human_wins}–${tally.hal_wins}`];
  const extra = tally.no_winner + tally.stopped;
  if (extra > 0) parts.push(`${extra} undecided`);
  return parts.join(" · ");
}

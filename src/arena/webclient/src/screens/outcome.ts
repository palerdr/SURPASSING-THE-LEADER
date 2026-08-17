import { RESULT_TEXT, type OutcomeView } from "../types";
import { escapeHtml } from "../render/escape";

/**
 * The reveal.
 *
 * Both seconds are shown here and nowhere else — they are secret until the
 * half-round resolves. All derived values come from the server-owned referee.
 */
export function renderOutcome(
  panel: HTMLElement,
  outcome: OutcomeView,
  onContinue: () => void,
): void {
  const success = outcome.result === "check_success";
  const lines: string[] = [
    `<p><strong>${escapeHtml(outcome.dropper)}</strong> dropped at <strong>${outcome.drop_time}</strong>` +
      ` · <strong>${escapeHtml(outcome.checker)}</strong> checked at <strong>${outcome.check_time}</strong></p>`,
    `<p class="${success ? "win" : "lose"}">${escapeHtml(RESULT_TEXT[outcome.result] ?? outcome.result)}</p>`,
  ];

  if (success) {
    lines.push(
      `<p class="arith">Squandered time: ${outcome.st_gained.toFixed(0)}s</p>`,
    );
  }
  if (outcome.death_duration > 0) {
    lines.push(
      `<p class="arith">Death procedure cost ${outcome.death_duration.toFixed(0)}s of TTD</p>`,
    );
  }
  if (outcome.survival_probability !== null) {
    const chance = (outcome.survival_probability * 100).toFixed(1);
    const verdict = outcome.survived === null ? "" : outcome.survived ? " — revived" : " — not revived";
    lines.push(`<p class="arith">Revival chance ${chance}%${verdict}</p>`);
  }

  panel.innerHTML = `
    <h2>HALF-ROUND RESOLVED</h2>
    ${lines.join("")}
    <form><button type="submit">${outcome.session_ending ? "See the result" : "Continue"}</button></form>
    <div class="error"></div>`;

  const button = panel.querySelector("button");
  button?.focus();
  panel.querySelector("form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    onContinue();
  });
}

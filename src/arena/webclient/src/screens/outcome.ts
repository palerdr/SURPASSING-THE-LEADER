import { RESULT_TEXT, type OutcomeView } from "../types";

/**
 * The reveal.
 *
 * Both seconds are shown here and nowhere else — they are secret until the
 * half-round resolves. The arithmetic is recomputed from the numbers rather
 * than sent as prose, so the server ships state and not sentences.
 */
export function renderOutcome(
  panel: HTMLElement,
  outcome: OutcomeView,
  onContinue: () => void,
): void {
  const success = outcome.result === "check_success";
  const lines: string[] = [
    `<p><strong>${outcome.dropper}</strong> dropped at <strong>${outcome.drop_time}</strong>` +
      ` · <strong>${outcome.checker}</strong> checked at <strong>${outcome.check_time}</strong></p>`,
    `<p class="${success ? "win" : "lose"}">${RESULT_TEXT[outcome.result] ?? outcome.result}</p>`,
  ];

  if (success) {
    // Inclusive elapsed time: ST = check - drop + 1.
    lines.push(
      `<p class="arith">ST = ${outcome.check_time} − ${outcome.drop_time} + 1 =` +
        ` ${outcome.st_gained.toFixed(0)}s squandered</p>`,
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
    <form><button type="submit">${outcome.game_over ? "See the result" : "Continue"}</button></form>
    <div class="error"></div>`;

  const button = panel.querySelector("button");
  button?.focus();
  panel.querySelector("form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    onContinue();
  });
}

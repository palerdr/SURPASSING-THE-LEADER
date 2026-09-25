import type { OutcomeView } from "../types";
import { escapeHtml } from "../render/escape";
import { deathLines, resultText, squanderedTimeLine } from "../render/text";

/**
 * The decision screen, worded after the terminal's `render_outcome`, with
 * commas for its dashes and the squandered time as a number rather than
 * the arithmetic that produced it. The round, half, and clock stay in the
 * top-right corner and are not repeated here; the revival chance is not
 * shown, since the result line already says whether the checker came back.
 *
 * Both seconds are shown here and nowhere else — they are secret until the
 * half-round resolves. All derived values come from the server-owned referee.
 */
export function renderOutcome(
  screen: HTMLElement,
  outcome: OutcomeView,
  onContinue: () => void,
): void {
  const [verdict, ...aftermath] = resultText(outcome.result).split(", ");
  const verdictHtml = escapeHtml(verdict).replace("FAILED", '<span class="lose">FAILED</span>');
  const lines: string[] = [
    `<p><strong>${escapeHtml(outcome.dropper)}</strong> dropped at second <strong>${outcome.drop_time}</strong></p>`,
    `<p><strong>${escapeHtml(outcome.checker)}</strong> checked at second <strong>${outcome.check_time}</strong></p>`,
  ];
  if (outcome.st_gained > 0) {
    lines.push(`<p>${escapeHtml(squanderedTimeLine(outcome))}</p>`);
  }
  for (const line of deathLines(outcome)) {
    lines.push(`<p>${escapeHtml(line)}</p>`);
  }
  if (outcome.game_over && outcome.winner_name !== null) {
    lines.push(`<p>GAME OVER, ${escapeHtml(outcome.winner_name)} wins.</p>`);
  } else if (outcome.game_over) {
    lines.push(`<p>GAME OVER, no surviving winner.</p>`);
  } else if (outcome.session_ending) {
    lines.push(`<p>The half-round cap has been reached.</p>`);
  }

  screen.innerHTML = `
    <div class="card outcome">
      <h2 class="verdict">${verdictHtml}${aftermath.length ? `<span class="aftermath">${escapeHtml(aftermath.join(", "))}</span>` : ""}</h2>
      <div class="outcome-details">
        ${lines.join("")}
      </div>
      <form>
        <button type="submit">${outcome.session_ending ? "See the result" : "Continue"}</button>
      </form>
      <div class="error"></div>
    </div>`;

  screen.querySelector("form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    onContinue();
  });
}

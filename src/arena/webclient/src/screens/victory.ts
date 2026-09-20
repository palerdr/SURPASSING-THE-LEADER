import type { Snapshot, Transcript } from "../types";
import { escapeHtml } from "../render/escape";
import { humanWon } from "../render/identity";
import { tallyText } from "../render/text";

/**
 * The end-of-game screen over the winner's still. A server with a leaderboard
 * puts that screen between this one and the next game.
 */
export function renderVictory(
  screen: HTMLElement,
  snapshot: Snapshot,
  transcript: Transcript | null,
  onNext: () => void,
  nextLabel = "Next game",
): void {
  let headline: string;
  if (snapshot.stopped) {
    headline = `<p class="big hint">SESSION STOPPED</p><p class="hint">No winner after ${snapshot.half_rounds} half-rounds.</p>`;
  } else if (snapshot.winner_name === null) {
    headline = `<p class="big lose">NO WINNER</p>`;
  } else {
    const won = humanWon(snapshot);
    headline = `<p class="big ${won ? "win" : "lose"}">${escapeHtml(snapshot.winner_name.toUpperCase())} WINS</p>`;
  }
  // A hosted game stands alone, so a series line appears only once one exists.
  const series = transcript && transcript.games.length > 1
    ? `<p class="hint">Series so far: ${escapeHtml(tallyText(transcript.tally))} over ${transcript.games.length} game${transcript.games.length === 1 ? "" : "s"}.</p>`
    : "";
  const summary =
    transcript && transcript.hal_summary
      ? `<div class="summary">Hal: ${escapeHtml(transcript.hal_summary)}</div>`
      : "";

  screen.innerHTML = `
    <div class="card victory" style="align-self:flex-end">
      ${headline}
      ${series}
      <form>
        <button type="submit">${escapeHtml(nextLabel)}</button>
      </form>
      <div class="error"></div>
      ${summary}
    </div>`;

  screen.querySelector("button")?.focus();
  screen.querySelector("form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    onNext();
  });
}

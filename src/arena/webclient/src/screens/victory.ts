import type { Snapshot, Transcript } from "../types";
import { escapeHtml } from "../render/escape";
import { humanWon } from "../render/identity";
import { tallyText } from "../render/text";

/**
 * The end-of-game screen over the winner's still. Hal keeps its opponent
 * model into the next game, exactly as `arena play --games N` retains one
 * provider across a series, so the button reads "Next game". A server with a
 * leaderboard puts that screen between this one and the next game.
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
    headline = `<p class="big lose">NO WINNER</p><p class="lose">Game over. No surviving winner.</p>`;
  } else {
    const won = humanWon(snapshot);
    headline =
      `<p class="big ${won ? "win" : "lose"}">${escapeHtml(snapshot.winner_name.toUpperCase())} WINS</p>` +
      `<p class="${won ? "win" : "lose"}">${escapeHtml(snapshot.winner_name)} wins the match after ${snapshot.half_rounds} half-rounds.</p>`;
  }
  const series = transcript
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
      <p class="hint">Hal remembers you between games</p>
      <div class="error"></div>
      ${summary}
    </div>`;

  screen.querySelector("button")?.focus();
  screen.querySelector("form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    onNext();
  });
}

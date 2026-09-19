import type { Leaderboard } from "../types";
import { escapeHtml } from "../render/escape";
import { scoreText, standingText } from "../render/text";

/** The name field's limit; the server enforces the same one. */
const NAME_LENGTH = 16;

/**
 * The standings after a finished game. The server ranks each player's latest
 * game, so this screen only lists what it is given. A winner with no posted
 * name gets the name field; the score is already in the ledger either way.
 */
export function renderLeaderboard(
  screen: HTMLElement,
  board: Leaderboard | null,
  onName: (name: string) => void,
  onNext: () => void,
): void {
  const rows = (board?.entries ?? [])
    .map(
      (entry) => `
        <li${entry.is_you ? ' class="you"' : ""}>
          <span>${entry.rank}</span>
          <bdi class="name">${escapeHtml(entry.name)}</bdi>
          <span class="hint">${entry.half_rounds} half-rounds</span>
          <span class="score">${escapeHtml(scoreText(entry.score))}</span>
        </li>`,
    )
    .join("");
  const standings = board === null
    ? `<p class="hint">Loading the standings</p>`
    : rows
      ? `<ol class="standings">${rows}</ol>`
      : `<p class="hint">No player holds a place yet.</p>`;
  const asksName = board !== null && board.your_score !== null && board.your_name === null;
  const nameForm = asksName
    ? `<form data-name>
        <input type="text" maxlength="${NAME_LENGTH}" required autocomplete="off" aria-label="Your name" placeholder="Your name" />
        <button type="submit">Post score</button>
      </form>`
    : "";

  screen.innerHTML = `
    <div class="card board">
      <h2>LEADERBOARD</h2>
      ${standings}
      ${board === null ? "" : `<p${board.your_score === null ? ' class="hint"' : ""}>${escapeHtml(standingText(board))}</p>`}
      ${nameForm}
      <form data-next><button type="submit">Next game</button></form>
      <div class="error" role="alert"></div>
    </div>`;

  const field = screen.querySelector<HTMLInputElement>("[data-name] input");
  (field ?? screen.querySelector<HTMLButtonElement>("[data-next] button"))?.focus();
  screen.querySelector("[data-name]")?.addEventListener("submit", (event) => {
    event.preventDefault();
    const name = field?.value.trim() ?? "";
    if (name) onName(name);
  });
  screen.querySelector("[data-next]")?.addEventListener("submit", (event) => {
    event.preventDefault();
    onNext();
  });
}

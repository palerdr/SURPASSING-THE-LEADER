import type { Snapshot } from "../types";
import { escapeHtml } from "../render/escape";
import { humanWon } from "../render/identity";

export function renderVictory(
  panel: HTMLElement,
  snapshot: Snapshot,
  onRestart: () => void,
): void {
  let headline: string;
  if (snapshot.stopped) {
    // The half-round cap ended the session; nobody won.
    headline = `<p class="hint">Session stopped after ${snapshot.half_rounds} half-rounds. No winner.</p>`;
  } else if (snapshot.winner_name === null) {
    headline = `<p class="lose">Game over. No surviving winner.</p>`;
  } else {
    const won = humanWon(snapshot);
    headline = `<p class="${won ? "win" : "lose"}">${escapeHtml(snapshot.winner_name)} wins.</p>`;
  }

  panel.innerHTML = `
    <h2>GAME OVER</h2>
    ${headline}
    <p class="hint">${snapshot.half_rounds} half-rounds played.</p>
    <form><button type="submit">New game</button></form>
    <div class="error"></div>`;

  panel.querySelector("button")?.focus();
  panel.querySelector("form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    onRestart();
  });
}

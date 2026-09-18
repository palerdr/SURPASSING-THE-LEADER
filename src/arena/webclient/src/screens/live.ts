import type { Snapshot } from "../types";
import { dialplateMarkup } from "../render/dialplate";
import { legalRange } from "../render/text";

/**
 * The action screen: one commit and the dial plate. There is no field; the
 * count beneath the plate names the second now passing, and Commit or Enter
 * plays that second. `shown` is the count's value when the screen is drawn;
 * main.ts advances it on each beat and reads it back at the commit.
 *
 * Legal seconds arrive from the server; this screen never derives them. Only
 * the engine knows that Baku as Dropper may play 61 inside the leap window.
 */
export function renderLive(
  screen: HTMLElement,
  snapshot: Snapshot,
  shown: number,
  onSubmit: () => void,
): void {
  const legal = snapshot.legal_seconds;
  screen.innerHTML = `
    <div class="card action">
      <div class="ask">
        <h2>YOU ARE THE ${snapshot.human_role.toUpperCase()}</h2>
        <p>The clock counts the seconds ${legalRange(legal)}.</p>
        <p>Press Commit or Enter on your second.</p>
        <button type="button" data-commit>Commit</button>
        <div class="error"></div>
      </div>
      <div class="plate-and-count">
        ${dialplateMarkup(snapshot)}
        <div class="count" data-count aria-live="off">${shown}</div>
      </div>
    </div>`;

  const button = screen.querySelector<HTMLButtonElement>("[data-commit]");
  button?.focus();
  button?.addEventListener("click", (event) => {
    event.preventDefault();
    onSubmit();
  });
}

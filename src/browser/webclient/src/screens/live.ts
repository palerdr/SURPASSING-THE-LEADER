import type { Snapshot } from "../types";
import { dialplateMarkup } from "../render/dialplate";
import { legalRange } from "../render/text";

/**
 * The action screen: one field, one commit, and the dial plate. A second
 * typed in the field plays at once. With the field empty, Commit or Enter
 * plays the second the count beneath the plate names. `shown` is the count's
 * value when the screen is drawn; main.ts advances it on each beat and reads
 * the field and the clock at the commit.
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
        <p>Type a second from ${legalRange(legal)} and commit it now.</p>
        <p>Leave the box empty to commit the second on the clock.</p>
        <form class="pick" novalidate>
          <input type="text" data-second inputmode="numeric" maxlength="2"
                 autocomplete="off" aria-label="Your second" />
          <button type="submit" data-commit>Commit</button>
        </form>
        <div class="error" role="alert"></div>
      </div>
      <div class="plate-and-count">
        ${dialplateMarkup(snapshot)}
        <div class="count" data-count aria-live="off">${shown}</div>
      </div>
    </div>`;

  // A phone opens its keyboard over the clock when the field takes focus, so
  // there the player taps the field to type.
  const typing = window.matchMedia("(pointer: fine)").matches;
  screen.querySelector<HTMLElement>(typing ? "[data-second]" : "[data-commit]")?.focus();
  screen.querySelector("form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    onSubmit();
  });
}

import type { Snapshot } from "../types";
import { escapeHtml } from "../render/escape";

/**
 * The establishing shot before each decision: the three figures on the
 * canvas beneath, and one caption saying who drops and who checks. It holds
 * until the player clicks or presses Enter to open the action screen.
 */
export function renderBeat(screen: HTMLElement, snapshot: Snapshot): void {
  const human = snapshot.human_role === "dropper" ? "You drop" : "You check";
  screen.innerHTML = `
    <div class="caption">
      ${escapeHtml(snapshot.dropper_name.toUpperCase())} DROPS · ${escapeHtml(snapshot.checker_name.toUpperCase())} CHECKS
      <small>${escapeHtml(human)} this half · Round ${snapshot.round}, Half ${snapshot.half}</small>
      <button type="button">Continue</button>
    </div>`;
}

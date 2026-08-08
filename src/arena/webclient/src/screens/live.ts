import type { Snapshot } from "../types";

/**
 * The action prompt.
 *
 * Legal seconds arrive from the server; this screen never derives them. Only
 * the engine knows that Baku as Dropper may play 61 inside the leap window.
 */
export function renderLive(
  panel: HTMLElement,
  snapshot: Snapshot,
  onSubmit: (second: number) => void,
): void {
  const legal = snapshot.legal_seconds;
  const first = legal[0];
  const last = legal[legal.length - 1];
  const contiguous = first !== undefined && last !== undefined && legal.length === last - first + 1;
  const allowed = contiguous ? `${first}–${last}` : legal.join(", ");
  const verb =
    snapshot.human_role === "dropper"
      ? "Choose the second to drop the handkerchief."
      : "Choose the second to check.";

  panel.innerHTML = `
    <h2>YOU ARE THE ${snapshot.human_role.toUpperCase()}</h2>
    <p>${verb}</p>
    <p class="hint">Legal seconds: ${allowed}. Hal decides only after you commit.</p>
    <form>
      <input type="number" name="second" min="${first ?? 1}" max="${last ?? 60}"
             step="1" required autocomplete="off" autofocus />
      <button type="submit">Commit</button>
    </form>
    <div class="error"></div>`;

  const form = panel.querySelector("form");
  const input = panel.querySelector<HTMLInputElement>('input[name="second"]');
  const error = panel.querySelector<HTMLElement>(".error");
  input?.focus();

  form?.addEventListener("submit", (event) => {
    event.preventDefault();
    const second = Number(input?.value);
    if (!Number.isInteger(second) || !legal.includes(second)) {
      if (error) error.textContent = `${input?.value || "that"} is not a legal second.`;
      return;
    }
    if (error) error.textContent = "";
    onSubmit(second);
  });
}

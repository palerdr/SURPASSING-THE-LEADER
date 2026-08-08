import type { Rules, Snapshot } from "../types";

export function renderRules(
  panel: HTMLElement,
  rules: Rules | null,
  snapshot: Snapshot,
  onBegin: () => void,
): void {
  const body = rules ? rules.lines.join("\n") : "Loading the rules…";
  panel.innerHTML = `
    <h2>ORDINARY TURN</h2>
    <div class="rules">${escape(body)}</div>
    <p class="hint">You play as ${escape(snapshot.human_name)}. Hal moves at the same time you do.</p>
    <form><button type="submit">Begin</button></form>
    <div class="error"></div>`;
  panel.querySelector("form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    onBegin();
  });
}

function escape(text: string): string {
  return text.replace(
    /[&<>"']/g,
    (character) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character] ?? character,
  );
}

/** Players open the rules before they begin the server-owned game. */
export function renderTitle(screen: HTMLElement, onStart: () => void): void {
  screen.innerHTML = `
    <div class="title-slide">
      <h1>DROP THE<br />HANDKERCHIEF</h1>
      <form><button type="submit">Start</button></form>
      <div class="error" role="alert"></div>
    </div>`;
  screen.querySelector("form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    onStart();
  });
}

/** Five rules, short enough to read before the first turn. */
export function renderRules(screen: HTMLElement, onBegin: () => void): void {
  screen.innerHTML = `
    <div class="rules-slide">
      <h1>THE RULES</h1>
      <dl>
        <dt>Each turn</dt>
        <dd>One player drops the handkerchief and the other checks for it. Each picks a second from 1 to 60 in secret. The roles swap after each turn.</dd>
        <dt>Check on or after the drop</dt>
        <dd>The check succeeds. The seconds from the drop to the check are squandered time (ST), and they go into the Checker's vial.</dd>
        <dt>Check before the drop</dt>
        <dd>The check fails. The Checker takes the vial plus 60 seconds as a dose and dies for that long. The vial empties.</dd>
        <dt>Revival</dt>
        <dd>Each dose adds to total time dead (TTD). More TTD lowers the odds of revival, and above 300 seconds a player stays dead.</dd>
        <dt>To win</dt>
        <dd>Be the last player alive.</dd>
      </dl>
      <form><button type="submit">Begin</button></form>
      <div class="error" role="alert"></div>
    </div>`;
  screen.querySelector("form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    onBegin();
  });
}

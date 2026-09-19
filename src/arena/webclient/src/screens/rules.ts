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

/** The chapter supplies the rules; the footer states the adaptation's additions. */
export function renderRules(screen: HTMLElement, onBegin: () => void): void {
  screen.innerHTML = `
    <div class="rules-slide">
      <img src="/art/panel/stl_rules" alt="Drop the Handkerchief: the chapter's rules" />
      <form><button type="submit">Begin</button></form>
      <p class="assumptions">Assumptions: same-second checks succeed and add 1s; modeled revival odds fall with dose and prior time dead, with no revival above 5 minutes total.</p>
      <p class="assumptions">The server records each game's moves and results for the leaderboard and for research. A one-year cookie links your games.</p>
      <div class="error" role="alert"></div>
    </div>`;
  screen.querySelector("form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    onBegin();
  });
}

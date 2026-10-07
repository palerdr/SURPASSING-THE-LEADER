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

/** Introduce the game and the vial before explaining each turn. */
export function renderRules(screen: HTMLElement, onBegin: () => void): void {
  screen.innerHTML = `
    <div class="rules-slide">
      <h1>THE RULES</h1>
      <dl>
        <dt>The game</dt>
        <dd>You play Baku against Hal in a game of timing and survival. The Checker sits in a chair with their back to the Dropper. The Dropper lets the handkerchief fall behind the chair, and the Checker turns to look for it. Each player can act once per turn.</dd>
        <dt>Your vial</dt>
        <dd>Each player has a vial for a drug that stops the heart. Both vials start empty. We measure the contents in seconds of death: more drug means a longer time without a heartbeat. You keep your own vial when the roles swap. The ST bar shows its contents.</dd>
        <dt>Each turn</dt>
        <dd>In a normal turn, you and Hal each choose a second from 1 to 60 and keep your choice secret until both actions finish. The roles swap after each turn that both players survive.</dd>
        <dt>Check on or after the drop</dt>
        <dd>You find the handkerchief, so your check succeeds. The seconds from the drop through the check add squandered time (ST) to your vial. A check in the same second adds one second.</dd>
        <dt>Check before the drop</dt>
        <dd>You find nothing, so your check fails. Yakou, the referee, injects the contents of your vial plus a 60-second penalty. Your heart stops for that many seconds, and he tries to revive you.</dd>
        <dt>Revival</dt>
        <dd>If Yakou revives you, your vial empties and the game continues. Each dose adds to your total time dead (TTD), shown on the other bar. A larger dose and more TTD reduce your chance of revival. Yakou cannot revive you if the dose reaches 300 seconds or your TTD exceeds 300 seconds.</dd>
        <dt>To win</dt>
        <dd>You win if Hal fails to revive. Hal wins if you fail to revive.</dd>
      </dl>
      <form><button type="submit">Begin</button></form>
      <div class="error" role="alert"></div>
    </div>`;
  screen.querySelector("form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    onBegin();
  });
}

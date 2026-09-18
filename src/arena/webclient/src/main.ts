// Entry point: one snapshot in, one screen out, on a single full-screen stage.
//
// The server owns all game state, so this holds only the latest snapshot and
// re-renders on every change. Every mutating call carries the sequence number
// of the snapshot it was decided from, which is what makes a double-submit or
// a stale second tab a 409 rather than a replayed move.
//
// Each decision is staged in two cuts: the scene (the three figures and a
// caption) waits for a click or Enter before the minute clock starts.
// The server sees one decision phase for these two screens.

import {
  ApiError,
  acknowledge,
  act,
  begin,
  getTranscript,
  newSession,
  readSession,
  restartSession,
  warmServer,
} from "./api";
import { keepServerWarm } from "./turn-warmup";
import { drawHud } from "./render/hud";
import { setHands } from "./render/dialplate";
import { drawScene, drawVictory } from "./render/scene";
import { cancelTurn, scheduleTurn, turnSeconds, unlockTicking } from "./audio/tick";
import { preload } from "./render/sprites";
import { renderBeat } from "./screens/beat";
import { renderLive } from "./screens/live";
import { renderOutcome } from "./screens/outcome";
import { renderRules, renderTitle } from "./screens/rules";
import { renderVictory } from "./screens/victory";
import { secondOnClock } from "./second";
import type { Snapshot, Transcript } from "./types";

/** How long the screen stays black after a commit before the result is shown. */
const HOLD_MS = 1500;
const REVEAL_MS = 3600;

const canvas = document.getElementById("scene") as HTMLCanvasElement;
const context = canvas.getContext("2d");
const screen = document.getElementById("screen");
if (!context || !screen) throw new Error("the page is missing its canvas or screen");

let snapshot: Snapshot | null = null;
let transcript: Transcript | null = null;
let busy = false;
let starting = true;
let beginRequested = false;
let titleOpen = true;
let opening = true;
const started = performance.now();
/** The decision the beat and clock belong to, and when its action screen opened. */
let decisionSequence = -1;
let beatOver = false;
/** The reveal being held back behind black, and the timer that lifts it. */
let heldSequence = -1;
let holdOver = false;
let holdTimer: number | null = null;
let committedAt: number | null = null;
let timeoutSubmitted = false;
let onClockSince = started;
/** The last whole second of the turn that has sounded. */
let tickedSecond = 0;
/** The second the count names now. Commit and Enter play this number. */
let shownSecond = 1;
let stopWarmup = () => {};

function keepServerReady(seconds?: number): void {
  stopWarmup();
  stopWarmup = keepServerWarm(
    () => document.hidden ? Promise.resolve() : warmServer(),
    seconds,
  );
}

function showError(message: string): void {
  const slot = screen?.querySelector<HTMLElement>(".error");
  if (slot) slot.textContent = message;
}

/** Run a server call, keeping the UI inert while it is in flight. */
async function commit(call: () => Promise<Snapshot>, timedOut = false): Promise<void> {
  if (busy) return;
  busy = true;
  stopWarmup();
  const deciding = snapshot?.phase === "awaiting_action";
  if (snapshot?.phase === "awaiting_ack" && holdTimer !== null) {
    window.clearTimeout(holdTimer);
    holdTimer = null;
    holdOver = true;
  }
  const elapsed = turnSeconds() ?? 0;
  if (deciding) {
    committedAt = performance.now();
    drawHud(null, transcript);
    screen!.innerHTML = "";
  }
  cancelTurn(timedOut);
  screen!.inert = true;
  try {
    snapshot = await call();
    // A transcript failure must not hide an accepted move or permit a replay.
    if (snapshot.phase === "game_over") {
      void getTranscript().then((history) => {
        transcript = history;
        if (snapshot?.phase === "game_over") render();
      }).catch(() => {});
    }
    render();
  } catch (error) {
    if (error instanceof ApiError && error.status === 409) {
      try {
        snapshot = await readSession();
        render();
        showError("That move was out of date, so the board was reloaded.");
        return;
      } catch (readError) {
        error = readError;
      }
    }
    committedAt = null;
    tickedSecond = -1;
    if (deciding && snapshot) {
      scheduleTurn(snapshot.turn_duration, elapsed);
      keepServerReady(Math.max(0, snapshot.turn_duration - elapsed));
    }
    render();
    if (snapshot?.phase === "awaiting_ack") keepServerReady();
    showError(error instanceof Error ? error.message : String(error));
  } finally {
    busy = false;
    screen!.inert = false;
  }
}

function requestBegin(): void {
  beginRequested = true;
  if (starting) {
    return;
  }
  opening = false;
  if (snapshot?.phase === "rules") {
    const current = snapshot;
    void commit(() => begin(current.sequence));
    return;
  }
  render();
}

function renderOpening(): void {
  if (!screen) return;
  screen.classList.add("opening");
  drawHud(null, transcript);
  if (titleOpen) {
    renderTitle(screen, () => {
      titleOpen = false;
      renderOpening();
      screen.querySelector("button")?.focus();
    });
  } else {
    renderRules(screen, requestBegin);
  }
}

/** The next game of the series: the rules were read once, so play resumes at once. */
async function nextGame(current: Snapshot): Promise<Snapshot> {
  const fresh = await newSession(current.sequence);
  return begin(fresh.sequence);
}

/** End the establishing shot and open the action screen. */
function cutToAction(): void {
  if (beatOver) return;
  beatOver = true;
  onClockSince = performance.now();
  tickedSecond = 0;
  timeoutSubmitted = false;
  if (snapshot) shownSecond = secondOnClock(0, snapshot.legal_seconds);
  // The whole turn is scheduled on the audio clock now, so the beats, the
  // count, and the pointer share one timeline from this instant.
  if (snapshot) {
    scheduleTurn(snapshot.turn_duration);
    keepServerReady(snapshot.turn_duration);
  }
  render();
}

/** Lift the black and show the result. */
function liftHold(): void {
  if (holdTimer !== null) {
    window.clearTimeout(holdTimer);
    holdTimer = null;
  }
  if (holdOver) return;
  holdOver = true;
  render();
  screen?.querySelector("button")?.focus();
}

function render(): void {
  if (!snapshot || !screen) return;
  if (opening) {
    renderOpening();
    return;
  }
  const current = snapshot;
  screen.classList.toggle("floor", false);
  screen.classList.toggle("opening", current.phase === "rules");

  // A fresh reveal is held behind black for a moment: the seconds are
  // decided, and the anticipation is the point.
  if (current.phase === "awaiting_ack" && current.sequence !== heldSequence) {
    keepServerReady();
    heldSequence = current.sequence;
    holdOver = false;
    if (holdTimer !== null) window.clearTimeout(holdTimer);
    const heldFor = committedAt === null ? 0 : performance.now() - committedAt;
    holdTimer = window.setTimeout(liftHold, Math.max(0, HOLD_MS - heldFor));
  }
  const held = current.phase === "awaiting_ack" && !holdOver;
  const stage = document.getElementById("stage")!;
  stage.classList.toggle("result-stage", current.phase === "awaiting_ack");
  stage.classList.toggle("revealing", current.phase === "awaiting_ack" && holdOver);
  stage.style.setProperty("--reveal-ms", `${REVEAL_MS}ms`);
  screen.inert = busy;
  drawHud(held ? null : current, transcript);
  if (held) {
    screen.innerHTML = "";
    return;
  }

  if (current.phase === "awaiting_action" && current.sequence !== decisionSequence) {
    decisionSequence = current.sequence;
    beatOver = false;
    keepServerReady();
  }

  switch (current.phase) {
    case "rules":
      keepServerReady();
      renderOpening();
      break;
    case "awaiting_action":
      if (beatOver) {
        // The commit reads the count at the moment of the gesture.
        renderLive(screen, current, shownSecond, () =>
          void commit(() => act(current.sequence, shownSecond)),
        );
      } else {
        renderBeat(screen, current);
      }
      break;
    case "awaiting_ack":
      if (current.last_outcome) {
        renderOutcome(screen, current.last_outcome, () =>
          void commit(() => acknowledge(current.sequence)),
        );
      }
      break;
    case "game_over":
      renderVictory(screen, current, transcript, () => void commit(() => nextGame(current)));
      break;
  }
}

function resize(): void {
  const ratio = window.devicePixelRatio || 1;
  const box = canvas.getBoundingClientRect();
  canvas.width = Math.round(box.width * ratio);
  canvas.height = Math.round(box.height * ratio);
}

/** What the floor shows under the current screen. */
function loop(): void {
  if (snapshot && context && !opening) {
    const showScene = snapshot.phase === "awaiting_action" && !beatOver;
    if (snapshot.phase === "game_over") {
      drawVictory(context, canvas.width, canvas.height, snapshot);
    } else if (showScene) {
      drawScene(context, canvas.width, canvas.height, snapshot, performance.now() - started);
    } else {
      context.fillStyle = "#000000";
      context.fillRect(0, 0, canvas.width, canvas.height);
    }
    if (snapshot.phase === "awaiting_action" && beatOver && !busy) {
      const plate = screen?.querySelector<HTMLElement>("[data-dialplate]");
      // The audio clock when the turn is scheduled, the frame clock otherwise.
      const heard = turnSeconds();
      const elapsed = heard !== null ? heard * 1000 : performance.now() - onClockSince;
      if (plate) setHands(plate, snapshot, elapsed);
      // The count beneath the plate rises on the instant each beat is heard:
      // after beat k it names second k + 1, the one now passing, and holds
      // at the last legal second for the final beat.
      const second = Math.min(snapshot.turn_duration, Math.floor(elapsed / 1000));
      if (second !== tickedSecond) {
        tickedSecond = second;
        shownSecond = secondOnClock(second, snapshot.legal_seconds);
        const count = screen?.querySelector<HTMLElement>("[data-count]");
        if (count) count.textContent = String(shownSecond);
      }
      // The minute is up: the rules give no more time, so the turn ends on
      // the last legal second. The referee still validates it like any other.
      if (second >= snapshot.turn_duration && !timeoutSubmitted) {
        timeoutSubmitted = true;
        const current = snapshot;
        const last = current.legal_seconds[current.legal_seconds.length - 1];
        if (last !== undefined) void commit(() => act(current.sequence, last), true);
      }
    }
  }
  requestAnimationFrame(loop);
}

// Enter advances every screen without reaching for the mouse, matching the
// terminal's press-Enter-to-continue pauses. Enter opens the clock from the
// scene, and on the clock it commits the second the count names.
document.addEventListener("keydown", (event) => {
  unlockTicking();
  if (event.repeat && event.key === "Enter") {
    event.preventDefault();
    return;
  }
  if (opening) {
    if (event.key === "Enter") {
      event.preventDefault();
      screen?.querySelector("form")?.requestSubmit();
    }
    return;
  }
  if (!snapshot) return;
  if (busy) return;
  if (snapshot.phase === "rules" && event.key === "Enter") {
    event.preventDefault();
    screen?.querySelector("form")?.requestSubmit();
    return;
  }
  if (snapshot.phase === "awaiting_action" && event.key === "Enter") {
    // Prevent the default here so a focused Commit button does not also
    // click; `commit` is guarded by `busy` either way.
    event.preventDefault();
    if (!beatOver) {
      cutToAction();
      return;
    }
    const current = snapshot;
    void commit(() => act(current.sequence, shownSecond));
    return;
  }
  if (snapshot.phase === "awaiting_ack" && event.key === "Enter") {
    event.preventDefault();
    const current = snapshot;
    void commit(() => acknowledge(current.sequence));
    return;
  }
  if (event.key === "Enter" && snapshot.phase === "game_over") {
    event.preventDefault();
    screen?.querySelector<HTMLButtonElement>('button[type="submit"]')?.click();
  }
});
document.addEventListener("pointerdown", () => {
  unlockTicking();
});
// Capture before a button submits, so one click cannot advance two screens.
document.addEventListener("click", (event) => {
  if (!opening && !busy && snapshot?.phase === "awaiting_action" && !beatOver) {
    event.preventDefault();
    cutToAction();
  }
}, true);

window.addEventListener("resize", resize);
window.addEventListener("pagehide", () => stopWarmup());
window.addEventListener("pageshow", (event) => {
  if (!event.persisted) return;
  if (opening || snapshot?.phase === "rules" || snapshot?.phase === "awaiting_ack"
      || (snapshot?.phase === "awaiting_action" && !beatOver)) keepServerReady();
  else if (snapshot?.phase === "awaiting_action" && beatOver) {
    keepServerReady(Math.max(0, snapshot.turn_duration - (performance.now() - onClockSince) / 1000));
  }
});

async function start(): Promise<void> {
  renderOpening();
  resize();
  preload();
  requestAnimationFrame(loop);
  try {
    const previous = await readSession();
    try {
      snapshot = await restartSession(previous.sequence);
    } catch (error) {
      if (!(error instanceof ApiError) || error.status !== 409) throw error;
      // The previous page may have committed its last request during reload.
      const latest = await readSession();
      snapshot = await restartSession(latest.sequence);
    }
    // Prepare the public decision while the player reads. Hal chooses no action here.
    snapshot = await begin(snapshot.sequence);
    starting = false;
    keepServerReady();
    if (beginRequested) requestBegin();
    else render();
  } catch (error) {
    starting = false;
    if (screen) {
      const card = document.createElement("div");
      card.className = "card";
      const heading = document.createElement("h2");
      heading.textContent = "COULD NOT OPEN YOUR GAME";
      const detail = document.createElement("p");
      detail.className = "lose";
      detail.textContent = error instanceof Error ? error.message : String(error);
      const retry = document.createElement("button");
      retry.textContent = "Retry";
      retry.addEventListener("click", () => window.location.reload());
      card.replaceChildren(heading, detail, retry);
      screen.replaceChildren(card);
    }
  }
}

void start();

// Entry point: one snapshot in, one screen out.
//
// The server owns all game state, so this holds only the latest snapshot and
// re-renders on every change. Every mutating call carries the sequence number
// of the snapshot it was decided from, which is what makes a double-submit or
// a stale second tab a 409 rather than a replayed move.

import { ApiError, acknowledge, act, begin, getRules, newSession, readSession } from "./api";
import { drawHud } from "./render/hud";
import { drawScene } from "./render/scene";
import { preload } from "./render/sprites";
import { renderLive } from "./screens/live";
import { renderOutcome } from "./screens/outcome";
import { renderRules } from "./screens/rules";
import { renderVictory } from "./screens/victory";
import type { Rules, Snapshot } from "./types";

const canvas = document.getElementById("scene") as HTMLCanvasElement;
const context = canvas.getContext("2d");
const panel = document.getElementById("panel");
if (!context || !panel) throw new Error("the page is missing its canvas or panel");

let snapshot: Snapshot | null = null;
let rules: Rules | null = null;
let busy = false;
const started = performance.now();

function showError(message: string): void {
  const slot = panel?.querySelector<HTMLElement>(".error");
  if (slot) slot.textContent = message;
}

/** Run a server call, keeping the UI inert while it is in flight. */
async function commit(call: () => Promise<Snapshot>): Promise<void> {
  if (busy) return;
  busy = true;
  try {
    snapshot = await call();
    render();
  } catch (error) {
    if (error instanceof ApiError && error.status === 409) {
      // Another tab or a double click moved the session on; resync rather than
      // guess at what happened.
      snapshot = await readSession();
      render();
      showError("That move was out of date, so the board was reloaded.");
    } else {
      showError(error instanceof Error ? error.message : String(error));
    }
  } finally {
    busy = false;
  }
}

function render(): void {
  if (!snapshot || !panel) return;
  drawHud(snapshot);
  const current = snapshot;
  switch (current.phase) {
    case "rules":
      renderRules(panel, rules, current, () => void commit(() => begin(current.sequence)));
      break;
    case "awaiting_action":
      renderLive(panel, current, (second) => void commit(() => act(current.sequence, second)));
      break;
    case "awaiting_ack":
      if (current.last_outcome) {
        renderOutcome(panel, current.last_outcome, () =>
          void commit(() => acknowledge(current.sequence)),
        );
      }
      break;
    case "game_over":
      renderVictory(panel, current, () => void commit(() => newSession(current.sequence)));
      break;
  }
}

function resize(): void {
  const ratio = window.devicePixelRatio || 1;
  const box = canvas.getBoundingClientRect();
  canvas.width = Math.round(box.width * ratio);
  canvas.height = Math.round(box.height * ratio);
}

function loop(): void {
  if (snapshot && context) {
    drawScene(context, canvas.width, canvas.height, snapshot, performance.now() - started);
  }
  requestAnimationFrame(loop);
}

// Enter advances the reveal without reaching for the mouse, matching the
// terminal's press-Enter-to-continue pause.
document.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || !snapshot) return;
  const active = document.activeElement;
  if (active instanceof HTMLInputElement) return;
  if (snapshot.phase === "awaiting_ack") {
    event.preventDefault();
    panel?.querySelector("button")?.click();
  }
});

window.addEventListener("resize", resize);

async function start(): Promise<void> {
  resize();
  preload();
  requestAnimationFrame(loop);
  try {
    [snapshot, rules] = await Promise.all([readSession(), getRules()]);
    render();
  } catch (error) {
    if (panel) {
      const heading = document.createElement("h2");
      heading.textContent = "NO SERVER";
      const detail = document.createElement("p");
      detail.className = "lose";
      detail.textContent = error instanceof Error ? error.message : String(error);
      const hint = document.createElement("p");
      hint.className = "hint";
      hint.textContent = "Start it with: uv run python -m arena.web";
      panel.replaceChildren(heading, detail, hint);
    }
  }
}

void start();

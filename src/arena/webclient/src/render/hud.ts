// The four bars in the top-left corner and the clock block in the top-right.
//
// Rendered as DOM rather than into the canvas so the text stays crisp at any
// device pixel ratio. Mirrors the columns `_player_column` builds in
// src/arena/tui.py, folded into one corner so the stage stays clear.

import type { PlayerView, Snapshot, Transcript } from "../types";
import { escapeHtml } from "./escape";
import { roleTitle } from "./identity";
import { tallyText } from "./text";

const el = (id: string): HTMLElement => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node;
};

function bar(value: number, maximum: number): string {
  const fraction = Math.max(0, Math.min(1, maximum > 0 ? value / maximum : 0));
  // The vial is a countdown to a fatal dose, so a nearly-full bar is danger.
  const hot = fraction >= 0.8 ? " hot" : "";
  return `<span class="bar${hot}"><i style="width:${(fraction * 100).toFixed(1)}%"></i></span>`;
}

function player(view: PlayerView, snapshot: Snapshot): string {
  const you = view.is_human ? ' <span class="you">(you)</span>' : "";
  const deaths = view.deaths > 0 ? ` · ${view.deaths} death${view.deaths === 1 ? "" : "s"}` : "";
  return `
    <div class="player">
      <h2>${escapeHtml(view.name.toUpperCase())}${you} <span class="you">${escapeHtml(roleTitle(view))}${escapeHtml(deaths)}</span></h2>
      <div class="stat"><span>ST</span>${bar(view.cylinder_seconds, snapshot.cylinder_max)}
        <span class="value">${view.cylinder_seconds.toFixed(0)}/${snapshot.cylinder_max.toFixed(0)}</span></div>
      <div class="stat"><span>TTD</span>${bar(view.ttd_seconds, snapshot.ttd_max)}
        <span class="value">${view.ttd_seconds.toFixed(0)}/${snapshot.ttd_max.toFixed(0)}</span></div>
    </div>`;
}

export function drawHud(snapshot: Snapshot | null, transcript: Transcript | null): void {
  const hud = el("hud");
  const corner = el("corner");
  if (snapshot === null || snapshot.phase === "rules") {
    hud.innerHTML = "";
    corner.innerHTML = "";
    return;
  }
  hud.innerHTML = snapshot.players.map((view) => player(view, snapshot)).join("");
  const lines = [
    `<div class="clock">${escapeHtml(snapshot.clock_display)}</div>`,
    `<div>Round <strong>${snapshot.round}</strong> · Half <strong>${snapshot.half}</strong>` +
      (snapshot.leap_window ? ' · <span class="leap">⚠ leap second</span>' : "") +
      "</div>",
    `<div>Game <strong>${snapshot.game_index + 1}</strong>` +
      (transcript ? ` · Series <strong>${escapeHtml(tallyText(transcript.tally))}</strong>` : "") +
      "</div>",
  ];
  corner.innerHTML = lines.join("");
}

// The four bars in the top-left corner and the clock block in the top-right.
//
// Rendered as DOM rather than into the canvas so the text stays crisp at any
// device pixel ratio. Mirrors the columns `_player_column` builds in
// src/arena/tui.py, folded into one corner so the stage stays clear.

import type { PlayerView, Snapshot, Transcript } from "../types";
import { escapeHtml } from "./escape";
import { type Gains, gainsSince } from "./gains";
import { roleTitle } from "./identity";
import { tallyText } from "./text";

const el = (id: string): HTMLElement => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node;
};

/**
 * One bar. `gained` is the part of `value` added by the half-round just
 * resolved; it is drawn in red at the end of the fill so the result screen
 * shows what the exchange cost or earned. Elsewhere it is zero and the fill
 * is one colour.
 */
function bar(value: number, maximum: number, gained = 0): string {
  const clamp = (v: number) => Math.max(0, Math.min(1, maximum > 0 ? v / maximum : 0));
  const fraction = clamp(value);
  const kept = clamp(value - Math.max(0, gained));
  const added = fraction - kept;
  // The vial is a countdown to a fatal dose, so a nearly-full bar is danger.
  const hot = fraction >= 0.8 ? " hot" : "";
  const gain = added > 0 ? `<i class="gain" style="width:${(added * 100).toFixed(1)}%"></i>` : "";
  return `<span class="bar${hot}"><i style="width:${(kept * 100).toFixed(1)}%"></i>${gain}</span>`;
}

function player(view: PlayerView, snapshot: Snapshot, gains: Gains): string {
  const you = view.is_human ? ' <span class="you">(you)</span>' : "";
  const deaths = view.deaths > 0 ? ` · ${view.deaths} death${view.deaths === 1 ? "" : "s"}` : "";
  return `
    <div class="player">
      <h2>${escapeHtml(view.name.toUpperCase())}${you} <span class="you">${escapeHtml(roleTitle(view))}${escapeHtml(deaths)}</span></h2>
      <div class="stat"><span>ST</span>${bar(view.cylinder_seconds, snapshot.cylinder_max, gains.cylinder)}
        <span class="value">${view.cylinder_seconds.toFixed(0)}/${snapshot.cylinder_max.toFixed(0)}</span></div>
      <div class="stat"><span>TTD</span>${bar(view.ttd_seconds, snapshot.ttd_max, gains.ttd)}
        <span class="value">${view.ttd_seconds.toFixed(0)}/${snapshot.ttd_max.toFixed(0)}</span></div>
    </div>`;
}

const NO_GAINS: Gains = { cylinder: 0, ttd: 0 };

/**
 * Draw both corners. `before` is the players as they stood when the human
 * committed; when given, each bar's growth since then is drawn in red.
 */
export function drawHud(
  snapshot: Snapshot | null,
  transcript: Transcript | null,
  before: readonly PlayerView[] | null = null,
): void {
  const hud = el("hud");
  const corner = el("corner");
  if (snapshot === null || snapshot.phase === "rules") {
    hud.innerHTML = "";
    corner.innerHTML = "";
    return;
  }
  hud.innerHTML = snapshot.players
    .map((view) => player(view, snapshot, before ? gainsSince(before, view) : NO_GAINS))
    .join("");
  const lines = [
    `<div class="clock">${escapeHtml(snapshot.clock_display)}</div>`,
    `<div>Round <strong>${snapshot.round}</strong> · Half <strong>${snapshot.half}</strong>` +
      (snapshot.leap_window ? ' · <span class="leap">⚠ leap second</span>' : "") +
      "</div>",
  ];
  // A hosted game stands alone, so a series line appears only once one exists.
  if (transcript && transcript.games.length > 1) {
    lines.push(`<div>Series <strong>${escapeHtml(tallyText(transcript.tally))}</strong></div>`);
  }
  corner.innerHTML = lines.join("");
}

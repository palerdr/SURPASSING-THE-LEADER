// Header, role line, and the two stat columns.
//
// Rendered as DOM rather than into the canvas: the canvas is for pixel art,
// and text stays crisp at any device pixel ratio this way. Mirrors the columns
// `_player_column` builds in src/arena/tui.py.

import type { PlayerView, Snapshot } from "../types";

const el = (id: string): HTMLElement => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node;
};

function bar(value: number, maximum: number): string {
  const fraction = Math.max(0, Math.min(1, maximum > 0 ? value / maximum : 0));
  // The cylinder is a countdown to injection, so a nearly-full bar is danger.
  const hot = fraction >= 0.8 ? " hot" : "";
  return `<span class="bar${hot}"><i style="width:${(fraction * 100).toFixed(1)}%"></i></span>`;
}

function column(player: PlayerView, snapshot: Snapshot): string {
  const you = player.is_human ? ' <span class="you">(you)</span>' : "";
  const role =
    player.name === snapshot.dropper_name
      ? "Dropper"
      : player.name === snapshot.checker_name
        ? "Checker"
        : "";
  return `
    <div class="player">
      <h2>${player.name}${you} <span class="you">${role}</span></h2>
      <div class="stat">
        <span>Cylinder</span>${bar(player.cylinder_seconds, snapshot.cylinder_max)}
        <span class="value">${player.cylinder_seconds.toFixed(0)}s</span>
      </div>
      <div class="stat">
        <span>TTD</span>${bar(player.ttd_seconds, snapshot.ttd_max)}
        <span class="value">${player.ttd_seconds.toFixed(0)}s</span>
      </div>
      <div class="stat"><span>Deaths</span><span class="value">${player.deaths}</span></div>
    </div>`;
}

export function drawHud(snapshot: Snapshot): void {
  el("clock").textContent = snapshot.clock_display;
  el("meta").innerHTML = [
    `Round ${snapshot.round} · Half ${snapshot.half}`,
    `Turn ${snapshot.turn_duration}s`,
    snapshot.leap_window ? '<span class="leap">⚠ LEAP WINDOW</span>' : "",
  ]
    .filter(Boolean)
    .join("");
  el("roles").innerHTML =
    `<span>DROPPER <strong>${snapshot.dropper_name}</strong></span>` +
    `<span>CHECKER <strong>${snapshot.checker_name}</strong></span>`;
  el("stats").innerHTML = snapshot.players.map((player) => column(player, snapshot)).join("");
}

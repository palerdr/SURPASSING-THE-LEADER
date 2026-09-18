// The dial plate from palerdr.github.io, ported line for line: the white
// plate with two rims, sixty engraved ticks, the 61st-second leap tick, an
// hour hand, a minute hand, and the red pointer that sweeps the seconds.
//
// Here the hour and minute hands are set to the game clock, and the pointer
// sweeps the turn from the moment the action screen opens. The readout in
// the core shows the game clock; the count beneath the
// plate, kept by main.ts, falls on the same frame each beat sounds.

import type { Snapshot } from "../types";
import { escapeHtml } from "./escape";

const TICKS = Array.from({ length: 60 }, (_, i) => {
  const major = i % 5 === 0;
  const rad = ((i * 6 - 90) * Math.PI) / 180;
  const r1 = major ? 440 : 452;
  return {
    major,
    x1: 500 + r1 * Math.cos(rad),
    y1: 500 + r1 * Math.sin(rad),
    x2: 500 + 466 * Math.cos(rad),
    y2: 500 + 466 * Math.sin(rad),
  };
});

const fixed = (value: number): string => value.toFixed(2);

/** The plate's markup. Hands are moved afterwards by `setHands`. */
export function dialplateMarkup(snapshot: Snapshot): string {
  const ticks = TICKS.map(
    (t) =>
      `<line class="tick${t.major ? " major" : ""}" x1="${fixed(t.x1)}" y1="${fixed(t.y1)}" x2="${fixed(t.x2)}" y2="${fixed(t.y2)}" />`,
  ).join("");
  return `
    <div class="dialplate" data-dialplate>
      <svg class="engraving" viewBox="0 0 1000 1000" aria-hidden="true">
        <circle class="rim" cx="500" cy="500" r="474" />
        <circle class="rim thin" cx="500" cy="500" r="424" />
        ${ticks}
        <!-- The 61st second, which most clocks pretend does not exist. -->
        <line class="leap-tick" x1="500" y1="16" x2="500" y2="78" />
      </svg>
      <svg class="hands" viewBox="0 0 1000 1000" aria-hidden="true">
        <g data-hand="hour"><polygon points="494,500 506,500 500,266" /></g>
        <g data-hand="minute"><polygon points="496,500 504,500 500,150" /></g>
        <g data-hand="pointer" data-pointer data-live>
          <polygon points="497.5,500 502.5,500 500,96" />
        </g>
        <circle class="hub" cx="500" cy="500" r="13" />
      </svg>
      <div class="core">
        <p class="who mono">${escapeHtml(snapshot.clock_display)}</p>
      </div>
    </div>`;
}

/** "8:59:60 AM" → hours, minutes, seconds. The engine formats the leap second itself. */
export function parseClock(display: string): { h: number; m: number; s: number } {
  const match = /(\d+):(\d\d)(?::(\d\d))?/.exec(display);
  if (!match) return { h: 0, m: 0, s: 0 };
  return { h: Number(match[1]), m: Number(match[2]), s: Number(match[3] ?? "0") };
}

/**
 * Set the hands. The pointer steps, one tick per whole second heard, so it
 * stands on exactly the mark the count beneath the plate names: after beat
 * k it points at tick k and the count reads the turn less k. The hour and
 * minute hands read the game clock plus those same whole seconds.
 */
export function setHands(plate: HTMLElement, snapshot: Snapshot, elapsedMs: number): void {
  const hour = plate.querySelector<SVGGElement>('[data-hand="hour"]');
  const minute = plate.querySelector<SVGGElement>('[data-hand="minute"]');
  const pointer = plate.querySelector<SVGGElement>("[data-pointer]");
  const clock = parseClock(snapshot.clock_display);
  // We hold at the red leap mark for the extra second of a 61-second turn.
  const swept = Math.min(60, Math.max(0, Math.floor(elapsedMs / 1000)));
  const s = clock.s + swept;
  const m = clock.m + s / 60;
  const h = (clock.h % 12) + m / 60;
  if (hour) hour.style.transform = `rotate(${h * 30}deg)`;
  if (minute) minute.style.transform = `rotate(${m * 6}deg)`;
  if (pointer) pointer.style.transform = `rotate(${swept * 6}deg)`;
}

import type { PlayerView } from "../types";

/** The seconds each player's bars grew in the half-round just resolved. */
export interface Gains {
  cylinder: number;
  ttd: number;
}

/**
 * Compare a player's bars before and after a resolved half-round. This is a
 * difference of two server states, not a rule: the referee decided who gained
 * what, and the client only reports which fill grew.
 */
export function gainsSince(before: readonly PlayerView[], after: PlayerView): Gains {
  const previous = before.find((view) => view.character === after.character);
  if (!previous) return { cylinder: 0, ttd: 0 };
  return {
    cylinder: Math.max(0, after.cylinder_seconds - previous.cylinder_seconds),
    ttd: Math.max(0, after.ttd_seconds - previous.ttd_seconds),
  };
}

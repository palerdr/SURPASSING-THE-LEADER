import type { PlayerView, Snapshot } from "../types";

export type PlayerRole = PlayerView["role"];

/** Resolve a role through the server-owned role field, never through a label. */
export function playerForRole(
  snapshot: Pick<Snapshot, "players">,
  role: PlayerRole,
): PlayerView {
  const matches = snapshot.players.filter((player) => player.role === role);
  const player = matches[0];
  if (player === undefined || matches.length !== 1) {
    throw new Error(`snapshot must contain exactly one ${role}`);
  }
  return player;
}

export function roleTitle(player: Pick<PlayerView, "role">): "Dropper" | "Checker" {
  return player.role === "dropper" ? "Dropper" : "Checker";
}

/** The server determines seat identity; display labels never decide a win. */
export function humanWon(
  snapshot: Pick<Snapshot, "winner_is_human">,
): boolean {
  return snapshot.winner_is_human === true;
}

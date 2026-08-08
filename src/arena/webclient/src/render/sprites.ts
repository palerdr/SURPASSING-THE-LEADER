// Frames come from the Python server already keyed, mirrored, and split.
//
// That is deliberate. `Sprite.keyed` (src/arena/sprites.py) clears the paper by
// flood-filling inward from the border, not by thresholding brightness — a
// threshold would punch holes through Baku's white coat and the handkerchief.
// The sheet splitter also repairs figures that straddle a quarter boundary.
// Reimplementing either here would be a second, divergent implementation of
// art the terminal front end has already validated, so we do not.

export type Character = "baku" | "hal" | "yakou";
export type Pose = "idle" | "dropping" | "seated" | "standing";

const cache = new Map<string, HTMLImageElement>();
const failed = new Set<string>();

/** Map a player's engine name onto the character whose art represents them. */
export function characterFor(name: string): Character {
  return name.toLowerCase() === "hal" ? "hal" : "baku";
}

function key(character: Character, pose: Pose, index: number): string {
  return `${character}/${pose}/${index}`;
}

/**
 * A frame, or null while it loads or if the art is absent.
 *
 * Art is optional by design — the terminal degrades to labelled placeholders
 * and so does this. Returning null rather than throwing keeps a missing sprite
 * from taking down the frame loop.
 */
export function frame(character: Character, pose: Pose, index: number): HTMLImageElement | null {
  const id = key(character, pose, index);
  if (failed.has(id)) return null;
  const cached = cache.get(id);
  if (cached) return cached.complete && cached.naturalWidth > 0 ? cached : null;

  const image = new Image();
  image.src = `/art/${id}.png`;
  image.addEventListener("error", () => failed.add(id));
  cache.set(id, image);
  return null;
}

/** Warm the frames a game actually uses, so the first paint is not empty. */
export function preload(): void {
  const players: Character[] = ["baku", "hal"];
  for (const character of players) {
    for (let i = 0; i < 4; i += 1) frame(character, "idle", i);
    frame(character, "dropping", 0);
    frame(character, "seated", 0);
  }
  for (let i = 0; i < 4; i += 1) frame("yakou", "idle", i);
  frame("yakou", "standing", 0);
}

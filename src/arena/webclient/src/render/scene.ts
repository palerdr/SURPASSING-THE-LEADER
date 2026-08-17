// The staged scene, after art/panels/stl1.jpg: the seated player on the left,
// Yakou standing at the centre, the Dropper on the right. Roles swap between
// halves, so the two players trade places while Yakou never moves.
//
// Constants are taken from src/arena/tui.py so both front ends stage the scene
// identically.

import type { Snapshot } from "../types";
import { playerForRole } from "./identity";
import { type Character, type Pose, frame } from "./sprites";

const BACKGROUND = "#000000";
/** Fraction of the canvas the figures occupy. Framed small, as the panel is. */
const SCENE_FILL = 0.4;
/** Relative figure heights, derived from pose rather than canvas size. */
const POSE_SCALE: Record<Pose, number> = {
  dropping: 1.0,
  standing: 0.9,
  idle: 1.0,
  seated: 0.8,
};
/** How far a pose floats above the floor line — Yakou stands a step back. */
const POSE_LIFT: Record<string, number> = { standing: 0.1 };
const GUTTER = 0.04;
const IDLE_MS = 180;

interface Figure {
  character: Character;
  pose: Pose;
  /** Baku and Hal frames are pre-mirrored to face left; the left slot faces in. */
  flip: boolean;
}

function figures(snapshot: Snapshot): Figure[] {
  const checker = playerForRole(snapshot, "checker");
  const dropper = playerForRole(snapshot, "dropper");
  return [
    { character: checker.character, pose: "seated", flip: true },
    { character: "yakou", pose: "idle", flip: false },
    { character: dropper.character, pose: "dropping", flip: false },
  ];
}

export function drawScene(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  snapshot: Snapshot,
  elapsedMs: number,
): void {
  ctx.fillStyle = BACKGROUND;
  ctx.fillRect(0, 0, width, height);
  ctx.imageSmoothingEnabled = false;

  const band = height * SCENE_FILL;
  const floor = height * 0.86;
  const gutter = width * GUTTER;
  const cast = figures(snapshot);

  // Yakou is the only animated figure; the players hold their pose, as in the
  // terminal scene.
  const tick = Math.floor(elapsedMs / IDLE_MS) % 4;

  const drawn = cast.map((figure) => {
    const index = figure.character === "yakou" ? tick : 0;
    const image =
      frame(figure.character, figure.pose, index) ??
      (figure.character === "yakou" ? frame("yakou", "standing", 0) : null);
    const scale = POSE_SCALE[figure.pose];
    const target = band * scale;
    const aspect = image && image.naturalHeight > 0 ? image.naturalWidth / image.naturalHeight : 0.5;
    return { figure, image, height: target, width: target * aspect };
  });

  const total = drawn.reduce((sum, item) => sum + item.width, 0) + gutter * 2;
  let x = (width - total) / 2;

  for (const item of drawn) {
    const lift = (POSE_LIFT[item.figure.pose] ?? 0) * band;
    const top = floor - item.height - lift;
    if (item.image) {
      ctx.save();
      if (item.figure.flip) {
        ctx.translate(x + item.width, top);
        ctx.scale(-1, 1);
        ctx.drawImage(item.image, 0, 0, item.width, item.height);
      } else {
        ctx.drawImage(item.image, x, top, item.width, item.height);
      }
      ctx.restore();
    } else {
      // Same graceful degradation the terminal has when art is absent.
      ctx.strokeStyle = "#333333";
      ctx.strokeRect(x, top, item.width, item.height);
      ctx.fillStyle = "#777777";
      ctx.font = "12px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.fillText(item.figure.character, x + item.width / 2, top + item.height / 2);
    }
    x += item.width + gutter;
  }
}

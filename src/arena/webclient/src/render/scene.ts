// The staged scene, after art/panels/stl1.jpg: the seated player on the left,
// Yakou standing at the centre, the Dropper on the right. Roles swap between
// halves, so the two players trade places while Yakou never moves.
//
// Constants are taken from src/arena/tui.py so both front ends stage the scene
// identically.

import type { Snapshot } from "../types";
import { humanWon, playerForRole } from "./identity";
import { type Character, type Pose, frame } from "./sprites";

const BACKGROUND = "#000000";
/**
 * Fraction of the canvas the figures occupy. The terminal uses 0.4 of a
 * scene band that is a slice of the frame; here the canvas is the whole
 * viewport, so the trio is a miniature, a small group low in the night
 * field, as the panel frames it.
 */
const SCENE_FILL = 0.11;
/** The winner's still stands twice as tall as the scene's miniatures. */
const VICTORY_FILL = 0.22;
/** Relative figure heights, derived from pose rather than canvas size. */
const POSE_SCALE: Record<Pose, number> = {
  dropping: 1.0,
  standing: 0.9,
  idle: 1.0,
  seated: 0.8,
  win_screen: 1.0,
};
/** Baku's dropping fixture fills its canvas edge to edge; Hal's leaves margin. */
const BAKU_DROPPING_SCALE = 0.9;
/** How far a pose floats above the floor line — Yakou stands a step back. */
const POSE_LIFT: Record<string, number> = { standing: 0.1 };
const GUTTER = 0.04;
const FLOOR = 0.6;

interface Figure {
  character: Character;
  pose: Pose;
  /** Baku and Hal frames are pre-mirrored to face left. */
  flip: boolean;
}

function figures(snapshot: Snapshot): Figure[] {
  const checker = playerForRole(snapshot, "checker");
  const dropper = playerForRole(snapshot, "dropper");
  return [
    // The Checker sits with his back to the drop: he must not watch for it.
    { character: checker.character, pose: "seated", flip: false },
    { character: "yakou", pose: "idle", flip: false },
    { character: dropper.character, pose: "dropping", flip: false },
  ];
}

function poseScale(figure: Figure): number {
  const scale = POSE_SCALE[figure.pose];
  return figure.pose === "dropping" && figure.character !== "hal"
    ? scale * BAKU_DROPPING_SCALE
    : scale;
}

function placeholder(
  ctx: CanvasRenderingContext2D,
  label: string,
  x: number,
  top: number,
  width: number,
  height: number,
): void {
  // Same graceful degradation the terminal has when art is absent.
  ctx.strokeStyle = "#333333";
  ctx.strokeRect(x, top, width, height);
  ctx.fillStyle = "#777777";
  ctx.font = "12px ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.fillText(label, x + width / 2, top + height / 2);
}

function clear(ctx: CanvasRenderingContext2D, width: number, height: number): void {
  ctx.fillStyle = BACKGROUND;
  ctx.fillRect(0, 0, width, height);
}

/**
 * Pixel rows a figure of full pose height is reduced to before it is blown
 * back up. Smaller is chunkier. The reduction is a smooth downsample, so the
 * pixels are averaged colours from the art rather than a shimmer of whichever
 * source pixel happened to fall on the grid.
 */
const PIXEL_ROWS = 44;

const coarse = new Map<string, HTMLCanvasElement>();

/** A figure reduced to its pixel grid, memoised per image and pose height. */
function pixelated(image: HTMLImageElement, rows: number): HTMLCanvasElement {
  const key = `${image.src}#${rows}`;
  const cached = coarse.get(key);
  if (cached) return cached;
  const aspect = image.naturalHeight > 0 ? image.naturalWidth / image.naturalHeight : 0.5;
  const small = document.createElement("canvas");
  small.height = Math.max(1, rows);
  small.width = Math.max(1, Math.round(rows * aspect));
  const ctx = small.getContext("2d");
  if (ctx) {
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(image, 0, 0, small.width, small.height);
  }
  coarse.set(key, small);
  return small;
}

/** Blow the coarse figure up with hard pixel edges. */
function blit(
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement,
  scale: number,
  x: number,
  top: number,
  width: number,
  height: number,
): void {
  const grid = pixelated(image, Math.round(PIXEL_ROWS * scale));
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(grid, x, top, width, height);
  ctx.imageSmoothingEnabled = true;
}

export function drawScene(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  snapshot: Snapshot,
  elapsedMs: number,
): void {
  clear(ctx, width, height);

  const band = height * SCENE_FILL;
  const floor = height * FLOOR;
  const gutter = width * GUTTER;
  const cast = figures(snapshot);

  // Every figure holds one still. The terminal cycles Yakou's idle sheet, but
  // its four frames are four separate drawings, each re-centred on the shared
  // canvas, so on a screen the cycle read as the referee twitching.
  void elapsedMs;

  const drawn = cast.map((figure) => {
    const image =
      frame(figure.character, figure.pose, 0) ??
      (figure.character === "yakou" ? frame("yakou", "standing", 0) : null);
    const target = band * poseScale(figure);
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
      const scale = poseScale(item.figure);
      if (item.figure.flip) {
        ctx.translate(x + item.width, top);
        ctx.scale(-1, 1);
        blit(ctx, item.image, scale, 0, 0, item.width, item.height);
      } else {
        blit(ctx, item.image, scale, x, top, item.width, item.height);
      }
      ctx.restore();
    } else {
      placeholder(ctx, item.figure.character, x, top, item.width, item.height);
    }
    x += item.width + gutter;
  }
}

/**
 * Show the winner at the centre: Hal's victory pose or Baku's first idle frame.
 */
export function drawVictory(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  snapshot: Snapshot,
): void {
  clear(ctx, width, height);
  if (snapshot.winner_name === null) {
    ctx.fillStyle = "#777777";
    ctx.font = "14px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.fillText(snapshot.stopped ? "SESSION STOPPED" : "NO WINNER", width / 2, height / 2);
    return;
  }
  const character: Character = humanWon(snapshot) ? "baku" : "hal";
  const image = frame(character, "win_screen", 0) ?? frame(character, "idle", 0);
  const target = height * VICTORY_FILL;
  const aspect = image && image.naturalHeight > 0 ? image.naturalWidth / image.naturalHeight : 0.5;
  const figureWidth = target * aspect;
  const x = (width - figureWidth) / 2;
  const top = height * FLOOR - target;
  if (image) {
    blit(ctx, image, 1, x, top, figureWidth, target);
  } else {
    placeholder(ctx, snapshot.winner_name, x, top, figureWidth, target);
  }
}

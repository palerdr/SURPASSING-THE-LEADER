"""Prepared character art shared by the terminal and the browser front ends.

This module turns the source sprite sheets under the repository ``art/`` tree
into keyed, split, and mirrored frames. The terminal draws those frames as
character cells, and the browser server serves them as PNG images. Every path
resolves from this file, so a front end finds the art from any working
directory.

When the art is missing, :meth:`SceneArt.load` returns no poses, and each front
end draws a labelled placeholder in place of the figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from arena.presentation.sprites import Sprite, SpriteError, load_sprite, write_png

# The idle sheets are four frames laid out horizontally.
IDLE_FRAMES = 4

# Longest edge used when preparing a fixture, with headroom for large layouts.
_WORK_EDGE = 320

# Bumped whenever the preparation pipeline changes, so stale caches are ignored.
_PIPELINE_VERSION = 5

# The repository ``art/`` tree. This file sits in src/arena/presentation/, so
# parents[3] is the repository root.
ART_ROOT = Path(__file__).resolve().parents[3] / "art"

_ART_ROOT = ART_ROOT / "sprites"

# Prepared frames are memoised here. Generated data, so it stays gitignored and
# lives beside the art it derives from rather than under arena/.
_CACHE_DIR = ART_ROOT / ".sprite-cache"

# Manga panels, such as the rules spread that opens the browser game.
PANEL_ROOT = ART_ROOT / "panels"

# Pose filenames per character, keyed by the action that character is taking.
_PLAYER_POSES = ("dropping", "seated", "idle")
_YAKOU_POSES = ("standing", "idle")


def _components(sprite: Sprite) -> list[tuple[list[tuple[int, int]], int, int, int, int]]:
    """4-connected regions of opaque pixels, as ``(cells, x0, y0, x1, y1)``."""
    width, height = sprite.width, sprite.height
    seen = [[False] * width for _ in range(height)]
    regions = []
    for y in range(height):
        row = sprite.rows[y]
        for x in range(width):
            if seen[y][x] or row[x][3] < 128:
                continue
            seen[y][x] = True
            stack = [(x, y)]
            cells: list[tuple[int, int]] = []
            x0 = x1 = x
            y0 = y1 = y
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                x0, x1 = min(x0, cx), max(x1, cx)
                y0, y1 = min(y0, cy), max(y1, cy)
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and not seen[ny][nx]
                        and sprite.rows[ny][nx][3] >= 128
                    ):
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            regions.append((cells, x0, y0, x1, y1))
    return regions


def _sheet_frames(sheet: Sprite, count: int) -> tuple[Sprite, ...] | None:
    """Extract ``count`` whole figures from a keyed sheet, or ``None``.

    The generated sheets do not place their drawings on exact quarter
    boundaries — a figure can straddle a cut line, so slicing at fixed
    quarters showed a slice of a neighbouring drawing beside Yakou and cut the
    same slice out of the frame it belonged to. Grouping connected regions by
    horizontal position instead recovers each figure whole wherever it sits.

    Debris is dropped on the way: regions wider than any single figure could
    be (edge artifact lines that run the length of the sheet), thin line-like
    segments, and specks. Every recovered figure is then centred on one shared
    canvas — max figure width, union of vertical extents — so an animation
    holds one stable shape and the figures pack tightly in the scene.
    """
    regions = _components(sheet)
    if not regions:
        return None
    frame_width = sheet.width / count
    largest = max(len(cells) for cells, *_ in regions)
    figures = []
    for cells, x0, y0, x1, y1 in regions:
        width, height = x1 - x0 + 1, y1 - y0 + 1
        if width > 1.5 * frame_width:
            continue  # artifact line running along the sheet
        if width >= 6 * height and height <= sheet.height // 20:
            continue  # line-like artifact segment
        if len(cells) < largest / 50:
            continue  # speck
        figures.append((cells, x0, y0, x1, y1))
    if len(figures) < count:
        return None

    figures.sort(key=lambda region: region[1] + region[3])
    centres = [(region[1] + region[3]) / 2 for region in figures]
    order = sorted(
        range(len(figures) - 1), key=lambda i: centres[i + 1] - centres[i], reverse=True
    )
    cuts = sorted(order[: count - 1])
    groups = []
    start = 0
    for cut in (*cuts, len(figures) - 1):
        groups.append(figures[start : cut + 1])
        start = cut + 1

    y0 = min(region[2] for group in groups for region in group)
    y1 = max(region[4] for group in groups for region in group)
    spans = [
        (min(region[1] for region in group), max(region[3] for region in group))
        for group in groups
    ]
    width = max(gx1 - gx0 + 1 for gx0, gx1 in spans)
    height = y1 - y0 + 1
    clear = (0, 0, 0, 0)
    frames = []
    for group, (gx0, gx1) in zip(groups, spans):
        rows = [[clear] * width for _ in range(height)]
        offset = (width - (gx1 - gx0 + 1)) // 2
        for cells, *_ in group:
            for cx, cy in cells:
                rows[cy - y0][cx - gx0 + offset] = sheet.rows[cy][cx]
        frames.append(Sprite(width, height, tuple(tuple(row) for row in rows)))
    return tuple(frames)


def _prepare_sheet(sheet: Sprite, count: int, *, mirror: bool = True) -> tuple[Sprite, ...]:
    """Key out the paper, isolate the figures, and optionally mirror them.

    The players are mirrored to face left as the panel stages them; Yakou is
    not — his raised watch arm is part of the canonical drawing, so the
    referee renders exactly as authored.

    Figure extraction falls back to fixed quarter slices cropped to one shared
    canvas when region grouping cannot find ``count`` figures — synthetic or
    damaged art still renders, just without the straddle repair.

    The rim light is deliberately not applied here. At source resolution a
    one-pixel rim is roughly a fifth of a terminal cell and scaling would
    dissolve it, so it is drawn after scaling instead — see
    :func:`arena.tui._scaled`.
    """
    keyed = sheet.keyed()
    frames = _sheet_frames(keyed, count)
    if frames is None:
        if count > 1:
            try:
                parts = keyed.frames(count)
            except SpriteError:
                parts = (keyed,)
        else:
            parts = (keyed,)
        bounds = [box for box in (part.opaque_bounds() for part in parts) if box is not None]
        if bounds:
            x0 = min(box[0] for box in bounds)
            y0 = min(box[1] for box in bounds)
            x1 = max(box[2] for box in bounds)
            y1 = max(box[3] for box in bounds)
            frames = tuple(
                part.crop(x0, y0, x1 - x0 + 1, y1 - y0 + 1) for part in parts
            )
        else:
            frames = parts
    if not mirror:
        return tuple(frames)
    return tuple(frame.mirrored() for frame in frames)


def _load_prepared(
    source: Path, *, count: int, cache_dir: Path | None, mirror: bool = True
) -> tuple[Sprite, ...]:
    """Prepare a fixture's frames, memoised on disk.

    The sources are ~1250px and Paeth-filtered, and Paeth is byte-sequential,
    so decoding them in pure Python costs seconds each. Prepared frames are
    therefore written back as small keyed PNGs and reused until the source
    changes; the cache key carries the source's mtime and size, the working
    resolution, and a pipeline version, so editing a fixture or changing how
    fixtures are prepared both invalidate it automatically.
    """
    try:
        stat = source.stat()
    except OSError:
        return ()

    token = (
        f"{stat.st_mtime_ns:x}-{stat.st_size:x}-{_WORK_EDGE}-{_PIPELINE_VERSION}-{int(mirror)}"
    )
    if cache_dir is not None:
        entries = sorted(cache_dir.glob(f"{source.stem}.{token}.*.png"))
        loaded = [load_sprite(entry) for entry in entries]
        if loaded and all(sprite is not None for sprite in loaded):
            return tuple(sprite for sprite in loaded if sprite is not None)

    # An idle sheet holds its frames side by side, so it is decoded wider to
    # leave each individual frame at working resolution.
    sprite = load_sprite(source, _WORK_EDGE * count)
    if sprite is None:
        return ()
    prepared = _prepare_sheet(sprite, count, mirror=mirror)

    if cache_dir is not None and prepared:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            for stale in cache_dir.glob(f"{source.stem}.*.png"):
                stale.unlink(missing_ok=True)
            for index, frame in enumerate(prepared):
                write_png(frame, cache_dir / f"{source.stem}.{token}.{index}.png")
        except OSError:
            pass  # a read-only tree just means we prepare again next run
    return prepared


@dataclass(frozen=True, slots=True)
class SceneArt:
    """Prepared fixtures keyed by ``(character, pose)``. Any may be absent.

    ``idle`` entries hold the four frames of that character's sprite sheet;
    every other pose holds a single frame.
    """

    poses: dict[tuple[str, str], tuple[Sprite, ...]] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        root: str | Path = _ART_ROOT,
        *,
        cache_dir: str | Path | None = _CACHE_DIR,
    ) -> SceneArt:
        base = Path(root)
        poses: dict[tuple[str, str], tuple[Sprite, ...]] = {}
        wanted = {
            "baku": (*_PLAYER_POSES, "win_screen"),
            "hal": (*_PLAYER_POSES, "win_screen"),
            "yakou": _YAKOU_POSES,
        }
        for character, names in wanted.items():
            for pose in names:
                source = base / character / f"{character}_{pose}.png"
                frames = _load_prepared(
                    source,
                    count=IDLE_FRAMES if pose == "idle" else 1,
                    cache_dir=None if cache_dir is None else Path(cache_dir),
                    # Yakou keeps his canonical handedness — the raised watch
                    # arm belongs where the artist put it.
                    mirror=character != "yakou",
                )
                if frames:
                    poses[(character, pose)] = frames
        return cls(poses)

    def frame(self, character: str, pose: str, index: int = 0) -> Sprite | None:
        """One frame of a pose, cycling for animated sheets."""
        sheet = self.poses.get((character.strip().lower(), pose))
        if not sheet:
            return None
        return sheet[index % len(sheet)]

    def for_action(self, player_name: str, pose: str, index: int = 0) -> Sprite | None:
        """The sprite for the action a player is taking, falling back to idle."""
        character = "hal" if player_name.strip().lower() == "hal" else "baku"
        return self.frame(character, pose, index) or self.frame(character, "idle", index)

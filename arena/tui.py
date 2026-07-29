"""Terminal interface for the canonical STL play loop.

This is a display layer and nothing more. Every function here takes canonical
engine objects and returns text; none of them advance the clock, resolve a
half-round, or write to a :class:`~stl.engine.game.Player`. The STL engine
remains the only referee, exactly as ``arena/AGENTS.md`` requires.

The scene is staged after ``art/panels/stl1.jpg``: the seated player on the
left, Yakou standing at the centre, and the Dropper on the right facing left
with the handkerchief held inward. Roles swap between halves, so the two players
swap around Yakou, who never moves.

Sprite art lives under the repository ``art/`` tree, which is gitignored. When
it is missing the scene degrades to a labelled text placeholder and the rest of
the interface is unaffected.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from arena.sprites import (
    GLYPH_GRIDS,
    Sprite,
    SpriteError,
    load_sprite,
    render_cells,
    write_png,
)
from stl.engine.game import (
    CYLINDER_MAX,
    TOTAL_TTD_MAX,
    Game,
    HalfRoundRecord,
    HalfRoundResult,
    Player,
)

# The idle sheets are four frames laid out horizontally.
IDLE_FRAMES = 4

# The scene is drawn on a black field, as the panel is a night exterior.
SCENE_BACKGROUND = (0, 0, 0)

# Cells between adjacent figures. The panel groups all three tightly, so this
# stays narrow; the figures are sized to their own shape rather than padded out
# to fill equal slots.
SCENE_GUTTER = 1

# Fraction of the scene band the figures occupy. The trio is framed small and
# centred against the night field, as the panel frames it, rather than blown up
# to the edges of the window. Slightly above a third: every extra row is
# sampling resolution, and at a strict third the figures went soft.
SCENE_FILL = 0.4

# Relative figure heights. Derived from the pose, never from the prepared pixel
# height: the sources have different canvas aspects (Baku 1254x1254, Hal
# 1024x1536), so measured heights describe the canvas rather than the character.
# Standing is Yakou's pose: he is drawn slightly smaller because he stands a
# step behind the players, not on their line.
POSE_SCALE = {"dropping": 1.0, "standing": 0.9, "idle": 1.0, "seated": 0.80}

# How far a pose floats above the common floor line, as a fraction of the band.
# Depth cue for the same step back: Yakou's feet sit a little higher than the
# players', as the panel stages him.
POSE_LIFT = {"standing": 0.1}

# Longest edge used when preparing a fixture, with headroom for large layouts.
_WORK_EDGE = 320

# Bumped whenever the preparation pipeline changes, so stale caches are ignored.
_PIPELINE_VERSION = 5

_RESET = "\x1b[0m"

_RESULT_TEXT = {
    HalfRoundResult.CHECK_SUCCESS: "CHECK SUCCESS",
    HalfRoundResult.CHECK_FAIL_SURVIVED: "CHECK FAILED — died, revived",
    HalfRoundResult.CHECK_FAIL_DIED: "CHECK FAILED — died permanently",
    HalfRoundResult.CYLINDER_OVERFLOW_SURVIVED: "VIAL OVERFLOW — died, revived",
    HalfRoundResult.CYLINDER_OVERFLOW_DIED: "VIAL OVERFLOW — died permanently",
}

_ART_ROOT = Path("art/sprites")

# Prepared frames are memoised here. Generated data, so it stays gitignored and
# lives beside the art it derives from rather than under arena/.
_CACHE_DIR = Path("art/.sprite-cache")

# Pose filenames per character, keyed by the action that character is taking.
_PLAYER_POSES = ("dropping", "seated", "idle")
_YAKOU_POSES = ("standing", "idle")


# ── layout ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Layout:
    """Frame dimensions in character cells.

    Sized from the terminal so the figures are as large as will fit: every
    extra cell is visible detail. At the row cap a sextant-rendered figure
    resolves 56 * 3 = 168 pixels of height, which is essentially the fixtures'
    ~170px native figure fidelity, so a maximised terminal loses almost
    nothing to sampling.
    """

    width: int = 100
    scene_rows: int = 16

    # Top border, three header rows, two rules, four stat rows, two footer rows,
    # bottom border. Whatever is left over becomes scene.
    CHROME_LINES = 13

    @property
    def inner(self) -> int:
        return self.width - 4

    @classmethod
    def detect(cls, columns: int | None = None, lines: int | None = None) -> Layout:
        """Fit to the terminal, clamped so the frame never wraps or collapses."""
        if columns is None or lines is None:
            size = shutil.get_terminal_size(fallback=(100, 32))
            columns = columns if columns is not None else size.columns
            lines = lines if lines is not None else size.lines
        width = max(80, min(240, columns - 1))
        scene_rows = max(10, min(56, lines - cls.CHROME_LINES))
        return cls(width=width, scene_rows=scene_rows)


DEFAULT_LAYOUT = Layout()


# ── fixture preparation ───────────────────────────────────────────────────


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
    dissolve it, so it is drawn after scaling instead — see :func:`_scaled`.
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
        wanted = {"baku": _PLAYER_POSES, "hal": _PLAYER_POSES, "yakou": _YAKOU_POSES}
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


# ── console ───────────────────────────────────────────────────────────────


def enable_ansi(stream=None) -> None:
    """Prepare the console for escape sequences and box-drawing glyphs.

    Two separate Windows problems are handled here. Virtual-terminal processing
    must be switched on or the escapes print literally, and the default console
    encoding is cp1252, which cannot encode the border or half-block characters
    and would raise ``UnicodeEncodeError`` mid-frame.
    """
    out = stream if stream is not None else sys.stdout
    reconfigure = getattr(out, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):  # pragma: no cover - depends on host stream
            pass
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        # -11 is STD_OUTPUT_HANDLE; 0x0007 sets ENABLE_VIRTUAL_TERMINAL_PROCESSING
        # alongside the processed-output and wrap flags already in use.
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:  # pragma: no cover - depends on host console
        pass


# ── drawing helpers ───────────────────────────────────────────────────────


def _bar(value: float, maximum: float, width: int = 14) -> str:
    filled = 0 if maximum <= 0 else max(0, min(width, round(width * value / maximum)))
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def _background_run(width: int, colour: bool) -> str:
    """A run of blank cells matching the scene's black field."""
    if width <= 0:
        return ""
    if not colour:
        return " " * width
    red, green, blue = SCENE_BACKGROUND
    return f"\x1b[48;2;{red};{green};{blue}m{' ' * width}{_RESET}"


# Scaling three figures per frame costs a tenth of a second or so, which would
# be paid again on every redraw. Layout is fixed for a session and there are at
# most ten distinct sprites, so the scaled results are memoised.
_SCALE_MEMO: dict[tuple[int, int, int, int], Sprite] = {}


def _scaled(sprite: Sprite, columns: int, rows: int, cell: tuple[int, int]) -> Sprite:
    """Scale a figure into its block and rim-light it at final resolution.

    The target grid is ``columns * cell_width`` by ``rows * cell_height`` —
    the subcell grid of the active glyph set. That grid is deliberately
    anisotropic, so the sprite is stretched to fill it rather than fitted to
    it. Display aspect is already handled by :func:`_figure_columns`, which
    chose the column count from the sprite's own shape; preserving aspect
    again here would shrink every figure inside its block and reopen the gaps
    between them.

    One pixel is reserved on each side so the rim has somewhere to land. Rimming
    here rather than during preparation keeps the outline one cell thick at any
    layout size.
    """
    cell_width, cell_height = cell
    key = (id(sprite), columns, rows, cell_height)
    cached = _SCALE_MEMO.get(key)
    if cached is None:
        width, height = columns * cell_width, rows * cell_height
        # A pure shrink, not a blend: box-averaging at this reduction ratio
        # melts the ink into grey, while nearest sampling keeps each pixel an
        # authentic colour from the artwork.
        inner = sprite.shrunk(max(1, width - 2), max(1, height - 2))
        cached = inner.padded(1).rimmed()
        _SCALE_MEMO[key] = cached
    return cached


def _figure_columns(sprite: Sprite | None, cell_rows: int) -> int:
    """Cell width a figure needs at ``cell_rows`` tall, preserving its shape.

    A terminal cell is about twice as tall as it is wide, so a figure occupying
    ``cell_rows`` rows must span ``2 * cell_rows * aspect`` columns to keep its
    proportions. Deriving width this way rather than centring inside a fixed
    slot is what lets the three figures stand shoulder to shoulder.
    """
    if sprite is None or sprite.height == 0:
        return max(6, cell_rows)
    return max(2, round(2 * cell_rows * sprite.width / sprite.height))


def _sprite_block(
    sprite: Sprite | None,
    label: str,
    columns: int,
    scene_rows: int,
    *,
    pose: str,
    colour: bool,
    glyphs: str = "sextant",
    lift: int = 0,
) -> list[str]:
    """Render one figure into a block of the scene field.

    The figure is scaled to ``POSE_SCALE[pose]`` of the block height and
    bottom-anchored, so a seated player is genuinely shorter than a standing one
    and all three stand on a common floor line. ``lift`` raises the figure that
    many rows off the floor — the depth cue that sets Yakou a step back.
    """
    if sprite is None:
        body = [_background_run(columns, colour)] * scene_rows
        text = f"[{label}]".center(columns)[:columns]
        if colour:
            red, green, blue = SCENE_BACKGROUND
            text = f"\x1b[38;2;200;200;200m\x1b[48;2;{red};{green};{blue}m{text}{_RESET}"
        body[scene_rows // 2] = text
        return body

    figure_rows = max(1, round(scene_rows * POSE_SCALE.get(pose, 1.0)))
    lines = render_cells(
        _scaled(sprite, columns, figure_rows, GLYPH_GRIDS[glyphs]),
        columns,
        figure_rows,
        colour=colour,
        paper=SCENE_BACKGROUND,
        prescaled=True,
        glyphs=glyphs,
    )
    lift = max(0, min(lift, scene_rows - len(lines)))
    pad = scene_rows - len(lines) - lift
    ground = _background_run(columns, colour)
    return [ground] * pad + lines + [ground] * lift


def _scene(
    dropper: Player,
    checker: Player,
    art: SceneArt,
    *,
    frame: int,
    colour: bool,
    layout: Layout,
    glyphs: str = "sextant",
) -> list[str]:
    """Stage the half-round as panel ``stl1`` does.

    Left to right: the Checker seated in the chair, Yakou standing between them,
    and the Dropper on the right holding the handkerchief inward. Only Yakou
    animates — the players hold the pose of the action they are taking, while
    the referee cycles his four-frame idle sheet.

    The trio occupies roughly :data:`SCENE_FILL` of the band and is centred in
    the black field, as the panel frames its figures small against the night.
    If even that does not fit a narrow frame, the whole group scales down
    together — a figure is never squeezed out of proportion to make room.

    The set itself never moves between halves. Each slot is sized for
    whichever of the two players is the wider occupant, so when roles swap the
    players trade places while Yakou and the chair hold their positions.
    """
    scene_rows = layout.scene_rows
    cast = (
        (art.for_action(checker.name, "seated"), checker.name, "seated"),
        (
            art.frame("yakou", "idle", frame) or art.frame("yakou", "standing"),
            "Yakou",
            "standing",
        ),
        (art.for_action(dropper.name, "dropping"), dropper.name, "dropping"),
    )
    players = (checker.name, dropper.name)
    occupants = (
        tuple((art.for_action(name, "seated"), name) for name in players),
        ((cast[1][0], "Yakou"),),
        tuple((art.for_action(name, "dropping"), name) for name in players),
    )

    # Each figure claims only the width its own shape needs at its own height,
    # so the three end up shoulder to shoulder as they are in the panel rather
    # than marooned in the middle of three equal slots.
    def figure_width(sprite: Sprite | None, label: str, rows: int) -> int:
        width = _figure_columns(sprite, rows)
        if sprite is None:
            width = max(width, len(label) + 2)
        return width

    def slot_widths(band: int) -> list[int]:
        widths = []
        for slot, (_, _, pose) in zip(occupants, cast):
            rows = max(1, round(band * POSE_SCALE.get(pose, 1.0)))
            widths.append(max(figure_width(sprite, label, rows) for sprite, label in slot))
        return widths

    gutter = SCENE_GUTTER
    band_rows = min(scene_rows, max(3, round(scene_rows * SCENE_FILL)))
    widths = slot_widths(band_rows)
    while sum(widths) + gutter * 2 > layout.inner and band_rows > 3:
        band_rows -= 1
        widths = slot_widths(band_rows)
    overflow = sum(widths) + gutter * 2 - layout.inner
    if overflow > 0:
        # Even the smallest band does not fit; shed columns as a last resort.
        for _ in range(overflow):
            widest = widths.index(max(widths))
            widths[widest] = max(2, widths[widest] - 1)

    blocks = []
    for (sprite, label, pose), slot in zip(cast, widths):
        rows = max(1, round(band_rows * POSE_SCALE.get(pose, 1.0)))
        lift = round(band_rows * POSE_LIFT.get(pose, 0.0))
        width = min(slot, figure_width(sprite, label, rows))
        block = _sprite_block(
            sprite, label, width, band_rows, pose=pose, colour=colour, glyphs=glyphs, lift=lift
        )
        # The current occupant is centred inside its fixed slot, never
        # stretched to fill it — slot width is layout, not figure shape.
        pad_left = (slot - width) // 2
        pad_right = slot - width - pad_left
        if pad_left or pad_right:
            left_run = _background_run(pad_left, colour)
            right_run = _background_run(pad_right, colour)
            block = [f"{left_run}{line}{right_run}" for line in block]
        blocks.append(block)

    visible = sum(widths) + gutter * 2
    lead_width = max(0, (layout.inner - visible) // 2)
    trail_width = max(0, layout.inner - visible - lead_width)
    lead = _background_run(lead_width, colour)
    trail = _background_run(trail_width, colour)
    gap = _background_run(gutter, colour)
    left, middle, right = blocks
    band = [
        f"{lead}{left[index]}{gap}{middle[index]}{gap}{right[index]}{trail}"
        for index in range(band_rows)
    ]
    top_pad = (scene_rows - band_rows) // 2
    field = _background_run(layout.inner, colour)
    return [field] * top_pad + band + [field] * (scene_rows - band_rows - top_pad)


def _player_column(player: Player, *, tag: str, width: int) -> list[str]:
    name = f"{player.name.upper()} {tag}".strip()
    return [
        name[:width].ljust(width),
        f"vial   {_bar(player.cylinder, CYLINDER_MAX)} {int(player.cylinder):>3}/{CYLINDER_MAX}"[:width].ljust(width),
        f"TTD    {_bar(player.ttd, TOTAL_TTD_MAX)} {int(player.ttd):>3}/{TOTAL_TTD_MAX}"[:width].ljust(width),
        f"deaths {player.deaths}"[:width].ljust(width),
    ]


def _row(text: str, inner: int, *, pad: bool = False) -> str:
    """Wrap one line in the frame border.

    ``pad`` marks lines that carry ANSI escapes, whose display width is not
    their string length; those are padded by the caller's cell count instead.
    """
    if pad:
        return f"║ {text} ║"
    return f"║ {text[:inner].ljust(inner)} ║"


def _stat_rows(game: Game, human_name: str, inner: int) -> list[str]:
    column_width = (inner - 3) // 2
    left = _player_column(
        game.player1,
        tag="(you)" if game.player1.name == human_name else "",
        width=column_width,
    )
    right = _player_column(
        game.player2,
        tag="(you)" if game.player2.name == human_name else "",
        width=column_width,
    )
    return [_row(f"{a} │ {b}", inner) for a, b in zip(left, right)]


def format_result(record: HalfRoundRecord) -> str:
    """One-line summary of a resolved half-round."""
    text = _RESULT_TEXT.get(record.result, record.result.value)
    detail = f" (ST {int(record.st_gained)})" if record.st_gained else ""
    return (
        f"R{record.round_num + 1} H{record.half}  "
        f"{record.dropper} dropped {record.drop_time} · "
        f"{record.checker} checked {record.check_time} → {text}{detail}"
    )


# ── screens ───────────────────────────────────────────────────────────────


def render_frame(
    game: Game,
    *,
    art: SceneArt,
    human_name: str = "Baku",
    frame: int = 0,
    layout: Layout = DEFAULT_LAYOUT,
    colour: bool = True,
    glyphs: str = "sextant",
) -> list[str]:
    """Build the live interface as a list of lines. Pure read of engine state.

    Shows live state only. The Dropper's second is secret until the half-round
    resolves, so nothing about the selections appears here; the reveal belongs
    to :func:`render_outcome`.
    """
    dropper, checker = game.get_roles_for_half(game.current_half)
    inner = layout.inner

    title = "SURPASSING THE LEADER"
    clock = game.format_game_clock()
    header = f"{title}{clock.rjust(inner - len(title))}"

    status = f"Round {game.round_num + 1} · Half {game.current_half}"
    if game.is_leap_second_turn():
        status += "  ⚠ LEAP WINDOW"
    roles = f"DROPPER {dropper.name}    CHECKER {checker.name}"

    span = "═" * (layout.width - 2)
    top, rule, bottom = f"╔{span}╗", f"╠{span}╣", f"╚{span}╝"

    lines = [top, _row(header, inner), _row(status, inner), _row(roles, inner), rule]
    lines.extend(
        _row(line, inner, pad=True)
        for line in _scene(
            dropper, checker, art, frame=frame, colour=colour, layout=layout, glyphs=glyphs
        )
    )
    lines.append(rule)
    lines.extend(_stat_rows(game, human_name, inner))
    lines.append(rule)
    lines.append(_row(f"{dropper.name} drops · {checker.name} checks", inner))
    lines.append(bottom)
    return lines


def render_outcome(
    record: HalfRoundRecord,
    game: Game,
    *,
    human_name: str = "Baku",
    layout: Layout = DEFAULT_LAYOUT,
    colour: bool = True,
) -> list[str]:
    """Build the between-halves screen for the half-round that just resolved.

    Only the most recent half-round is shown; no history is retained.
    """
    inner = layout.inner
    span = "═" * (layout.width - 2)
    top, rule, bottom = f"╔{span}╗", f"╠{span}╣", f"╚{span}╝"

    heading = f"ROUND {record.round_num + 1} · HALF {record.half}"
    heading = f"{heading}{game.format_game_clock().rjust(inner - len(heading))}"

    body = [
        f"{record.dropper} dropped at second {record.drop_time}",
        f"{record.checker} checked at second {record.check_time}",
        "",
        _RESULT_TEXT.get(record.result, record.result.value),
    ]
    if record.st_gained:
        # Inclusive elapsed time, spelled out so the rule is visible in play.
        body.append(
            f"squandered time  ST = {record.check_time} - {record.drop_time} + 1"
            f" = {int(record.st_gained)}s into {record.checker}'s vial"
        )
    if record.death_duration:
        body.append("")
        body.append(f"injected dose    {int(record.death_duration)}s")
        if record.survival_probability is not None:
            body.append(f"revival chance   {record.survival_probability * 100:.1f}%")
        body.append(f"revived          {'yes' if record.survived else 'no'}")

    # The body is padded so this screen is exactly as tall as the live frame.
    # Anything else makes the terminal jump between half-rounds. The live frame
    # is scene_rows + 13 lines and this one is body + 11, hence the offset.
    body_height = layout.scene_rows + 2
    body = body[:body_height]
    body.extend([""] * (body_height - len(body)))

    lines = [top, _row(heading, inner), rule]
    lines.extend(_row(line, inner) for line in body)
    lines.append(rule)
    lines.extend(_stat_rows(game, human_name, inner))
    lines.append(rule)
    if game.game_over and game.winner is not None:
        lines.append(_row(f"GAME OVER — {game.winner.name} wins", inner))
    else:
        lines.append(_row("press Enter to continue", inner))
    lines.append(bottom)
    return lines


def render_victory(
    game: Game,
    *,
    art: SceneArt,
    human_name: str = "Baku",
    layout: Layout = DEFAULT_LAYOUT,
    colour: bool = True,
    glyphs: str = "sextant",
) -> list[str]:
    """Build the end-of-game screen: one still frame of the winner, centred.

    No animation — the first frame of the winner's idle sheet is drawn once and
    holds until the process exits. The screen is the same size as the live
    frame — one top border, three header rows, two rules around the scene band,
    four stat rows, one more rule, one footer, and the bottom border — so the
    terminal does not jump on the final beat.
    """
    winner = game.winner
    inner = layout.inner

    title = "SURPASSING THE LEADER"
    clock = game.format_game_clock()
    header = f"{title}{clock.rjust(inner - len(title))}"
    status = "GAME OVER"
    verdict = f"{winner.name.upper()} WINS" if winner is not None else "NO WINNER"

    span = "═" * (layout.width - 2)
    top, rule, bottom = f"╔{span}╗", f"╠{span}╣", f"╚{span}╝"

    sprite = art.for_action(winner.name, "idle") if winner is not None else None
    label = winner.name if winner is not None else "?"
    band_rows = min(layout.scene_rows, max(3, round(layout.scene_rows * SCENE_FILL)))
    columns = _figure_columns(sprite, band_rows)
    if sprite is None:
        columns = max(columns, len(label) + 2)
    columns = min(inner, columns)
    block = _sprite_block(
        sprite, label, columns, band_rows, pose="idle", colour=colour, glyphs=glyphs
    )
    lead_width = max(0, (inner - columns) // 2)
    trail_width = max(0, inner - columns - lead_width)
    lead = _background_run(lead_width, colour)
    trail = _background_run(trail_width, colour)
    top_pad = (layout.scene_rows - band_rows) // 2
    field = _background_run(inner, colour)
    scene = (
        [field] * top_pad
        + [f"{lead}{line}{trail}" for line in block]
        + [field] * (layout.scene_rows - band_rows - top_pad)
    )

    lines = [top, _row(header, inner), _row(status, inner), _row(verdict, inner), rule]
    lines.extend(_row(line, inner, pad=True) for line in scene)
    lines.append(rule)
    lines.extend(_stat_rows(game, human_name, inner))
    lines.append(rule)
    footer = f"{winner.name} wins the match" if winner is not None else "match over"
    lines.append(_row(footer, inner))
    lines.append(bottom)
    return lines


def draw(lines: list[str], *, stream=None) -> None:
    """Clear the screen and paint a frame."""
    out = stream if stream is not None else sys.stdout
    out.write("\x1b[H\x1b[2J")
    out.write("\n".join(lines))
    out.write("\n")
    out.flush()

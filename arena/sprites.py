"""Terminal sprite rendering for the arena play surface.

Display only. Nothing in this module reads, derives, or mutates canonical game
state; it turns PNG files into character cells and nothing else.

PNG decoding is stdlib-only (``zlib``) because the project carries no image
dependency and the arena must not add one. Only the subset the art pipeline
actually produces is supported: 8-bit non-interlaced greyscale, RGB, indexed,
and their alpha variants.

Sprites live in the repository's ``art/`` tree, which is gitignored, so every
entry point here must tolerate a missing file rather than fail the play loop.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

# Darkest-to-lightest glyphs for terminals without truecolor. Which end of the
# ramp a pixel maps to depends on the field it is drawn on; see _cell_ascii.
_ASCII_RAMP = "@%#*+=-:. "

_RESET = "\x1b[0m"


def luma(red: float, green: float, blue: float) -> float:
    """Perceptual brightness, used for keying, rimming, and glyph choice."""
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


class SpriteError(ValueError):
    """Raised when a PNG cannot be decoded by the stdlib reader."""


@dataclass(frozen=True, slots=True)
class Sprite:
    """An RGBA image as a row-major tuple of ``(r, g, b, a)`` pixels."""

    width: int
    height: int
    rows: tuple[tuple[tuple[int, int, int, int], ...], ...]

    def crop(self, x: int, y: int, width: int, height: int) -> Sprite:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x0 + width), min(self.height, y0 + height)
        if x1 <= x0 or y1 <= y0:
            raise SpriteError(f"empty crop {(x, y, width, height)} from {self.width}x{self.height}")
        rows = tuple(row[x0:x1] for row in self.rows[y0:y1])
        return Sprite(x1 - x0, y1 - y0, rows)

    def opaque_bounds(self) -> tuple[int, int, int, int] | None:
        """Bounding box ``(x0, y0, x1, y1)`` of the opaque pixels, or ``None``.

        Exposed separately from :meth:`trimmed` so a sprite sheet's frames can
        be cropped to the union of their bounds and keep one shared canvas.
        """
        xs_min, xs_max, ys_min, ys_max = self.width, -1, self.height, -1
        for y, row in enumerate(self.rows):
            for x, px in enumerate(row):
                if px[3] >= 128:
                    xs_min, xs_max = min(xs_min, x), max(xs_max, x)
                    ys_min, ys_max = min(ys_min, y), max(ys_max, y)
        if xs_max < 0:
            return None
        return (xs_min, ys_min, xs_max, ys_max)

    def trimmed(self) -> Sprite:
        """Drop fully transparent margins so framing is driven by the artwork."""
        bounds = self.opaque_bounds()
        if bounds is None:
            return self
        x0, y0, x1, y1 = bounds
        return self.crop(x0, y0, x1 - x0 + 1, y1 - y0 + 1)

    def frames(self, count: int) -> tuple[Sprite, ...]:
        """Slice a horizontal sprite sheet into ``count`` equal-width frames."""
        if count < 1 or self.width % count:
            raise SpriteError(f"cannot split width {self.width} into {count} frames")
        step = self.width // count
        return tuple(self.crop(index * step, 0, step, self.height) for index in range(count))

    def mirrored(self) -> Sprite:
        """Flip horizontally. The scene stages every figure facing left."""
        return Sprite(self.width, self.height, tuple(tuple(reversed(row)) for row in self.rows))

    def padded(self, margin: int) -> Sprite:
        """Add a transparent border.

        :meth:`trimmed` removes every transparent pixel, leaving no room for
        :meth:`rimmed` to draw into, so a margin is restored before rimming.
        """
        if margin <= 0:
            return self
        clear = (0, 0, 0, 0)
        width = self.width + margin * 2
        blank = tuple([clear] * width)
        side = tuple([clear] * margin)
        rows = [blank] * margin
        rows.extend(side + row + side for row in self.rows)
        rows.extend([blank] * margin)
        return Sprite(width, self.height + margin * 2, tuple(rows))

    def keyed(self, threshold: int = 236) -> Sprite:
        """Clear the paper by flood-filling bright pixels in from the border.

        These fixtures are opaque RGB on white paper, so a plain brightness
        threshold would also punch holes through Baku's white coat and the
        handkerchief. Filling only from the border keeps enclosed white areas
        intact, because the figure's own outline blocks the fill.

        Matching on luma rather than exact white means this no longer depends on
        a prior quantisation pass, which is what let the tone snapping go.
        """
        rows = [[list(px) for px in row] for row in self.rows]
        stack: list[tuple[int, int]] = []
        for x in range(self.width):
            stack.append((x, 0))
            stack.append((x, self.height - 1))
        for y in range(self.height):
            stack.append((0, y))
            stack.append((self.width - 1, y))
        seen = set()
        while stack:
            x, y = stack.pop()
            if not (0 <= x < self.width and 0 <= y < self.height) or (x, y) in seen:
                continue
            seen.add((x, y))
            pixel = rows[y][x]
            if pixel[3] == 0:
                continue
            if luma(pixel[0], pixel[1], pixel[2]) < threshold:
                continue
            pixel[3] = 0
            stack.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))
        return Sprite(self.width, self.height, tuple(tuple(tuple(px) for px in row) for row in rows))

    def rimmed(self, colour: tuple[int, int, int] = (110, 110, 110), dark_below: int = 90) -> Sprite:
        """Light the silhouette edge wherever the figure is dark.

        On a black field Hal is 63% near-black ink and Yakou 54%, so their
        silhouettes would dissolve into the background. Outlining only the edges
        of dark masses restores them while leaving Baku's pale coat untouched —
        the same separation the manga panel gets from rim lighting.
        """
        rimmed = [list(row) for row in self.rows]
        for y, row in enumerate(self.rows):
            for x, pixel in enumerate(row):
                if pixel[3] >= 128:
                    continue
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < self.width and 0 <= ny < self.height):
                        continue
                    neighbour = self.rows[ny][nx]
                    if neighbour[3] >= 128 and luma(neighbour[0], neighbour[1], neighbour[2]) < dark_below:
                        rimmed[y][x] = (*colour, 255)
                        break
        return Sprite(self.width, self.height, tuple(tuple(row) for row in rimmed))

    def resized(self, width: int, height: int) -> Sprite:
        """Box-average down to a target grid.

        Alpha is averaged separately and used to weight colour, so transparent
        margins never bleed dark pixels into the silhouette edge. Tones are kept
        as-is: the fixtures carry roughly two thousand distinct greys, and
        quantising them onto a five-tone ramp is what made the figures look
        washed out.
        """
        if width < 1 or height < 1:
            raise SpriteError(f"invalid target size {width}x{height}")
        rows = []
        for ty in range(height):
            y0 = ty * self.height // height
            y1 = max(y0 + 1, (ty + 1) * self.height // height)
            row = []
            for tx in range(width):
                x0 = tx * self.width // width
                x1 = max(x0 + 1, (tx + 1) * self.width // width)
                acc_r = acc_g = acc_b = acc_a = 0.0
                count = 0
                for sy in range(y0, y1):
                    src = self.rows[sy]
                    for sx in range(x0, x1):
                        r, g, b, a = src[sx]
                        weight = a / 255.0
                        acc_r += r * weight
                        acc_g += g * weight
                        acc_b += b * weight
                        acc_a += a
                        count += 1
                alpha = acc_a / count if count else 0.0
                if acc_a <= 0.0:
                    row.append((0, 0, 0, 0))
                    continue
                scale = acc_a / 255.0
                row.append(
                    (
                        min(255, round(acc_r / scale)),
                        min(255, round(acc_g / scale)),
                        min(255, round(acc_b / scale)),
                        255 if alpha >= 128 else 0,
                    )
                )
            rows.append(tuple(row))
        return Sprite(width, height, tuple(rows))


    def shrunk(self, width: int, height: int) -> Sprite:
        """Nearest-pixel resample: a pure shrink with no tone blending.

        Box-averaging (:meth:`resized`) mixes ink and ground into intermediate
        greys, which reads as mush at the reduction ratios small figures need.
        Sampling the nearest source pixel instead keeps every output pixel an
        authentic colour from the artwork, so pixel-art edges stay hard at any
        size.
        """
        if width < 1 or height < 1:
            raise SpriteError(f"invalid target size {width}x{height}")
        rows = []
        for ty in range(height):
            src = self.rows[min(self.height - 1, ((2 * ty + 1) * self.height) // (2 * height))]
            rows.append(
                tuple(
                    src[min(self.width - 1, ((2 * tx + 1) * self.width) // (2 * width))]
                    for tx in range(width)
                )
            )
        return Sprite(width, height, tuple(rows))


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def decode_png(path: str | Path, max_edge: int | None = None) -> Sprite:
    """Decode a PNG into a :class:`Sprite` using only the standard library.

    ``max_edge`` decimates while decoding: rows and columns are sampled so the
    longest side lands near ``max_edge``. Row filters are sequential, so every
    row is still unfiltered, but only the sampled ones are expanded into
    pixels. On the ~1250px fixtures that is the difference between roughly two
    seconds per sprite and a tenth of one, and the sources are upscaled pixel
    art so the sampling discards nothing.
    """
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise SpriteError(f"{path} is not a PNG")

    header: tuple[int, ...] | None = None
    palette = b""
    transparency = b""
    idat = bytearray()
    offset = 8
    while offset + 8 <= len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        tag = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + length]
        if tag == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif tag == b"PLTE":
            palette = body
        elif tag == b"tRNS":
            transparency = body
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
        offset += 12 + length

    if header is None:
        raise SpriteError(f"{path} has no IHDR chunk")
    width, height, depth, colour_type, _, _, interlace = header
    if interlace:
        raise SpriteError(f"{path} is interlaced, which is unsupported")
    if depth not in (1, 2, 4, 8):
        raise SpriteError(f"{path} has unsupported bit depth {depth}")
    if colour_type not in (0, 2, 3, 4, 6):
        raise SpriteError(f"{path} has unsupported colour type {colour_type}")
    if colour_type != 3 and depth != 8:
        raise SpriteError(f"{path} has unsupported depth {depth} for colour type {colour_type}")

    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[colour_type]
    bits = width * channels * depth
    stride = (bits + 7) // 8
    step = max(1, channels * depth // 8)

    raw = zlib.decompress(bytes(idat))
    if len(raw) < height * (stride + 1):
        raise SpriteError(f"{path} has truncated image data")

    out_width, out_height = width, height
    if max_edge is not None and max(width, height) > max_edge:
        scale = max_edge / max(width, height)
        out_width = max(1, round(width * scale))
        out_height = max(1, round(height * scale))
    wanted = {(ty * height) // out_height: ty for ty in range(out_height)}
    columns = [(tx * width) // out_width for tx in range(out_width)] if out_width != width else None

    rows: list[tuple[tuple[int, int, int, int], ...] | None] = [None] * out_height
    previous = bytearray(stride)
    for y in range(height):
        base = y * (stride + 1)
        filter_type = raw[base]
        line = bytearray(raw[base + 1 : base + 1 + stride])
        if filter_type == 1:
            for i in range(step, stride):
                line[i] = (line[i] + line[i - step]) & 0xFF
        elif filter_type == 2:
            line = bytearray([(a + b) & 0xFF for a, b in zip(line, previous)])
        elif filter_type == 3:
            for i in range(stride):
                left = line[i - step] if i >= step else 0
                line[i] = (line[i] + ((left + previous[i]) >> 1)) & 0xFF
        elif filter_type == 4:
            for i in range(stride):
                left = line[i - step] if i >= step else 0
                up = previous[i]
                up_left = previous[i - step] if i >= step else 0
                delta = left + up - up_left
                pa, pb, pc = abs(delta - left), abs(delta - up), abs(delta - up_left)
                if pa <= pb and pa <= pc:
                    predictor = left
                elif pb <= pc:
                    predictor = up
                else:
                    predictor = up_left
                line[i] = (line[i] + predictor) & 0xFF
        elif filter_type != 0:
            raise SpriteError(f"{path} uses unknown row filter {filter_type}")
        target = wanted.get(y)
        if target is not None:
            expanded = _expand_row(line, width, depth, colour_type, palette, transparency)
            rows[target] = expanded if columns is None else tuple(expanded[x] for x in columns)
        previous = line

    if any(row is None for row in rows):
        raise SpriteError(f"{path} produced an incomplete image")
    return Sprite(out_width, out_height, tuple(rows))


def _expand_row(
    line: bytearray,
    width: int,
    depth: int,
    colour_type: int,
    palette: bytes,
    transparency: bytes,
) -> tuple[tuple[int, int, int, int], ...]:
    """Turn one decoded scanline into RGBA pixels."""
    if colour_type == 3:
        indices = _unpack_indices(line, width, depth)
        out = []
        for index in indices:
            base = index * 3
            rgb = (palette[base], palette[base + 1], palette[base + 2])
            alpha = transparency[index] if index < len(transparency) else 255
            out.append((*rgb, alpha))
        return tuple(out)

    out = []
    for x in range(width):
        if colour_type == 0:
            value = line[x]
            out.append((value, value, value, 255))
        elif colour_type == 4:
            value = line[x * 2]
            out.append((value, value, value, line[x * 2 + 1]))
        elif colour_type == 2:
            base = x * 3
            out.append((line[base], line[base + 1], line[base + 2], 255))
        else:
            base = x * 4
            out.append((line[base], line[base + 1], line[base + 2], line[base + 3]))
    return tuple(out)


def _unpack_indices(line: bytearray, width: int, depth: int) -> list[int]:
    if depth == 8:
        return list(line[:width])
    per_byte = 8 // depth
    mask = (1 << depth) - 1
    indices: list[int] = []
    for byte in line:
        for slot in range(per_byte):
            shift = 8 - depth * (slot + 1)
            indices.append((byte >> shift) & mask)
            if len(indices) == width:
                return indices
    return indices[:width]


def write_png(sprite: Sprite, path: str | Path) -> None:
    """Write a sprite as an 8-bit RGBA PNG with no row filtering.

    Used only for the prepared-frame cache, where the images are small and
    decode speed matters far more than file size.
    """
    raw = bytearray()
    for row in sprite.rows:
        raw.append(0)  # filter: none, the cheapest to read back
        for red, green, blue, alpha in row:
            raw += bytes((red, green, blue, alpha))

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + tag
            + body
            + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
        )

    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", sprite.width, sprite.height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


def load_sprite(path: str | Path, max_edge: int | None = None) -> Sprite | None:
    """Decode a sprite, returning ``None`` when the art is absent or unreadable.

    The art tree is gitignored, so a missing sprite is an ordinary condition
    and must never break the play loop.
    """
    try:
        return decode_png(path, max_edge)
    except (OSError, SpriteError, zlib.error, IndexError):
        return None


# Subcell grid resolved by one character cell, per glyph set. Sextants come
# from Symbols for Legacy Computing (Unicode 13); Windows Terminal and current
# Cascadia fonts cover them, but quadrants stay available for fonts that do not.
GLYPH_GRIDS: dict[str, tuple[int, int]] = {
    "sextant": (2, 3),
    "quadrant": (2, 2),
}


def render_cells(
    sprite: Sprite,
    columns: int,
    rows: int,
    *,
    colour: bool = True,
    paper: tuple[int, int, int] | None = None,
    prescaled: bool = False,
    glyphs: str = "sextant",
) -> list[str]:
    """Render a sprite into ``rows`` lines of ``columns`` character cells.

    Each cell resolves a block of pixels, not a single sample, and is drawn
    with whichever glyph of the chosen set exactly matches that block's shape.
    ``sextant`` resolves 2x3 pixels per cell — half again the vertical
    resolution of the Block Elements quadrants — which is the highest-fidelity
    sampling a cell can carry without leaving well-supported Unicode; octants
    exist but their font coverage is still too thin to be a default. Keeping
    per-cell shapes exact instead of averaging is the difference between hard
    silhouette edges and grey mush, so the sampling, rather than the source
    art, stops being the limit on how sharp the figures look.

    ``paper`` fills transparent pixels with a solid colour instead of letting
    them fall through to the terminal background, so the scene reads as one
    continuous field rather than three floating cut-outs.

    ``prescaled`` skips the internal resize for a sprite already laid out at
    ``columns * cell_width`` x ``rows * cell_height``, as the scene renderer
    supplies.
    """
    if columns < 1 or rows < 1:
        return []
    cell_width, cell_height = GLYPH_GRIDS[glyphs]
    glyph = _GLYPH_SHAPES[glyphs]
    dark_field = paper is not None and luma(*paper) < 128
    grid = sprite if prescaled else sprite.resized(columns * cell_width, rows * cell_height)
    fill = None if paper is None else (*paper, 255)
    lines: list[str] = []
    for row in range(rows):
        band = grid.rows[row * cell_height : (row + 1) * cell_height]
        parts: list[str] = []
        for column in range(columns):
            left = column * cell_width
            pixels = tuple(
                band_row[left + dx] for band_row in band for dx in range(cell_width)
            )
            if fill is not None:
                pixels = tuple(px if px[3] >= 128 else fill for px in pixels)
            parts.append(
                _cell_colour(pixels, glyph) if colour else _cell_ascii(pixels, dark_field)
            )
        line = "".join(parts)
        lines.append(f"{line}{_RESET}" if colour else line)
    return lines


# Every 2x2 on/off pattern has an exact Block Elements glyph, so a cell can
# always be drawn as two colours plus a shape with no approximation of the
# shape itself. Bits are top-left, top-right, bottom-left, bottom-right.
_QUADRANTS = {
    0b0000: " ",
    0b1000: "▘",
    0b0100: "▝",
    0b1100: "▀",
    0b0010: "▖",
    0b1010: "▌",
    0b0110: "▞",
    0b1110: "▛",
    0b0001: "▗",
    0b1001: "▚",
    0b0101: "▐",
    0b1101: "▜",
    0b0011: "▄",
    0b1011: "▙",
    0b0111: "▟",
    0b1111: "█",
}


def _quadrant_glyph(flags: tuple[bool, ...]) -> str:
    bits = sum(1 << (3 - index) for index, lit in enumerate(flags) if lit)
    return _QUADRANTS[bits]


# Sextant patterns as bit values, least significant bit first in row-major
# order: top-left, top-right, mid-left, mid-right, bottom-left, bottom-right.
_LEFT_COLUMN = 0b010101
_RIGHT_COLUMN = 0b101010
_ALL_SIX = 0b111111


def _sextant_glyph(flags: tuple[bool, ...]) -> str:
    """The Block Sextant glyph for a 2x3 on/off pattern.

    U+1FB00..U+1FB3B enumerate every pattern in bit order except the four that
    already exist elsewhere — empty, the two half blocks, and full — so those
    are special-cased and the rest of the range is indexed by skipping them.
    """
    bits = sum(1 << index for index, lit in enumerate(flags) if lit)
    if bits == 0:
        return " "
    if bits == _LEFT_COLUMN:
        return "▌"
    if bits == _RIGHT_COLUMN:
        return "▐"
    if bits == _ALL_SIX:
        return "█"
    return chr(0x1FB00 + bits - 1 - (bits > _LEFT_COLUMN) - (bits > _RIGHT_COLUMN))


_GLYPH_SHAPES = {
    "sextant": _sextant_glyph,
    "quadrant": _quadrant_glyph,
}


def _split(pixels) -> tuple[tuple[bool, ...], list, list]:
    """Split a cell's pixels into a bright and a dark group.

    Thresholding at the cell's own midpoint rather than a global value is what
    preserves local contrast: an edge inside a dark mass still resolves.
    """
    lit = [luma(px[0], px[1], px[2]) for px in pixels]
    low, high = min(lit), max(lit)
    middle = (low + high) / 2
    flags: list[bool] = []
    bright: list = []
    dark: list = []
    for pixel, value in zip(pixels, lit):
        if value > middle:
            flags.append(True)
            bright.append(pixel)
        else:
            flags.append(False)
            dark.append(pixel)
    return tuple(flags), bright, dark


def _average(pixels, fallback) -> tuple[int, int, int]:
    if not pixels:
        return fallback
    count = len(pixels)
    return (
        sum(px[0] for px in pixels) // count,
        sum(px[1] for px in pixels) // count,
        sum(px[2] for px in pixels) // count,
    )


def _cell_colour(pixels, glyph) -> str:
    """Draw one cell as a shape glyph over a foreground/background pair."""
    if all(px[3] < 128 for px in pixels):
        return f"{_RESET} "
    flags, bright, dark = _split(pixels)
    foreground = _average(bright, _average(dark, (0, 0, 0)))
    background = _average(dark, foreground)
    return (
        f"\x1b[38;2;{foreground[0]};{foreground[1]};{foreground[2]}m"
        f"\x1b[48;2;{background[0]};{background[1]};{background[2]}m"
        f"{glyph(flags)}"
    )


def _cell_ascii(pixels, dark_field: bool = False) -> str:
    """Average the cell's pixels onto the ASCII density ramp.

    On a dark field the mapping inverts: bright pixels become the dense glyphs.
    Without this the fallback renders as a photographic negative, because ink
    that should read as shadow would be drawn as the heaviest character against
    an already-black background.
    """
    samples = [px for px in pixels if px[3] >= 128]
    if not samples:
        return " "
    brightness = sum(luma(px[0], px[1], px[2]) for px in samples) / len(samples)
    if dark_field:
        brightness = 255.0 - brightness
    index = min(len(_ASCII_RAMP) - 1, int(brightness / 256 * len(_ASCII_RAMP)))
    return _ASCII_RAMP[index]

"""Tests for the arena terminal interface.

The interface is a display layer, so the contract under test is that it reports
canonical engine state faithfully and never changes it.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from arena.sprites import (
    Sprite,
    SpriteError,
    _sextant_glyph,
    decode_png,
    load_sprite,
    luma,
    render_cells,
    write_png,
)
from arena.tui import (
    IDLE_FRAMES,
    POSE_SCALE,
    Layout,
    SceneArt,
    _figure_columns,
    _sprite_block,
    format_result,
    render_frame,
    render_outcome,
    render_victory,
)
from stl.engine.game import (
    LS_WINDOW_START,
    OPENING_START_CLOCK,
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
    TOTAL_TTD_MAX,
    Game,
    Player,
    Referee,
)


def _game(*, baku_first: bool = False, clock: int = OPENING_START_CLOCK) -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    first, second = (baku, hal) if baku_first else (hal, baku)
    game = Game(player1=first, player2=second, referee=Referee())
    game.game_clock = clock
    return game


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _write_png(path: Path, pixels: list[list[tuple[int, int, int, int]]]) -> Path:
    """Write a minimal 8-bit RGBA PNG with no row filtering."""
    height, width = len(pixels), len(pixels[0])
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for red, green, blue, alpha in row:
            raw += bytes((red, green, blue, alpha))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _chunk(b"IEND", b"")
    )
    return path


# ── display fidelity ──────────────────────────────────────────────────────


def test_frame_reports_every_required_field() -> None:
    game = _game()
    game.player1.cylinder, game.player1.ttd, game.player1.deaths = 120, 61, 1
    text = "\n".join(render_frame(game, art=SceneArt(), colour=False))

    assert game.format_game_clock() in text          # game clock
    assert "Round 1" in text and "Half 1" in text    # round number
    assert "DROPPER Hal" in text                     # current dropper
    assert "CHECKER Baku" in text                    # current checker
    assert "120/300" in text                         # cylinder
    assert " 61/300" in text                         # TTD
    assert "deaths 1" in text                        # deaths


def test_live_frame_hides_the_selections_and_the_last_result() -> None:
    """The Dropper's second is secret until the half-round resolves."""
    game = _game()
    record = game.play_half_round(34, 41)
    text = "\n".join(render_frame(game, art=SceneArt(), colour=False))
    assert "34" not in text
    assert "check success" not in text.lower()
    assert record.drop_time == 34  # the reveal belongs to the outcome screen


def test_header_omits_legal_ranges_and_turn_duration() -> None:
    game = _game()
    text = "\n".join(render_frame(game, art=SceneArt(), colour=False))
    assert "legal" not in text
    assert "turn 60" not in text


def test_result_line_uses_inclusive_squandered_time() -> None:
    game = _game()
    record = game.play_half_round(10, 10)
    assert record.st_gained == 1
    assert "ST 1" in format_result(record)


def test_leap_window_is_flagged_in_the_header() -> None:
    game = _game(baku_first=True, clock=LS_WINDOW_START)
    text = "\n".join(render_frame(game, art=SceneArt(), colour=False))
    assert "LEAP WINDOW" in text
    assert "DROPPER Baku" in text and "CHECKER Hal" in text


def test_outside_the_leap_window_no_badge_is_shown() -> None:
    game = _game(baku_first=True)
    text = "\n".join(render_frame(game, art=SceneArt(), colour=False))
    assert "LEAP WINDOW" not in text


@pytest.mark.parametrize("layout", [Layout(80, 10), Layout(100, 16), Layout(160, 40)])
def test_frame_lines_share_one_width_so_the_border_aligns(layout: Layout) -> None:
    game = _game()
    lines = render_frame(game, art=SceneArt(), layout=layout, colour=False)
    assert {len(line) for line in lines} == {layout.width}


# ── the outcome screen ────────────────────────────────────────────────────


def test_outcome_reveals_both_seconds_and_the_verdict() -> None:
    game = _game()
    record = game.play_half_round(34, 41)
    text = "\n".join(render_outcome(record, game, colour=False))
    assert "dropped at second 34" in text
    assert "checked at second 41" in text
    assert "CHECK SUCCESS" in text


def test_outcome_spells_out_inclusive_squandered_time() -> None:
    game = _game()
    record = game.play_half_round(34, 41)
    text = "\n".join(render_outcome(record, game, colour=False))
    # ST = check - drop + 1 = 41 - 34 + 1 = 8
    assert "41 - 34 + 1 = 8s" in text


def test_outcome_reports_the_death_and_revival_detail() -> None:
    game = _game()
    record = game.play_half_round(50, 41)  # check before the drop: a failed check
    assert record.death_duration
    text = "\n".join(render_outcome(record, game, colour=False))
    assert "CHECK FAILED" in text
    assert "injected dose" in text
    assert "revival chance" in text
    assert "revived" in text


def test_outcome_announces_game_over() -> None:
    game = _game()
    game.player2.ttd = TOTAL_TTD_MAX  # any further death is unsurvivable
    record = game.play_half_round(50, 41)
    text = "\n".join(render_outcome(record, game, colour=False))
    assert game.game_over
    assert "GAME OVER" in text


@pytest.mark.parametrize("layout", [Layout(80, 10), Layout(120, 24)])
def test_outcome_lines_share_one_width(layout: Layout) -> None:
    game = _game()
    record = game.play_half_round(34, 41)
    lines = render_outcome(record, game, layout=layout, colour=False)
    assert {len(line) for line in lines} == {layout.width}


@pytest.mark.parametrize("layout", [Layout(80, 10), Layout(100, 16), Layout(160, 40)])
def test_outcome_screen_is_the_same_size_as_the_live_frame(layout: Layout) -> None:
    """The terminal must not jump between the round and the summary."""
    game = _game()
    record = game.play_half_round(34, 41)
    live = render_frame(game, art=SceneArt(), layout=layout, colour=False)
    summary = render_outcome(record, game, layout=layout, colour=False)
    assert len(summary) == len(live)
    assert {len(line) for line in summary} == {len(line) for line in live}


def test_outcome_body_is_not_truncated_by_the_padding() -> None:
    """A death report is the longest body; it must still fit the smallest frame."""
    game = _game()
    record = game.play_half_round(50, 41)
    text = "\n".join(render_outcome(record, game, layout=Layout(80, 10), colour=False))
    assert "CHECK FAILED" in text
    assert "injected dose" in text
    assert "revived" in text


# ── the victory screen ────────────────────────────────────────────────────


def _finished_game() -> Game:
    """A game played to completion: Baku's next death is unsurvivable."""
    game = _game()
    game.player2.ttd = TOTAL_TTD_MAX
    game.play_half_round(50, 41)  # failed check kills the checker for good
    assert game.game_over and game.winner is not None
    return game


def test_victory_screen_names_the_winner() -> None:
    game = _finished_game()
    text = "\n".join(render_victory(game, art=SceneArt(), colour=False))
    assert "GAME OVER" in text
    assert f"{game.winner.name.upper()} WINS" in text


@pytest.mark.parametrize("layout", [Layout(80, 10), Layout(100, 16), Layout(160, 40)])
def test_victory_screen_is_the_same_size_as_the_live_frame(layout: Layout) -> None:
    """The final beat must not make the terminal jump either."""
    game = _finished_game()
    live = render_frame(game, art=SceneArt(), layout=layout, colour=False)
    victory = render_victory(game, art=SceneArt(), layout=layout, colour=False)
    assert len(victory) == len(live)
    assert {len(line) for line in victory} == {layout.width}


def test_victory_screen_is_a_still_of_the_first_idle_frame() -> None:
    """No end-of-game animation: exactly one frame of the sheet, held."""
    game = _finished_game()
    character = "hal" if game.winner.name.lower() == "hal" else "baku"
    ink, grey = (0, 0, 0, 255), (122, 122, 122, 255)
    # Frames alternate, so using any frame but the first would show up here.
    sheet = tuple(_block(ink if index % 2 else grey) for index in range(IDLE_FRAMES))
    full = SceneArt({(character, "idle"): sheet})
    still = SceneArt({(character, "idle"): (sheet[0],)})
    assert render_victory(game, art=full, colour=False) == render_victory(
        game, art=still, colour=False
    )


def test_victory_rendering_does_not_mutate_canonical_game_state() -> None:
    game = _finished_game()
    before = game.get_state_summary()
    render_victory(game, art=SceneArt(), colour=False)
    assert game.get_state_summary() == before


# ── the display must not referee ──────────────────────────────────────────


def test_rendering_does_not_mutate_canonical_game_state() -> None:
    game = _game()
    record = game.play_half_round(34, 41)
    before = game.get_state_summary()
    history_length = len(game.history)
    for frame in range(IDLE_FRAMES * 2):
        render_frame(game, art=SceneArt(), frame=frame, colour=False)
        render_outcome(record, game, colour=False)
    assert game.get_state_summary() == before
    assert len(game.history) == history_length


def test_rendering_does_not_advance_the_clock_or_swap_roles() -> None:
    game = _game()
    clock, half, roles = game.game_clock, game.current_half, game.get_roles_for_half(1)
    render_frame(game, art=SceneArt(), colour=False)
    assert (game.game_clock, game.current_half) == (clock, half)
    assert game.get_roles_for_half(1) == roles


# ── degradation when the gitignored art tree is absent ────────────────────


def test_missing_art_renders_placeholders_instead_of_failing(tmp_path: Path) -> None:
    art = SceneArt.load(tmp_path, cache_dir=None)
    assert art.poses == {}
    lines = render_frame(_game(), art=art, colour=False)
    text = "\n".join(lines)
    assert "[Hal]" in text and "[Baku]" in text and "[Yakou]" in text
    assert {len(line) for line in lines} == {Layout().width}


def test_load_sprite_returns_none_for_missing_and_corrupt_files(tmp_path: Path) -> None:
    assert load_sprite(tmp_path / "absent.png") is None
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"\x89PNG\r\n\x1a\nnot actually a png")
    assert load_sprite(corrupt) is None


# ── sprite decoding and cell rendering ────────────────────────────────────


def test_decode_png_round_trips_pixels(tmp_path: Path) -> None:
    pixels = [
        [(0, 0, 0, 255), (255, 255, 255, 255)],
        [(122, 122, 122, 255), (0, 0, 0, 0)],
    ]
    sprite = decode_png(_write_png(tmp_path / "s.png", pixels))
    assert (sprite.width, sprite.height) == (2, 2)
    assert sprite.rows[0][0] == (0, 0, 0, 255)
    assert sprite.rows[1][1][3] == 0


def test_decode_png_rejects_a_non_png(tmp_path: Path) -> None:
    path = tmp_path / "x.png"
    path.write_bytes(b"definitely not a png")
    with pytest.raises(SpriteError):
        decode_png(path)


def test_trimmed_drops_transparent_margins(tmp_path: Path) -> None:
    clear = (0, 0, 0, 0)
    ink = (0, 0, 0, 255)
    pixels = [
        [clear, clear, clear],
        [clear, ink, clear],
        [clear, clear, clear],
    ]
    sprite = decode_png(_write_png(tmp_path / "t.png", pixels))
    trimmed = sprite.trimmed()
    assert (trimmed.width, trimmed.height) == (1, 1)


def test_render_cells_matches_the_requested_grid() -> None:
    solid = tuple(tuple((0, 0, 0, 255) for _ in range(8)) for _ in range(8))
    sprite = Sprite(8, 8, solid)
    lines = render_cells(sprite, 6, 3, colour=False)
    assert len(lines) == 3
    assert all(len(line) == 6 for line in lines)


def test_render_cells_maps_transparency_to_blanks() -> None:
    clear = tuple(tuple((0, 0, 0, 0) for _ in range(4)) for _ in range(4))
    lines = render_cells(Sprite(4, 4, clear), 4, 2, colour=False)
    assert lines == ["    ", "    "]


def test_colour_rendering_emits_truecolor_escapes() -> None:
    solid = tuple(tuple((196, 196, 196, 255) for _ in range(4)) for _ in range(4))
    lines = render_cells(Sprite(4, 4, solid), 4, 2, colour=True)
    assert any("\x1b[38;2;196;196;196m" in line for line in lines)


def test_sextant_glyphs_cover_the_special_cased_patterns() -> None:
    """Empty, full, and the two half blocks live outside the sextant range."""
    assert _sextant_glyph((False,) * 6) == " "
    assert _sextant_glyph((True,) * 6) == "█"
    assert _sextant_glyph((True, False, True, False, True, False)) == "▌"
    assert _sextant_glyph((False, True, False, True, False, True)) == "▐"
    # Top-left pixel alone is BLOCK SEXTANT-1, the first of the range.
    assert _sextant_glyph((True, False, False, False, False, False)) == "\U0001FB00"


def test_sextant_indexing_skips_the_half_block_patterns() -> None:
    """The range omits the two half-block patterns, so indexing must skip them."""

    def flags(bits: int) -> tuple[bool, ...]:
        return tuple(bool(bits >> index & 1) for index in range(6))

    assert _sextant_glyph(flags(20)) == chr(0x1FB13)
    assert _sextant_glyph(flags(22)) == chr(0x1FB14)  # 21 is the left half block
    assert _sextant_glyph(flags(41)) == chr(0x1FB27)
    assert _sextant_glyph(flags(43)) == chr(0x1FB28)  # 42 is the right half block
    assert _sextant_glyph(flags(62)) == chr(0x1FB3B)  # last of the range


def test_sextant_cells_resolve_three_vertical_samples() -> None:
    """One cell must distinguish a lit top third — quadrants cannot see it."""
    bright, dark = (240, 240, 240, 255), (10, 10, 10, 255)
    rows = ((bright, bright), (dark, dark), (dark, dark))
    lines = render_cells(Sprite(2, 3, rows), 1, 1, colour=True, prescaled=True)
    assert chr(0x1FB02) in lines[0]  # BLOCK SEXTANT-12: only the top row lit


def test_quadrant_fallback_still_renders_block_elements() -> None:
    bright, dark = (240, 240, 240, 255), (10, 10, 10, 255)
    rows = ((bright, bright), (dark, dark))
    lines = render_cells(
        Sprite(2, 2, rows), 1, 1, colour=True, prescaled=True, glyphs="quadrant"
    )
    assert "▀" in lines[0]


def test_resize_preserves_tone_instead_of_quantising_it() -> None:
    """Snapping to a five-tone ramp is what made the figures look washed out."""
    shade = (137, 137, 137, 255)
    resized = Sprite(4, 4, tuple(tuple(shade for _ in range(4)) for _ in range(4))).resized(2, 2)
    for row in resized.rows:
        for pixel in row:
            assert pixel[:3] == (137, 137, 137)


def test_ascii_ramp_inverts_on_a_dark_field() -> None:
    """On black, bright pixels must become the dense glyphs, not the sparse ones."""
    bright = tuple(tuple((240, 240, 240, 255) for _ in range(4)) for _ in range(4))
    on_white = render_cells(Sprite(4, 4, bright), 4, 2, colour=False, paper=(255, 255, 255))
    on_black = render_cells(Sprite(4, 4, bright), 4, 2, colour=False, paper=(0, 0, 0))
    assert on_white != on_black
    assert on_black[0].strip(), "a bright figure must not vanish on a dark field"


def _block(fill: tuple[int, int, int, int], size: int = 8) -> Sprite:
    return Sprite(size, size, tuple(tuple(fill for _ in range(size)) for _ in range(size)))


def _tall_block(fill: tuple[int, int, int, int], width: int = 8, height: int = 32) -> Sprite:
    return Sprite(width, height, tuple(tuple(fill for _ in range(width)) for _ in range(height)))


def test_yakou_animates_through_his_four_idle_frames() -> None:
    ink, grey = (0, 0, 0, 255), (122, 122, 122, 255)
    # Frames alternate so a changed frame index must change the rendered scene.
    sheet = tuple(_block(ink if index % 2 else grey) for index in range(IDLE_FRAMES))
    art = SceneArt({("yakou", "idle"): sheet})
    game = _game()
    rendered = {
        "\n".join(render_frame(game, art=art, frame=index, colour=False))
        for index in range(IDLE_FRAMES)
    }
    assert len(rendered) > 1, "the referee's idle cycle should not be static"


def test_scene_is_staged_as_checker_yakou_dropper() -> None:
    """Panel stl1: seated player left, Yakou centre, dropper right."""
    lines = render_frame(_game(), art=SceneArt(), colour=False)
    scene = [line for line in lines if "[Yakou]" in line]
    assert len(scene) == 1
    row = scene[0]
    # Hal drops in half 1, so Baku is the seated checker on the left.
    assert row.index("[Baku]") < row.index("[Yakou]") < row.index("[Hal]")


def test_yakou_and_the_chair_hold_position_between_halves() -> None:
    """The referee and the chair are fixtures of the set; only players move.

    The two players' sprites have different widths, so slots sized to the
    current occupant would re-centre the whole group on every role swap.
    """
    wide = _tall_block((220, 220, 220, 255), width=16, height=32)
    narrow = _tall_block((220, 220, 220, 255), width=8, height=32)
    # Bright enough that only Yakou's interior cells map to the densest glyph.
    marker = _tall_block((250, 250, 250, 255), width=8, height=32)
    art = SceneArt(
        {
            ("baku", "dropping"): (wide,),
            ("baku", "seated"): (wide,),
            ("hal", "dropping"): (narrow,),
            ("hal", "seated"): (narrow,),
            ("yakou", "standing"): (marker,),
        }
    )
    game = _game()
    layout = Layout(100, 24)

    def yakou_columns(lines: list[str]) -> set[int]:
        scene = lines[5 : 5 + layout.scene_rows]
        return {index for line in scene for index, ch in enumerate(line) if ch == "@"}

    first = yakou_columns(render_frame(game, art=art, layout=layout, colour=False))
    game.current_half = 2
    second = yakou_columns(render_frame(game, art=art, layout=layout, colour=False))
    assert first, "the marker figure must be visible"
    assert first == second


def test_the_players_swap_around_yakou_when_roles_swap() -> None:
    game = _game()
    first = [l for l in render_frame(game, art=SceneArt(), colour=False) if "[Yakou]" in l][0]
    game.current_half = 2
    second = [l for l in render_frame(game, art=SceneArt(), colour=False) if "[Yakou]" in l][0]
    assert first.index("[Baku]") < first.index("[Hal]")
    assert second.index("[Hal]") < second.index("[Baku]")


def test_seated_figures_render_shorter_than_standing_ones() -> None:
    """A player in the chair must not be scaled up to a standing player's height."""
    assert POSE_SCALE["seated"] < POSE_SCALE["standing"]
    # Tall and narrow like the real fixtures (~92x175), so height is the binding
    # dimension. A square sprite would be width-bound and the scale could not
    # bite. Pale, because black ink on a black field is legitimately invisible.
    pale = _tall_block((220, 220, 220, 255))
    seated = _sprite_block(pale, "x", 12, 16, pose="seated", colour=False)
    standing = _sprite_block(pale, "x", 12, 16, pose="dropping", colour=False)
    assert len(seated) == len(standing) == 16, "both occupy the same block"
    assert sum(1 for line in seated if line.strip()) < sum(
        1 for line in standing if line.strip()
    )


def test_scene_figures_are_small_and_centred_in_the_field() -> None:
    """The trio takes about a third of the band, framed by black on all sides."""
    pale = _tall_block((220, 220, 220, 255))
    art = SceneArt(
        {
            ("hal", "dropping"): (pale,),
            ("baku", "seated"): (pale,),
            ("yakou", "standing"): (pale,),
        }
    )
    layout = Layout(100, 24)
    lines = render_frame(_game(), art=art, layout=layout, colour=False)
    scene = [line[2:-2] for line in lines[5 : 5 + layout.scene_rows]]
    lit = [index for index, row in enumerate(scene) if row.strip()]
    assert lit, "the figures must be visible"
    # Small: nowhere near filling the band vertically.
    assert len(lit) <= round(layout.scene_rows / 2)
    # Centred vertically: black field above and below the group.
    assert lit[0] >= 4 and lit[-1] <= layout.scene_rows - 4
    # Centred horizontally: the group hugs the middle, not the borders.
    rows = [scene[index] for index in lit]
    leftmost = min(len(row) - len(row.lstrip()) for row in rows)
    rightmost = max(len(row.rstrip()) for row in rows)
    assert leftmost >= layout.inner // 4
    assert rightmost <= layout.inner - layout.inner // 4


def _misaligned_sheet() -> list[list[tuple[int, int, int, int]]]:
    """A 4-frame sheet whose figures ignore the quarter boundaries.

    Frame windows are 10 wide. Figures are 3x4 blocks of distinct tones —
    except the fourth, which is 5 wide and straddles the third cut at x=30,
    exactly the defect in the real sheets.
    """
    white = (255, 255, 255, 255)
    pixels = [[white] * 40 for _ in range(8)]
    tones = [(50, 50, 50, 255), (80, 80, 80, 255), (110, 110, 110, 255), (140, 140, 140, 255)]
    spans = [(2, 4), (12, 14), (21, 23), (28, 32)]
    for tone, (x0, x1) in zip(tones, spans):
        for y in range(2, 6):
            for x in range(x0, x1 + 1):
                pixels[y][x] = tone
    return pixels


def test_figures_straddling_a_quarter_cut_are_recovered_whole(tmp_path: Path) -> None:
    """Slicing at fixed quarters duplicated one figure's edge beside another."""
    (tmp_path / "yakou").mkdir()
    _write_png(tmp_path / "yakou" / "yakou_idle.png", _misaligned_sheet())
    art = SceneArt.load(tmp_path, cache_dir=None)
    frames = art.poses[("yakou", "idle")]
    assert len(frames) == IDLE_FRAMES
    # One shared canvas: as wide as the widest figure, no frame-width margin.
    assert {(frame.width, frame.height) for frame in frames} == {(5, 4)}
    # Each frame holds exactly its own figure: the straddling fourth figure is
    # whole in its frame and leaks into no other.
    for index, tone in enumerate((50, 80, 110, 140)):
        greys = {px[0] for row in frames[index].rows for px in row if px[3] >= 128}
        assert greys == {tone}
    assert sum(1 for row in frames[3].rows for px in row if px[3] >= 128) == 20


def test_sheet_edge_artifact_lines_are_dropped(tmp_path: Path) -> None:
    """The real Hal sheet carries a dark line along its bottom edge."""
    pixels = _misaligned_sheet()
    pixels[7] = [(30, 30, 30, 255)] * 40  # full-width artifact line
    (tmp_path / "yakou").mkdir()
    _write_png(tmp_path / "yakou" / "yakou_idle.png", pixels)
    art = SceneArt.load(tmp_path, cache_dir=None)
    frames = art.poses[("yakou", "idle")]
    # The canvas stops at the figures; the line row is not part of any frame.
    assert {(frame.width, frame.height) for frame in frames} == {(5, 4)}
    assert all(
        px[0] != 30 for frame in frames for row in frame.rows for px in row if px[3] >= 128
    )


def test_shrunk_keeps_authentic_pixels_instead_of_blending() -> None:
    """A pure shrink: every output pixel is a real pixel of the source."""
    ink, paper = (0, 0, 0, 255), (255, 255, 255, 255)
    half = Sprite(8, 2, tuple((ink,) * 4 + (paper,) * 4 for _ in range(2)))
    assert half.shrunk(2, 1).rows == ((ink, paper),)
    blended = half.resized(2, 1).rows[0]
    assert blended[0] == ink and blended[1] == paper  # box agrees on halves...
    mixed = Sprite(2, 1, ((ink, paper),)).resized(1, 1).rows[0][0]
    shrunk = Sprite(2, 1, ((ink, paper),)).shrunk(1, 1).rows[0][0]
    assert mixed[:3] not in (ink[:3], paper[:3]), "box-averaging invents greys"
    assert shrunk in (ink, paper), "shrinking never invents a colour"


def test_yakou_keeps_his_canonical_handedness(tmp_path: Path) -> None:
    """Players are mirrored to face left; the referee renders as authored."""
    dark, grey = (10, 10, 10, 255), (120, 120, 120, 255)
    asymmetric = [[dark, grey], [dark, grey]]  # dark column on the left
    (tmp_path / "yakou").mkdir()
    (tmp_path / "baku").mkdir()
    _write_png(tmp_path / "yakou" / "yakou_standing.png", asymmetric)
    _write_png(tmp_path / "baku" / "baku_dropping.png", asymmetric)
    art = SceneArt.load(tmp_path, cache_dir=None)
    yakou = art.poses[("yakou", "standing")][0]
    baku = art.poses[("baku", "dropping")][0]
    assert yakou.rows[0][0] == dark, "Yakou must not be mirrored"
    assert baku.rows[0][0] == grey, "players must still be mirrored"


def test_idle_frames_share_one_canvas_after_preparation(tmp_path: Path) -> None:
    """Per-frame trimming let an animated figure change shape between frames."""
    white, ink = (255, 255, 255, 255), (0, 0, 0, 255)
    # A four-frame sheet, two pixels per frame, with ink in different corners of
    # different frames. Trimmed individually the frames would have four
    # different shapes; on a shared canvas they must all come out identical.
    pixels = [[white] * 8 for _ in range(4)]
    pixels[0][0] = ink  # frame 0: top-left
    pixels[3][7] = ink  # frame 3: bottom-right
    (tmp_path / "yakou").mkdir()
    _write_png(tmp_path / "yakou" / "yakou_idle.png", pixels)
    art = SceneArt.load(tmp_path, cache_dir=None)
    frames = art.poses[("yakou", "idle")]
    assert len(frames) == IDLE_FRAMES
    assert {(frame.width, frame.height) for frame in frames} == {(2, 4)}


def test_figures_share_a_common_floor_line() -> None:
    pale = _tall_block((220, 220, 220, 255))
    seated = _sprite_block(pale, "x", 12, 16, pose="seated", colour=False)
    standing = _sprite_block(pale, "x", 12, 16, pose="dropping", colour=False)
    # Bottom-anchored, so both figures reach the last row of the block.
    assert seated[-1].strip() and standing[-1].strip()


def test_dropper_and_checker_use_their_action_poses() -> None:
    dropping, seated, idle = _block((0, 0, 0, 255)), _block((59, 59, 59, 255)), _block((255, 255, 255, 255))
    art = SceneArt(
        {
            ("hal", "dropping"): (dropping,),
            ("hal", "seated"): (seated,),
            ("baku", "idle"): (idle,),
        }
    )
    # Hal drops in half 1, so the dropping pose is what the scene must pick.
    assert art.for_action("Hal", "dropping") is dropping
    assert art.for_action("Hal", "seated") is seated
    # Baku has no dropping fixture here, so the idle sheet is the fallback.
    assert art.for_action("Baku", "dropping") is idle


def test_scene_block_width_is_stable_without_art() -> None:
    lines = render_frame(_game(), art=SceneArt(), colour=False)
    scene = [line for line in lines if "[Hal]" in line or "[Baku]" in line]
    assert scene
    assert all(len(line) == Layout().width for line in scene)


# ── layout ────────────────────────────────────────────────────────────────


def test_layout_fills_the_terminal_it_is_given() -> None:
    layout = Layout.detect(columns=121, lines=45)
    assert layout.width == 120
    assert layout.scene_rows == 45 - Layout.CHROME_LINES


def test_layout_clamps_absurd_terminal_sizes() -> None:
    tiny = Layout.detect(columns=20, lines=4)
    assert tiny.width == 80 and tiny.scene_rows == 10
    # The row cap is where a sextant-rendered figure reaches the fixtures'
    # ~170px native height (56 * 3 = 168), so it is the fidelity ceiling too.
    huge = Layout.detect(columns=500, lines=500)
    assert huge.width == 240 and huge.scene_rows == 56


def test_figure_width_follows_its_own_shape() -> None:
    """Width is derived from aspect so figures are not padded out to fill slots."""
    tall = _tall_block((255, 255, 255, 255), width=8, height=32)
    wide = _tall_block((255, 255, 255, 255), width=32, height=32)
    # A cell is about twice as tall as it is wide, hence the factor of two.
    assert _figure_columns(tall, 20) == round(2 * 20 * 8 / 32)
    assert _figure_columns(wide, 20) > _figure_columns(tall, 20)


def test_oversized_figures_are_shed_to_fit_the_frame() -> None:
    """Very wide art must not burst the border on a narrow terminal."""
    wide = _tall_block((255, 255, 255, 255), width=200, height=20)
    art = SceneArt(
        {
            ("hal", "dropping"): (wide,),
            ("baku", "seated"): (wide,),
            ("yakou", "standing"): (wide,),
        }
    )
    layout = Layout(80, 12)
    lines = render_frame(_game(), art=art, layout=layout, colour=False)
    assert {len(line) for line in lines} == {layout.width}


# ── fixture preparation ───────────────────────────────────────────────────


def test_frames_splits_a_horizontal_sheet() -> None:
    sheet = Sprite(8, 2, tuple(tuple((x * 8, 0, 0, 255) for x in range(8)) for _ in range(2)))
    parts = sheet.frames(4)
    assert len(parts) == 4
    assert all(part.width == 2 and part.height == 2 for part in parts)
    assert parts[0].rows[0][0][0] == 0
    assert parts[3].rows[0][0][0] == 48


def test_frames_rejects_a_width_that_does_not_divide() -> None:
    with pytest.raises(SpriteError):
        Sprite(7, 1, ((((0, 0, 0, 255),) * 7),)).frames(4)


def test_mirroring_reverses_each_row_and_is_its_own_inverse() -> None:
    left, right = (0, 0, 0, 255), (255, 255, 255, 255)
    sprite = Sprite(2, 1, ((left, right),))
    assert sprite.mirrored().rows[0] == (right, left)
    assert sprite.mirrored().mirrored().rows == sprite.rows


def test_padding_restores_the_margin_that_trimming_removed() -> None:
    ink = Sprite(1, 1, (((0, 0, 0, 255),),))
    padded = ink.padded(2)
    assert (padded.width, padded.height) == (5, 5)
    assert padded.rows[0][0][3] == 0
    assert padded.rows[2][2][3] == 255


def test_rim_lights_dark_masses_but_leaves_pale_ones_alone() -> None:
    """Hal and Yakou are mostly near-black and would dissolve into the field."""
    clear = (0, 0, 0, 0)
    dark = Sprite(3, 3, ((clear, clear, clear), (clear, (10, 10, 10, 255), clear), (clear, clear, clear)))
    pale = Sprite(3, 3, ((clear, clear, clear), (clear, (240, 240, 240, 255), clear), (clear, clear, clear)))
    rim = (110, 110, 110)
    assert dark.rimmed(rim).rows[0][1][:3] == rim
    assert dark.rimmed(rim).rows[0][0][3] == 0, "corners are not 4-adjacent"
    assert pale.rimmed(rim).rows[0][1][3] == 0, "a pale figure needs no rim"


def test_rim_does_not_overwrite_the_figure() -> None:
    ink = Sprite(2, 1, (((0, 0, 0, 255), (0, 0, 0, 0)),))
    rimmed = ink.rimmed((110, 110, 110))
    assert rimmed.rows[0][0][:3] == (0, 0, 0)


def test_luma_orders_black_below_white() -> None:
    assert luma(0, 0, 0) < luma(128, 128, 128) < luma(255, 255, 255)


def test_keying_clears_the_border_paper_but_keeps_enclosed_white() -> None:
    """Baku's coat and the handkerchief are white and must survive keying."""
    white, ink = (255, 255, 255, 255), (0, 0, 0, 255)
    rows = (
        (white, white, white, white, white),
        (white, ink, ink, ink, white),
        (white, ink, white, ink, white),
        (white, ink, ink, ink, white),
        (white, white, white, white, white),
    )
    keyed = Sprite(5, 5, rows).keyed()
    assert keyed.rows[0][0][3] == 0, "border paper should be cleared"
    assert keyed.rows[2][2][3] == 255, "enclosed white should be preserved"
    assert keyed.rows[1][1][3] == 255, "ink should be preserved"




def test_render_cells_fills_transparency_with_paper_when_asked() -> None:
    clear = tuple(tuple((0, 0, 0, 0) for _ in range(4)) for _ in range(4))
    lines = render_cells(Sprite(4, 4, clear), 4, 2, colour=True, paper=(255, 255, 255))
    assert all("\x1b[48;2;255;255;255m" in line for line in lines)


def test_prepared_frames_round_trip_through_the_cache(tmp_path: Path) -> None:
    original = Sprite(
        2,
        2,
        (
            ((0, 0, 0, 255), (255, 255, 255, 255)),
            ((122, 122, 122, 255), (0, 0, 0, 0)),
        ),
    )
    path = tmp_path / "cached.png"
    write_png(original, path)
    restored = decode_png(path)
    assert restored.rows == original.rows


def test_decode_respects_max_edge(tmp_path: Path) -> None:
    pixels = [[(0, 0, 0, 255)] * 40 for _ in range(20)]
    path = _write_png(tmp_path / "big.png", pixels)
    assert (decode_png(path).width, decode_png(path).height) == (40, 20)
    small = decode_png(path, max_edge=10)
    assert (small.width, small.height) == (10, 5)

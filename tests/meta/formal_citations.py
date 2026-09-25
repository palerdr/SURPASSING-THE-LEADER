"""Resolve the file:line citations in the Lean proofs and lock the cited text.

The docstrings in ``src/formal/**/*.lean`` and the rows of
``src/formal/CLAIMS.md`` cite source lines as ``path:line`` or
``path:first-last`` tokens. One token can list several ranges, as in
``path:13-16,191-206``, with no space after a comma. This module resolves every
token and records the SHA-256 of each cited span in ``src/formal/citations.lock``.
An edit that moves or changes cited text then fails
``tests/meta/test_formal_citations.py``.

Resolution rules:

- A token that names a source file relative to the repository root resolves
  as written.
- A bare file name (no ``/``) resolves when one source file, and no other,
  has that name.
- Any other token is an error: write the repository-relative path.
- A token that a comma, whitespace, and a number follow is an error, because
  the ranges after the space would escape the lock.

The lock also records the anchor of each span: the fewest lines of context on
each side that make the span's text unique in its file, and the SHA-256 of that
wider range. The checker follows cited text to a new range only when the
anchor occurs exactly once in the file. When the anchor is missing or occurs
more than once, the checker reports the span as changed, because it cannot
tell which copy of the text the proof means.

Source files are the files that git tracks, plus untracked files that git does
not ignore, outside the generated trees in ``IGNORED_PARTS``.

Commands, from the repository root::

    uv run python tests/meta/formal_citations.py
    uv run python tests/meta/formal_citations.py --refresh
    uv run python tests/meta/formal_citations.py --accept src/stl/solver/leap_build.py

The first command checks. ``--refresh`` follows cited text that moved: it
rewrites the ranges in the Lean text and in the lock, and it drops lock entries
that no Lean file cites. ``--accept PATH`` locks the current text of every span
that PATH cites or that sits in PATH. PATH can be a citing file, a cited file,
or a directory. Review the proofs against the new text before you run
``--accept``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FORMAL = "src/formal"
CLAIMS = f"{FORMAL}/CLAIMS.md"
LOCK = f"{FORMAL}/citations.lock"
COMMAND = "uv run python tests/meta/formal_citations.py"

# Generated trees. A file under one of these directory names is never a
# citation target, so vendored Mathlib under src/formal/build, the Lake cache in
# src/formal/.lake, and bundles such as src/browser/build do not make a bare
# file name ambiguous.
IGNORED_PARTS = frozenset(
    {
        ".git",
        ".lake",
        ".venv",
        ".tools",
        ".pytest_cache",
        ".ruff_cache",
        "graphify-out",
        "outputs",
        "checkpoints",
        "artifacts",
        "target",
        "build",
        "_build",
        "_opam",
        "__pycache__",
        "node_modules",
        "dist",
    }
)

TOKEN = re.compile(
    r"(?<![\w./-])(?P<path>\w[\w./-]*\.[A-Za-z0-9]+)"
    r":(?P<ranges>\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)",
    re.ASCII,
)
# TOKEN reads a range list only when no space follows a comma. This pattern,
# matched where a token ends, finds a list that continues after a space.
LOOSE_LIST = re.compile(r",\s+\d")
# The most context lines on each side that an anchor can take.
ANCHOR_LINES = 3
LOCK_HEADER = (
    "# Written by tests/meta/formal_citations.py. Do not edit by hand.\n"
    "# citing file\ttoken\tresolved path\tlines\tsha256 of the cited lines"
    "\tanchor context lines\tsha256 of the anchor lines\n"
)


@dataclass(frozen=True)
class Span:
    """One cited line range, as the citing file writes it."""

    citing: str
    line: int
    token: str
    first: int
    last: int

    @property
    def lines(self) -> str:
        return _range_text(self.first, self.last)

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (self.citing, self.token, self.first, self.last)

    @property
    def where(self) -> str:
        return f"{self.citing}:{self.line}: `{self.token}:{self.lines}`"


@dataclass(frozen=True)
class Locked:
    citing: str
    token: str
    resolved: str
    first: int
    last: int
    digest: str
    # (context lines on each side, digest of the wider range), or None when no
    # context up to ANCHOR_LINES makes the text unique in its file.
    anchor: tuple[int, str] | None

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (self.citing, self.token, self.first, self.last)


@dataclass(frozen=True)
class Problem:
    kind: str
    message: str


@dataclass
class Report:
    spans: list[Span] = field(default_factory=list)
    problems: list[Problem] = field(default_factory=list)
    # Spans that resolve and fit their file, with the text they cite today.
    current: dict[tuple[str, str, int, int], Locked] = field(default_factory=dict)
    # Locked spans whose text now sits at another range or in another file.
    moves: dict[tuple[str, str, int, int], Locked] = field(default_factory=dict)


def _range_text(first: int, last: int) -> str:
    return str(first) if first == last else f"{first}-{last}"


def _ignored(relative: str) -> bool:
    return not IGNORED_PARTS.isdisjoint(relative.split("/"))


def walk_source_files(root: Path) -> frozenset[str]:
    """List the files under ``root`` outside the generated trees."""

    found = set()
    for directory, names, files in os.walk(root):
        names[:] = [name for name in names if name not in IGNORED_PARTS]
        base = Path(directory).relative_to(root)
        for name in files:
            found.add((base / name).as_posix())
    return frozenset(found)


def source_files(root: Path) -> frozenset[str]:
    """List the files git tracks or would track, outside the generated trees.

    Outside a git checkout, fall back to a walk of the working tree.
    """

    try:
        listing = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return walk_source_files(root)
    names = (entry.decode("utf-8") for entry in listing.split(b"\0") if entry)
    return frozenset(name for name in names if not _ignored(name) and (root / name).is_file())


def citing_files(root: Path) -> list[str]:
    """List the Lean sources of ``src/formal`` and its claims ledger."""

    base = root / FORMAL
    found = []
    for directory, names, files in os.walk(base):
        names[:] = sorted(name for name in names if name not in IGNORED_PARTS)
        for name in sorted(files):
            if name.endswith(".lean"):
                found.append((Path(directory) / name).relative_to(root).as_posix())
    if (root / CLAIMS).is_file():
        found.append(CLAIMS)
    return found


def collect_spans(root: Path, citing: str) -> tuple[list[Span], list[Problem]]:
    """Read the cited spans of one file, and flag range lists with spaces."""

    spans, problems = [], []
    lines = (root / citing).read_text(encoding="utf-8").split("\n")
    for number, line in enumerate(lines, start=1):
        following = lines[number] if number < len(lines) else ""
        for match in TOKEN.finditer(line):
            for part in match.group("ranges").split(","):
                first, _, last = part.partition("-")
                spans.append(Span(citing, number, match.group("path"), int(first), int(last or first)))
            if LOOSE_LIST.match(line[match.end() :] + "\n" + following):
                problems.append(
                    Problem(
                        "malformed",
                        f"{citing}:{number}: `{match.group(0)}` continues its range list after a comma "
                        f"and a space, so the lock misses the later ranges. Write the list with no "
                        f"spaces, as in `{match.group('path')}:1-2,5-6`.",
                    )
                )
    return spans, problems


class Resolver:
    def __init__(self, sources: frozenset[str]) -> None:
        self.sources = sources
        self.by_name: dict[str, list[str]] = defaultdict(list)
        for path in sorted(sources):
            self.by_name[path.rsplit("/", 1)[-1]].append(path)

    def resolve(self, token: str) -> tuple[str | None, Problem | None]:
        if token in self.sources:
            return token, None
        name = token.rsplit("/", 1)[-1]
        if "/" not in token:
            matches = self.by_name.get(name, [])
            if len(matches) == 1:
                return matches[0], None
        else:
            matches = [path for path in self.by_name.get(name, []) if path.endswith("/" + token)]
        if not matches:
            return None, Problem("missing", "names no source file")
        choices = ", ".join(matches)
        return None, Problem("ambiguous", f"is ambiguous; write the repository-relative path ({choices})")


def read_lines(root: Path, path: str) -> list[str]:
    text = (root / path).read_bytes().decode("utf-8", errors="replace").replace("\r\n", "\n")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def span_digest(lines: list[str], first: int, last: int) -> str:
    return hashlib.sha256("\n".join(lines[first - 1 : last]).encode("utf-8")).hexdigest()


def occurrences(lines: list[str], digest: str, length: int) -> list[tuple[int, int]]:
    """List every range of ``length`` lines whose text hashes to ``digest``."""

    return [
        (first, first + length - 1)
        for first in range(1, len(lines) - length + 2)
        if span_digest(lines, first, first + length - 1) == digest
    ]


def anchor(lines: list[str], first: int, last: int) -> tuple[int, str] | None:
    """Widen a span by the fewest context lines that make its text unique.

    Return the count of context lines on each side and the digest of the wider
    range. Return None when no count up to ANCHOR_LINES works inside the file.
    """

    for context in range(ANCHOR_LINES + 1):
        start, end = first - context, last + context
        if start < 1 or end > len(lines):
            return None
        window = lines[start - 1 : end]
        copies = sum(
            1
            for index in range(len(lines) - len(window) + 1)
            if lines[index] == window[0] and lines[index : index + len(window)] == window
        )
        if copies == 1:
            return context, span_digest(lines, start, end)
    return None


def locate(lines: list[str], locked: Locked) -> tuple[int, int] | None:
    """Find the range that holds the locked text, when its anchor is unique.

    Return None when the lock has no anchor or the anchor text occurs zero
    times or more than once. Two copies of the cited text can then remain, and
    one of them can be a line that someone edited.
    """

    if locked.anchor is None:
        return None
    context, digest = locked.anchor
    found = occurrences(lines, digest, locked.last - locked.first + 1 + 2 * context)
    if len(found) != 1:
        return None
    first = found[0][0] + context
    return first, first + locked.last - locked.first


def read_lock(root: Path) -> dict[tuple[str, str, int, int], Locked]:
    entries = {}
    path = root / LOCK
    if not path.is_file():
        return entries
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith("#"):
            continue
        citing, token, resolved, lines, digest, context, anchor_digest = raw.split("\t")
        first, _, last = lines.partition("-")
        found = None if context == "-" else (int(context), anchor_digest)
        entry = Locked(citing, token, resolved, int(first), int(last or first), digest, found)
        entries[entry.key] = entry
    return entries


def write_lock(root: Path, entries: dict[tuple[str, str, int, int], Locked]) -> None:
    rows = [
        "\t".join(
            (
                e.citing,
                e.token,
                e.resolved,
                _range_text(e.first, e.last),
                e.digest,
                *(("-", "-") if e.anchor is None else (str(e.anchor[0]), e.anchor[1])),
            )
        )
        for e in sorted(entries.values(), key=lambda e: e.key)
    ]
    (root / LOCK).write_text(LOCK_HEADER + "".join(row + "\n" for row in rows), encoding="utf-8", newline="\n")


def audit(root: Path = ROOT, sources: frozenset[str] | None = None) -> Report:
    """Check every citation against the source tree and the lock."""

    resolver = Resolver(source_files(root) if sources is None else sources)
    lock = read_lock(root)
    report = Report()
    cache: dict[str, list[str]] = {}
    for citing in citing_files(root):
        spans, problems = collect_spans(root, citing)
        report.spans.extend(spans)
        report.problems.extend(problems)

    for span in report.spans:
        resolved, problem = resolver.resolve(span.token)
        if problem is not None:
            report.problems.append(Problem(problem.kind, f"{span.where} {problem.message}."))
            continue
        if resolved not in cache:
            cache[resolved] = read_lines(root, resolved)
        lines = cache[resolved]
        locked = lock.get(span.key)
        in_range = 1 <= span.first <= span.last <= len(lines)
        digest = span_digest(lines, span.first, span.last) if in_range else None
        if in_range:
            report.current[span.key] = Locked(
                *span.key[:2], resolved, span.first, span.last, digest, anchor(lines, span.first, span.last)
            )
        if locked is not None and locked.resolved == resolved and locked.digest == digest:
            continue
        if locked is not None:
            found = locate(lines, locked)
            if found is not None:
                moved = Locked(span.citing, span.token, resolved, *found, locked.digest, anchor(lines, *found))
                report.moves[span.key] = moved
                target = f"{resolved}:{_range_text(*found)}"
                report.problems.append(
                    Problem(
                        "moved",
                        f"{span.where} moved: the locked text now sits at {target}. "
                        f"Run `{COMMAND} --refresh`.",
                    )
                )
                continue
        if not in_range:
            report.problems.append(
                Problem(
                    "out-of-range",
                    f"{span.where} is out of range: {resolved} has {len(lines)} lines.",
                )
            )
        elif locked is None:
            report.problems.append(
                Problem(
                    "unlocked",
                    f"{span.where} has no lock entry. Check the proof against "
                    f"{resolved}:{span.lines}, then run `{COMMAND} --accept {span.citing}`.",
                )
            )
        else:
            copies = occurrences(lines, locked.digest, locked.last - locked.first + 1)
            elsewhere = ", ".join(f"{resolved}:{_range_text(*copy)}" for copy in copies)
            note = (
                f"The locked text also sits at {elsewhere}, and the checker cannot tell "
                f"whether the proof means that copy. "
                if copies
                else ""
            )
            report.problems.append(
                Problem(
                    "changed",
                    f"{span.where} cites text that changed in {resolved}:{span.lines}. {note}"
                    f"Review the proof and point the citation at the text it mirrors, then run "
                    f"`{COMMAND} --accept {resolved}`.",
                )
            )

    cited = {span.key for span in report.spans}
    for key, entry in sorted(lock.items()):
        if key not in cited:
            report.problems.append(
                Problem(
                    "stale",
                    f"{LOCK} locks `{entry.token}:{_range_text(entry.first, entry.last)}` for "
                    f"{entry.citing}, which no longer cites it. Run `{COMMAND} --refresh`.",
                )
            )
    return report


def _rewrite(root: Path, citing: str, moves: dict[tuple[str, str, int, int], Locked]) -> None:
    path = root / citing
    text = path.read_bytes().decode("utf-8")

    def replace(match: re.Match[str]) -> str:
        token = match.group("path")
        parts = []
        for part in match.group("ranges").split(","):
            first, _, last = part.partition("-")
            moved = moves.get((citing, token, int(first), int(last or first)))
            if moved is None:
                parts.append(part)
            elif last:
                parts.append(f"{moved.first}-{moved.last}")
            else:
                parts.append(str(moved.first))
        return f"{token}:{','.join(parts)}"

    updated = TOKEN.sub(replace, text)
    if updated != text:
        path.write_bytes(updated.encode("utf-8"))


def refresh(root: Path = ROOT, sources: frozenset[str] | None = None) -> Report:
    """Follow moved text into the Lean ranges and the lock, then re-check."""

    report = audit(root, sources)
    for citing in sorted({key[0] for key in report.moves}):
        _rewrite(root, citing, report.moves)
    cited = {span.key for span in report.spans}
    entries = {}
    for key, entry in read_lock(root).items():
        if key in report.moves:
            moved = report.moves[key]
            entries[moved.key] = moved
        elif key in cited:
            entries[key] = entry
    write_lock(root, entries)
    return audit(root, sources)


def _under(path: str, targets: list[str]) -> bool:
    return any(path == target or path.startswith(target + "/") for target in targets)


def accept(root: Path, targets: list[str], sources: frozenset[str] | None = None) -> Report:
    """Lock the current text of the spans that ``targets`` cite or contain."""

    normalized = []
    for target in targets:
        candidate = Path(target)
        if candidate.is_absolute():
            candidate = candidate.resolve().relative_to(root.resolve())
        normalized.append(candidate.as_posix().removeprefix("./").rstrip("/"))
    report = audit(root, sources)
    cited = {span.key for span in report.spans}
    entries = {}
    for key, entry in read_lock(root).items():
        if key in cited or not _under(entry.citing, normalized):
            entries[key] = entry
    for key, entry in report.current.items():
        if _under(entry.citing, normalized) or _under(entry.resolved, normalized):
            entries[key] = entry
    write_lock(root, entries)
    return audit(root, sources)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--refresh", action="store_true", help="follow moved text into the Lean ranges and the lock")
    action.add_argument(
        "--accept",
        action="append",
        metavar="PATH",
        help="lock the current text of the spans that PATH cites or contains (repeatable)",
    )
    args = parser.parse_args(argv)
    if args.refresh:
        report = refresh(ROOT)
    elif args.accept:
        report = accept(ROOT, args.accept)
    else:
        report = audit(ROOT)
    for problem in report.problems:
        print(problem.message)
    print(f"{len(report.spans)} cited spans, {len(report.problems)} problems")
    return 1 if report.problems else 0


if __name__ == "__main__":
    sys.exit(main())

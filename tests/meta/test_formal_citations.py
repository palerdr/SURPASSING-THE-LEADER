"""Every file:line citation in src/formal points at the text its proof mirrors.

The resolver and the lock live in ``formal_citations.py`` beside this file. The
first test checks the real tree against ``src/formal/citations.lock``. The
others build a small repository in a temporary directory and show that the
checker catches each kind of drift.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location("formal_citations", Path(__file__).with_name("formal_citations.py"))
citations = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = citations
_SPEC.loader.exec_module(citations)


def test_every_lean_citation_resolves_to_its_locked_text():
    report = citations.audit(citations.ROOT)
    assert report.spans, "src/formal cites no source lines"
    assert not report.problems, "\n".join(problem.message for problem in report.problems)


CODE = "def first():\n    return 1\n\n\ndef second():\n    return 2\n"
PROOF = "/-- Mirrors `src/pkg/code.py:5-6` and `other.py:1`. -/\ntheorem t : True := trivial\n"


def _repository(root: Path) -> frozenset[str]:
    files = {
        "src/pkg/code.py": CODE,
        "src/pkg/other.py": "VALUE = 1\n",
        "src/formal/Formal/T.lean": PROOF,
        "src/formal/CLAIMS.md": "# Claims\n",
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    sources = frozenset(files)
    report = citations.accept(root, ["src/formal"], sources)
    assert [span.lines for span in report.spans] == ["5-6", "1"]
    assert not report.problems
    return sources


def _kinds(root: Path, sources: frozenset[str]) -> list[str]:
    return [problem.kind for problem in citations.audit(root, sources).problems]


def test_a_missing_file_fails(tmp_path):
    sources = _repository(tmp_path)
    (tmp_path / "src/pkg/other.py").unlink()
    assert _kinds(tmp_path, sources - {"src/pkg/other.py"}) == ["missing"]


def test_a_line_past_the_end_fails(tmp_path):
    sources = _repository(tmp_path)
    (tmp_path / "src/pkg/code.py").write_text("def first():\n    return 1\n", encoding="utf-8")
    assert _kinds(tmp_path, sources) == ["out-of-range"]


def test_changed_text_fails_until_accepted(tmp_path):
    sources = _repository(tmp_path)
    (tmp_path / "src/pkg/code.py").write_text(CODE.replace("return 2", "return 3"), encoding="utf-8")
    report = citations.audit(tmp_path, sources)
    assert [problem.kind for problem in report.problems] == ["changed"]
    assert "--accept src/pkg/code.py" in report.problems[0].message
    assert not citations.accept(tmp_path, ["src/pkg/code.py"], sources).problems


def test_moved_text_names_its_new_range_and_refresh_follows_it(tmp_path):
    sources = _repository(tmp_path)
    (tmp_path / "src/pkg/code.py").write_text("\n" + CODE, encoding="utf-8")
    report = citations.audit(tmp_path, sources)
    assert [problem.kind for problem in report.problems] == ["moved"]
    assert "src/pkg/code.py:6-7" in report.problems[0].message
    assert "--refresh" in report.problems[0].message

    assert not citations.refresh(tmp_path, sources).problems
    proof = (tmp_path / "src/formal/Formal/T.lean").read_text(encoding="utf-8")
    assert "`src/pkg/code.py:6-7`" in proof
    assert "`other.py:1`" in proof


def test_a_bare_name_with_two_owners_is_ambiguous(tmp_path):
    sources = _repository(tmp_path)
    twin = tmp_path / "src/lib/other.py"
    twin.parent.mkdir(parents=True)
    twin.write_text("VALUE = 1\n", encoding="utf-8")
    assert _kinds(tmp_path, sources | {"src/lib/other.py"}) == ["ambiguous"]


def test_a_partial_path_is_ambiguous(tmp_path):
    sources = _repository(tmp_path)
    lean = tmp_path / "src/formal/Formal/T.lean"
    lean.write_text(PROOF.replace("src/pkg/code.py", "pkg/code.py"), encoding="utf-8")
    assert _kinds(tmp_path, sources) == ["ambiguous", "stale"]


@pytest.mark.parametrize("separator", [", ", ",\n"])
def test_a_range_list_with_a_space_is_malformed(tmp_path, separator):
    sources = _repository(tmp_path)
    lean = tmp_path / "src/formal/Formal/T.lean"
    lean.write_text(PROOF.replace("src/pkg/code.py:5-6", f"src/pkg/code.py:1-2{separator}5-6"), encoding="utf-8")
    report = citations.accept(tmp_path, ["src/formal"], sources)
    assert [problem.kind for problem in report.problems] == ["malformed"]
    assert report.problems[0].message.startswith("src/formal/Formal/T.lean:1: `src/pkg/code.py:1-2`")


# The cited line 2 has a twin at line 7. Line 1 before it makes it unique.
TWIN_CODE = (
    "def first(key):\n    step = child(key)\n    return step\n\n\n"
    "def second(key):\n    step = child(key)\n    return -step\n"
)
TWIN_PROOF = "/-- Mirrors `src/pkg/twin.py:2`. -/\ntheorem t : True := trivial\n"


def _twin_repository(root: Path) -> frozenset[str]:
    files = {
        "src/pkg/twin.py": TWIN_CODE,
        "src/formal/Formal/T.lean": TWIN_PROOF,
        "src/formal/CLAIMS.md": "# Claims\n",
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    sources = frozenset(files)
    assert not citations.accept(root, ["src/formal"], sources).problems
    return sources


def test_an_edited_line_with_a_twin_is_changed_and_not_moved(tmp_path):
    sources = _twin_repository(tmp_path)
    edited = TWIN_CODE.replace("step = child(key)\n    return step", "step = child(key) + 1\n    return step")
    (tmp_path / "src/pkg/twin.py").write_text(edited, encoding="utf-8")
    report = citations.audit(tmp_path, sources)
    assert [problem.kind for problem in report.problems] == ["changed"]
    assert "src/pkg/twin.py:7" in report.problems[0].message

    assert [problem.kind for problem in citations.refresh(tmp_path, sources).problems] == ["changed"]
    assert (tmp_path / "src/formal/Formal/T.lean").read_text(encoding="utf-8") == TWIN_PROOF


def test_a_shifted_line_with_a_twin_moves_by_its_anchor(tmp_path):
    sources = _twin_repository(tmp_path)
    (tmp_path / "src/pkg/twin.py").write_text("\n" + TWIN_CODE, encoding="utf-8")
    report = citations.audit(tmp_path, sources)
    assert [problem.kind for problem in report.problems] == ["moved"]
    assert "src/pkg/twin.py:3" in report.problems[0].message

    assert not citations.refresh(tmp_path, sources).problems
    assert "`src/pkg/twin.py:3`" in (tmp_path / "src/formal/Formal/T.lean").read_text(encoding="utf-8")


def test_generated_trees_never_make_a_name_ambiguous(tmp_path):
    _repository(tmp_path)
    for generated in ("src/formal/.lake/other.py", "src/formal/build/packages/other.py", "src/app/web/build/other.py"):
        path = tmp_path / generated
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("VALUE = 2\n", encoding="utf-8")
    sources = citations.walk_source_files(tmp_path)
    assert "src/pkg/other.py" in sources
    assert not citations.audit(tmp_path, sources).problems

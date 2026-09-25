"""hal_lab.provenance reproduces the recorded verification of the frozen evidence.

The verifier reads bound source files from the pre-restructure tag. A shallow
checkout, such as the CI clone, holds no tags, so the checks that read the tag
skip there. The checks of the parser and the path map need no tag.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from hal_lab import provenance


def _tag() -> provenance.TagTree:
    try:
        return provenance.TagTree()
    except provenance.TagUnavailable as error:
        pytest.skip(str(error))


def test_report_equals_the_recorded_baseline() -> None:
    tag = _tag()
    baseline = json.loads(provenance.BASELINE.read_text(encoding="utf-8"))
    differences = provenance.compare(baseline, provenance.report(tag=tag))
    # A checkout without the ignored outputs/ store cannot read the bindings
    # that name files on disk. The git-history bindings and the record bytes
    # still have to match.
    if not (provenance.ROOT / "outputs").is_dir():
        differences = [difference for difference in differences if difference.source != "disk"]
    assert not differences, "\n".join(str(difference) for difference in differences)


def test_check_fails_on_a_changed_baseline(tmp_path, capsys) -> None:
    _tag()
    baseline = json.loads(provenance.BASELINE.read_text(encoding="utf-8"))
    record = baseline["records"]["src/arena/config/translated_hal_v1_selection.json"]
    binding = next(entry for entry in record["bindings"] if entry["source"] == "git")
    binding["status"] = "mismatch"
    changed = tmp_path / "baseline.json"
    changed.write_text(json.dumps(baseline), encoding="utf-8")

    assert provenance.main(["--check", "--baseline", str(changed)]) == 1
    assert "status: 'mismatch' -> 'verified'" in capsys.readouterr().out


def test_every_record_exists_at_its_mapped_path() -> None:
    path_map = provenance.load_path_map()
    for name in provenance.RECORDS:
        assert name in path_map.moves, f"PATH_MAP.toml has no [path] entry for {name}"
        assert (provenance.ROOT / path_map.current(name)).is_file(), name


def test_path_map_resolves_absolute_and_bare_keys() -> None:
    path_map = provenance.PathMap(
        absolute_roots=("/old/checkout/",),
        basenames={"perfect_hal.py": "src/arena/policies/perfect_hal.py"},
        moves={"src/arena/policies/perfect_hal.py": "src/hal_lab/perfect_hal.py"},
    )

    assert path_map.historical("/old/checkout/outputs/a.json") == ("outputs/a.json", None)
    assert path_map.historical("/elsewhere/outputs/a.json")[0] is None
    assert path_map.historical("perfect_hal.py") == ("src/arena/policies/perfect_hal.py", None)
    assert path_map.historical("value.npy")[0] is None
    assert path_map.historical("src/arena/match.py") == ("src/arena/match.py", None)
    assert path_map.current("src/arena/policies/perfect_hal.py") == "src/hal_lab/perfect_hal.py"
    assert path_map.current("outputs/a.json") == "outputs/a.json"


def test_bindings_of_reads_tables_pairs_and_path_free_hashes() -> None:
    sha = "a" * 64
    document = {
        "input_sha256": {"/root/src/x.py": sha, "src/y.py": sha, "z.py": sha},
        "report_path": "outputs/r.json",
        "report_sha256": sha,
        "bindings": {"config": {"path": "c.yaml", "sha256": sha, "canonical_yaml_sha256": sha}},
        "goal_manifest": "g.json",
        "goal_manifest_canonical_json_sha256": sha,
        "table_digest": sha,
        "source": {"url": "https://example.org/data.csv", "sha256": sha},
        "config_sha256": sha,
        "seeds": [sha],
        "note": "not a hash",
    }

    found = {binding.pointer: binding for binding in provenance.bindings_of(document)}

    assert [(binding.key, binding.method) for binding in found.values() if binding.key] == [
        ("/root/src/x.py", "bytes"),
        ("src/y.py", "bytes"),
        ("z.py", "bytes"),
        ("outputs/r.json", "bytes"),
        ("c.yaml", "bytes"),
        ("c.yaml", "canonical-json"),
        ("g.json", "canonical-json"),
    ]
    assert "/input_sha256/~1root~1src~1x.py" in found
    assert found["/table_digest"].reason == "artifact digest, not the hash of one file"
    assert found["/source/sha256"].reason == "the hash binds a download URL"
    assert found["/config_sha256"].reason == "no field beside the hash names its file"
    assert found["/seeds/0"].reason == "a list holds the hash without a file"


def test_canonical_json_hash_sorts_keys_and_drops_whitespace() -> None:
    expected = hashlib.sha256(b'{"a":[1,2],"b":"\\u00e9"}').hexdigest()
    assert provenance.canonical_json_sha256({"b": "é", "a": [1, 2]}) == expected
    assert provenance.digest(b"b: \xc3\xa9\na: [1, 2]\n", "canonical-json", "x.yaml") == expected


def test_compare_names_each_changed_field() -> None:
    entry = {"pointer": "/p", "path": "outputs/a", "source": "disk", "status": "verified"}
    baseline = {"tag_commit": "c", "records": {"r": {"sha256": "1", "bindings": [entry]}}}
    current = {"tag_commit": "c", "records": {"r": {"sha256": "2", "bindings": [dict(entry, status="missing")]}}}

    differences = provenance.compare(baseline, current)

    assert [str(difference) for difference in differences] == [
        "r: sha256: 1 -> 2",
        "r/p: status: 'verified' -> 'missing'",
    ]
    assert differences[1].source == "disk"

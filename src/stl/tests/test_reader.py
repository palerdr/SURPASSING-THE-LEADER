"""We test stl.reader, the verified read path into a completed leap artifact."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

ARTIFACT = Path(__file__).resolve().parents[1]/'outputs'/'leap-full-native'
DTH_PATH = Path(__file__).resolve().parents[2]/'dth_compact'/'artifacts'/'V.npy'
needs_artifact = pytest.mark.skipif(
    not (ARTIFACT/'manifest.json').is_file() or not DTH_PATH.is_file(),
    reason='src/stl/outputs/leap-full-native or the DTH tail is unavailable')


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def fake_artifact(tmp_path):
    """A manifest, one listed file, and a DTH tail; the tables stay unread."""
    dth = tmp_path/'V.npy'
    np.save(dth, np.arange(6, dtype=np.float64).reshape(2, 3))
    artifact = tmp_path/'leap'
    artifact.mkdir()
    (artifact/'audit.json').write_text('{}\n')
    manifest = {'schema': 'stl-leap-values-v1', 'complete': True, 'builder_sha256': 'b'*64,
                'dth_sha256': _sha256(dth), 'files': {'audit.json': _sha256(artifact/'audit.json')}}
    (artifact/'manifest.json').write_text(json.dumps(manifest))
    return artifact, dth, manifest


def _rewrite(artifact, manifest, **changes):
    (artifact/'manifest.json').write_text(json.dumps({**manifest, **changes}))


def test_open_leap_checks_schema_completion_and_dth_hash(fake_artifact, tmp_path):
    from stl.reader import open_leap
    artifact, dth, manifest = fake_artifact
    identity = open_leap(artifact, dth_values=dth).identity
    assert identity.artifact == artifact.resolve() and identity.dth_values == dth.resolve()
    assert identity.builder_sha256 == 'b'*64 and identity.dth_sha256 == manifest['dth_sha256']
    assert identity.schema == 'stl-leap-values-v1' and identity.verified == 'manifest'
    with pytest.raises(ValueError, match='verify'):
        open_leap(artifact, dth_values=dth, verify='none')
    with pytest.raises(FileNotFoundError, match='manifest'):
        open_leap(tmp_path/'missing', dth_values=dth)
    _rewrite(artifact, manifest, schema='stl-leap-values-v0')
    with pytest.raises(ValueError, match='schema'):
        open_leap(artifact, dth_values=dth)
    _rewrite(artifact, manifest, complete=False)
    with pytest.raises(ValueError, match='complete'):
        open_leap(artifact, dth_values=dth)
    _rewrite(artifact, manifest, dth_sha256='0'*64)
    with pytest.raises(ValueError, match='DTH tail'):
        open_leap(artifact, dth_values=dth)


def test_files_mode_rehashes_every_listed_file(fake_artifact):
    from stl.reader import open_leap
    artifact, dth, _ = fake_artifact
    assert open_leap(artifact, dth_values=dth, verify='files').identity.verified == 'files'
    (artifact/'audit.json').write_text('{"changed": true}\n')
    assert open_leap(artifact, dth_values=dth).identity.verified == 'manifest'
    with pytest.raises(ValueError, match='audit.json'):
        open_leap(artifact, dth_values=dth, verify='files')


def test_default_paths_come_from_the_package_or_absolute_environment(fake_artifact, tmp_path, monkeypatch):
    from stl import reader
    artifact, dth, _ = fake_artifact
    package = Path(reader.__file__).resolve().parent
    assert reader.DEFAULT_ARTIFACT == package/'outputs'/'leap-full-native'
    assert reader.DEFAULT_DTH == package.parent/'dth_compact'/'artifacts'/'V.npy'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('STL_LEAP_ARTIFACT', str(artifact))
    monkeypatch.setenv('STL_LEAP_DTH', str(dth))
    assert reader.open_leap().identity.artifact == artifact.resolve()
    monkeypatch.setenv('STL_LEAP_ARTIFACT', 'leap')
    with pytest.raises(ValueError, match='absolute'):
        reader.open_leap()


def test_profile_and_column_lookups_reject_missing_profiles(fake_artifact):
    from stl.reader import open_leap
    from stl.solver.leap_profiles import WIN, profiles
    artifact, dth, _ = fake_artifact
    table = open_leap(artifact, dth_values=dth)
    p = profiles()
    index = table.profile_index(st=60, ttd=120)
    assert (p.st[index], p.ttd[index]) == (60, 120)
    with pytest.raises(ValueError, match='no profile'):
        table.profile_index(st=299, ttd=0)
    with pytest.raises(ValueError, match='no column'):
        table.value(('REV', 3600), 0, index)
    with pytest.raises(ValueError, match='out of range'):
        table.value(('H1', 44), 0, WIN+1)


def test_stage_enforces_the_1e6_residual_and_gap_gates(fake_artifact, monkeypatch):
    """Synthetic certificates drive each gate; a stored 0.0 makes the 1e-6 residual exact."""
    from stl.reader import open_leap
    artifact, dth, _ = fake_artifact
    table = open_leap(artifact, dth_values=dth)
    monkeypatch.setattr(table, 'value', lambda key, checker, dropper: 0.0)
    monkeypatch.setattr('stl.reader.stage_inputs', lambda *args: (None, None))
    monkeypatch.setattr('stl.reader.explicit_matrix', lambda *args: None)

    def certify(certificate):
        monkeypatch.setattr('stl.reader.certify_matrix', lambda matrix: dict(certificate))

    certify({'value': 2e-6, 'gap': 0.0})
    with pytest.raises(ValueError, match='Bellman residual'):
        table.stage(('H1', 44), 0, 0)
    certify({'value': 0.0, 'gap': 2e-6})
    with pytest.raises(ValueError, match='saddle gap'):
        table.stage(('H1', 44), 0, 0)
    certify({'value': 1e-6, 'gap': 1e-6})
    assert table.stage(('H1', 44), 0, 0) == {'stored_value': 0.0, 'bellman_residual': 1e-6,
                                             'value': 1e-6, 'gap': 1e-6}

    def lp_failure(matrix):
        raise RuntimeError('opening/audit LP failed the 1e-6 gate')

    monkeypatch.setattr('stl.reader.certify_matrix', lp_failure)
    with pytest.raises(ValueError, match='H1_44 checker 0 dropper 0: opening/audit LP failed the 1e-6 gate'):
        table.stage(('H1', 44), 0, 0)


@needs_artifact
def test_reader_reproduces_the_h1_strategy_records(monkeypatch):
    """The 16 records of paper/make_stl_figures.py, minutes 44-59, through the reader."""
    from stl.reader import DEFAULT_ARTIFACT, open_leap
    from stl.solver.leap_audit import PackedReader, certify_matrix, explicit_matrix, stage_inputs
    monkeypatch.delenv('STL_LEAP_ARTIFACT', raising=False)
    monkeypatch.delenv('STL_LEAP_DTH', raising=False)
    table = open_leap()
    manifest = json.loads((ARTIFACT/'manifest.json').read_text())
    assert table.identity.artifact == DEFAULT_ARTIFACT == ARTIFACT
    assert table.identity.builder_sha256 == manifest['builder_sha256']
    assert table.identity.dth_sha256 == manifest['dth_sha256']
    checker = table.profile_index(st=60, ttd=120)
    dropper = table.profile_index(st=0, ttd=180)
    # We rebuild each record on the path the figure script used before stl.reader.
    packed = PackedReader(ARTIFACT/'tables')
    dth = np.load(DTH_PATH, mmap_mode='r')
    for minute in range(44, 60):
        key = ('H1', minute)
        record = table.stage(key, checker, dropper)
        stored = float(packed.get(key, checker, dropper))
        certificate = certify_matrix(explicit_matrix(*stage_inputs(packed, key, checker, dropper, dth), False))
        assert record == {'stored_value': stored, 'bellman_residual': abs(stored-certificate['value']),
                          **certificate}
        assert record['bellman_residual'] <= 1e-6 and record['gap'] <= 1e-6
        assert len(record['drop']) == len(record['check']) == 60
        assert abs(sum(record['drop'])-1) < 1e-9 and abs(sum(record['check'])-1) < 1e-9


@needs_artifact
def test_reader_maps_revival_columns_and_certifies_window_stages():
    from stl.reader import open_leap
    from stl.solver.leap_audit import PackedReader
    from stl.solver.leap_build import is_window
    from stl.solver.leap_profiles import profiles
    table = open_leap(ARTIFACT, dth_values=DTH_PATH)
    packed = PackedReader(ARTIFACT/'tables')
    rng = np.random.default_rng(93482)
    for key in (('REV', 3600), ('H2', 58), ('H1', 12)):
        (checker,), (column,) = packed.sample(key, 1, rng)
        dropper = int(profiles().s0[column]) if key[0] == 'REV' else int(column)
        assert table.value(key, int(checker), dropper) == float(packed.get(key, checker, column))
        record = table.stage(key, int(checker), dropper)
        assert len(record['drop']) == 60+int(is_window(key))
        assert record['bellman_residual'] <= 1e-6 and record['gap'] <= 1e-6

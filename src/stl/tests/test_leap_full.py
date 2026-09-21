"""We test the full sweep and packed-value audit before running the opening."""
from pathlib import Path
import json

import numpy as np
import pytest


def test_full_mode_crosses_calibration_floor(tmp_path, monkeypatch):
    from stl.solver import leap_build as b
    path = Path('src/dth_compact/artifacts/V.npy')
    if not path.exists():
        pytest.skip('DTH artifact unavailable')
    dth = np.load(path, mmap_mode='r')
    store = b.TableStore(tmp_path)
    output = np.lib.format.open_memmap(tmp_path/'one.npy', mode='w+', shape=(1, 1), dtype=float)
    monkeypatch.setattr(store, 'create', lambda key: output)
    monkeypatch.setattr(store, 'load', lambda key: dth)
    bits = np.zeros((b.N, (b.N+7)//8), np.uint8); bits[0, 0] = 1
    with pytest.raises(ValueError, match='3420'):
        b.sweep_key(('H1', 56), bits, store, dth, min_clock=720)
    result = b.sweep_key(('H1', 56), bits, store, dth, min_clock=720, full=True)
    assert result['states'] == 1 and result['failures'] == 0
    assert abs(output[0, 0]-dth[0, 0]) < 1e-9
    with pytest.raises(ValueError, match='720'):
        b.sweep_key(('H1', 11), bits, store, dth, min_clock=660, full=True)


@pytest.fixture
def small_packed(tmp_path, monkeypatch):
    from stl.solver import leap_build as b, leap_audit as a
    monkeypatch.setattr(b, 'N', 400); monkeypatch.setattr(a, 'N', 400)
    key = ('REV', 3600); store = b.TableStore(tmp_path/'tables')
    table = store.create(key)
    rng = np.random.default_rng(384)
    mask = rng.random((400, 183)) < .17
    data = rng.uniform(-1, 1, (400, 183))
    table[:, :183][mask] = data[mask]; table.flush(); del table
    store.pack(key)
    return a, b, store, key, mask, data


def test_packed_reader_rank_and_select(small_packed):
    a, b, store, key, mask, data = small_packed
    reader = a.PackedReader(store.directory)
    rows, cols = np.nonzero(mask)
    assert np.array_equal(reader.get(key, rows, cols), data[mask])
    assert np.array_equal(reader.get(key, [0, 399], [183, 183]), [-1., -1.])
    sampled_rows, sampled_cols = reader.sample(key, 200, np.random.default_rng(52))
    assert len(sampled_rows) == 200
    assert mask[sampled_rows, sampled_cols].all()
    assert len(set(zip(sampled_rows.tolist(), sampled_cols.tolist()))) == 200
    assert np.array_equal(reader.get(key, sampled_rows, sampled_cols), data[sampled_rows, sampled_cols])
    row, col = np.argwhere(~mask)[0]
    with pytest.raises(ValueError, match='unreachable'):
        reader.get(key, int(row), int(col))
    with pytest.raises(ValueError, match='range'):
        reader.get(key, -1, 0)


def test_packed_audit_checks_membership_and_values(small_packed):
    a, b, store, key, mask, data = small_packed
    reached = np.packbits(mask, axis=1, bitorder='little')
    result = a.audit_packed_key(store.directory, key, reached, int(mask.sum()))
    assert result['states'] == int(mask.sum())
    changed = mask.copy(); row, col = np.argwhere(~mask)[0]; changed[row, col] = True
    with pytest.raises(ValueError, match='reachability'):
        a.audit_packed_key(store.directory, key, np.packbits(changed, axis=1, bitorder='little'), int(changed.sum()))
    values = np.load(store.directory/'REV_3600.values.npy', mmap_mode='r+')
    values[0] = np.nan; values.flush()
    with pytest.raises(ValueError, match='finite'):
        a.audit_packed_key(store.directory, key, reached, int(mask.sum()))


def test_cold_pack_commits_before_dense_removal(tmp_path, monkeypatch):
    from stl.solver import leap_build as b
    monkeypatch.setattr(b, 'N', 8)
    store = b.TableStore(tmp_path/'tables'); key = ('REV', 3600)
    table = store.create(key); table[0, 0] = .3; table.flush(); del table
    files = {'tables/REV_3600.npy': b.file_hash(store.path(key))}
    records = [{'key': list(key), 'states': 1}]
    b.write_json(tmp_path/'checkpoint.json', {'files': files})
    old_checkpoint = (tmp_path/'checkpoint.json').read_bytes()
    real_write = b.write_json
    def fail_write(*args):
        raise OSError('injected checkpoint failure')
    monkeypatch.setattr(b, 'write_json', fail_write)
    with pytest.raises(OSError, match='injected'):
        b.commit_full_checkpoint(tmp_path, {}, records, files.copy(), store, 3059, 1.)
    assert store.path(key).exists()
    assert (tmp_path/'checkpoint.json').read_bytes() == old_checkpoint
    b.validate_files(tmp_path, files)
    monkeypatch.setattr(b, 'write_json', real_write)
    b.commit_full_checkpoint(tmp_path, {}, records, files, store, 3059, 1.)
    saved = json.loads((tmp_path/'checkpoint.json').read_text())
    b.validate_files(tmp_path, saved['files'])
    assert 'tables/REV_3600.npy' not in saved['files']
    assert 'tables/REV_3600.values.npy' in saved['files']
    assert not store.path(key).exists()


def test_explicit_lp_certifies_both_policies():
    from stl.solver.leap_audit import certify_matrix
    result = certify_matrix(np.array([[1., -1.], [-1., 1.]]))
    assert abs(result['value']) < 1e-12
    assert result['gap'] <= 1e-6
    assert np.allclose(result['drop'], [.5, .5])
    assert np.allclose(result['check'], [.5, .5])


def test_audit_uses_root_roles_and_reset_columns():
    from stl.solver.leap_audit import stage_inputs
    calls = []
    class Reader:
        def get(self, key, rows, cols):
            calls.append((key, np.asarray(rows), np.asarray(cols)))
            return np.full(np.broadcast_arrays(rows, cols)[0].shape, .25)
    s, f = stage_inputs(Reader(), ('H1', 12), 0, 0, None)
    assert np.array_equal(s, np.full(60, -.25))
    assert abs(f - (-.25*.95+.05)) < 1e-15
    assert calls[0][0] == ('H2', 12)
    assert np.array_equal(calls[0][2], np.arange(1, 61))
    assert calls[1][0] == ('REV', 1020)
    assert calls[1][1] == 0 and calls[1][2] == 1


def test_full_sweep_groups_compressed_failure_tables(tmp_path, monkeypatch):
    from stl.solver import leap_build as b
    from stl.solver.leap_profiles import profiles
    from stl.solver.leap_oracle import solve_stage
    path = Path('src/dth_compact/artifacts/V.npy')
    if not path.exists(): pytest.skip('DTH artifact unavailable')
    dth = np.load(path, mmap_mode='r'); p = profiles()
    first = np.full((b.N, 184), .1); first[:, -1] = -1.
    second = np.full((b.N, 184), .7); second[:, -1] = -1.
    store = b.TableStore(tmp_path)
    output = np.lib.format.open_memmap(tmp_path/'out.npy', mode='w+', shape=(2, 1), dtype=float)
    monkeypatch.setattr(store, 'create', lambda key: output)
    children = {('H2', 12): dth, ('REV', 1020): first, ('REV', 1021): second}
    monkeypatch.setattr(store, 'load', lambda key: children[key])
    bits = np.zeros((b.N, (b.N+7)//8), np.uint8); bits[:2, 0] = 1
    result = b.sweep_key(('H1', 12), bits, store, dth, min_clock=720, full=True)
    assert result['states'] == 2
    for pc, fail_value in [(0, .1), (1, .7)]:
        expected = solve_stage(-dth[0, p.succ[pc]], p.rev[pc]*-fail_value+1-p.rev[pc], False)
        assert abs(output[pc, 0]-expected.value) < 1e-9

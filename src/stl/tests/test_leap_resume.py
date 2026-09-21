"""We preserve source provenance when adding the HiGHS IPM retry."""
import hashlib
import json

import numpy as np
import pytest


def test_retry_upgrade_accepts_only_the_numerical_retry():
    from stl.solver.leap_resume import verify_retry_change
    old = '''
def solve_lp(s, f, window):
    result = linprog(method='highs')
    if not result.success:
        raise RuntimeError('status')
    lower, upper = bounds(result)
    if not math.isfinite(upper-lower) or upper-lower > GAP:
        raise RuntimeError('gap')
    return StageResult((lower+upper)/2, 3, lower, upper)
'''
    new = '''
def solve_lp(s, f, window, *, _method='highs'):
    result = linprog(method=_method)
    if not result.success:
        if _method == 'highs':
            return solve_lp(s, f, window, _method='highs-ipm')
        raise RuntimeError('status')
    lower, upper = bounds(result)
    if not math.isfinite(upper-lower) or upper-lower > GAP:
        if _method == 'highs':
            return solve_lp(s, f, window, _method='highs-ipm')
        raise RuntimeError('gap')
    return StageResult((lower+upper)/2, 3, lower, upper)
'''
    verify_retry_change(old, new)
    with pytest.raises(ValueError, match='retry'):
        verify_retry_change(old, new.replace('upper-lower > GAP', 'upper-lower > 1e-3'))
    with pytest.raises(ValueError, match='retry'):
        verify_retry_change(old, new.replace('/2, 3', '/3, 3'))


def test_source_snapshot_verifies_bytes_and_framed_digest(tmp_path):
    from stl.solver.leap_resume import snapshot_digest
    name = 'one.py'; data = b'x = 1\n'; (tmp_path/name).write_bytes(data)
    digest = hashlib.sha256(len(name).to_bytes(8, 'big')+name.encode()+len(data).to_bytes(8, 'big')+data).hexdigest()
    manifest = {'builder_sha256': digest, 'files': {name: hashlib.sha256(data).hexdigest()}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    assert snapshot_digest(tmp_path) == digest
    (tmp_path/name).write_bytes(b'x = 2\n')
    with pytest.raises(ValueError, match='hash'):
        snapshot_digest(tmp_path)


def test_prefix_audit_rejects_wrong_dense_membership(tmp_path, monkeypatch):
    from stl.solver import leap_build as b
    from stl.solver.leap_resume import verify_prefix_tables
    monkeypatch.setattr(b, 'N', 8)
    store = b.TableStore(tmp_path/'tables'); key = ('REV', 3600)
    table = store.create(key); table[0, 0] = .2; table.flush()
    reached = np.zeros((8, 23), np.uint8); reached[0, 0] = 1
    class Reach:
        counts = {key: 1}
        def load(self, key): return reached
    records = [{'key': list(key), 'states': 1}]
    assert verify_prefix_tables(tmp_path, records, Reach()) == 1
    table[0, 1] = .3; table.flush()
    with pytest.raises(ValueError, match='membership'):
        verify_prefix_tables(tmp_path, records, Reach())

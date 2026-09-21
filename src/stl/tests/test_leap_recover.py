"""We preserve certified batches after a rejected LP stops a key."""
import numpy as np
import pytest


def test_centered_upgrade_rejects_changes_to_certificate():
    from stl.solver.leap_recover import verify_centered_change
    old = '''
def solve_lp(s, f, window, *, _method='highs'):
    matrix = stage_matrix(s, f, window)
    rows = len(matrix)
    result = linprog(A_ub=np.c_[-matrix.T, np.ones(60)], method=_method)
    if not result.success:
        if _method == 'highs':
            return solve_lp(s, f, window, _method='highs-ipm')
        raise RuntimeError('status')
    lower, upper = bounds(result, matrix)
    if upper-lower > GAP:
        if _method == 'highs':
            return solve_lp(s, f, window, _method='highs-ipm')
        raise RuntimeError('gap')
    return (lower+upper)/2
'''
    new = old.replace("_method='highs'):", "_method='highs', _shift=False):")
    new = new.replace('    result =', '    lp_matrix = matrix-f if _shift else matrix\n    result =')
    new = new.replace('-matrix.T', '-lp_matrix.T')
    new = new.replace("        raise RuntimeError", "        if not _shift:\n            return solve_lp(s, f, window, _method='highs-ipm', _shift=True)\n        raise RuntimeError")
    verify_centered_change(old, new, 'solve_lp')
    with pytest.raises(ValueError, match='retry'):
        verify_centered_change(old, new.replace('bounds(result, matrix)', 'bounds(result, lp_matrix)'), 'solve_lp')
    with pytest.raises(ValueError, match='retry'):
        verify_centered_change(old, new.replace('> GAP', '> 1e-3'), 'solve_lp')


def test_partial_recovery_reuses_certified_lp_values(tmp_path, monkeypatch):
    from stl.solver import leap_build as b
    from stl.solver.leap_recover import recover_key, partial_membership
    monkeypatch.setattr(b, 'N', 8)
    key = ('REV', 3600); store = b.TableStore(tmp_path/'tables')
    table = store.create(key); table[0, 0] = .2; table[0, 1] = .3; table.flush()
    bitmap = np.zeros((8, 23), np.uint8); bitmap[0, 0] = 7
    assert partial_membership(table, bitmap) == 2
    def reject(*args):
        args[-2][:] = np.nan; args[-1][:] = 255
        return len(args[0])
    monkeypatch.setattr(b, 'require_kernel', lambda: reject)
    class Fallback:
        def solve(self, success, fail, success_table, fail_table, pcs, pds, window):
            assert len(pcs) == 1
            return np.full(1, .4), 1, 0
    record = recover_key(key, bitmap, store, np.zeros((b.N, 17012)), Fallback())
    result = np.load(store.path(key))
    assert np.array_equal(result[0, :3], [.2, .3, .4])
    assert record['states'] == record['failures'] == 3
    assert record['recovered_lp_solves'] == 2
    assert record['lp_solves'] == 3 and record['cached_failures'] == 0
    result[0, 3] = .5
    with pytest.raises(ValueError, match='membership'):
        partial_membership(result, bitmap)


def test_centered_audit_keeps_input_validation():
    from stl.solver.leap_recover import verify_centered_change
    old = '''
def certify_matrix(matrix, *, _method='highs'):
    if not finite(matrix):
        raise ValueError('input')
    result = linprog(A_ub=np.c_[-matrix.T, np.ones(60)], method=_method)
    if not result.success:
        if _method == 'highs':
            return certify_matrix(matrix, _method='highs-ipm')
        raise RuntimeError('status')
    if gap(result, matrix) > 1e-6:
        if _method == 'highs':
            return certify_matrix(matrix, _method='highs-ipm')
        raise RuntimeError('gap')
    return result
'''
    new = old.replace("_method='highs'):", "_method='highs', _shift=False):")
    new = new.replace('    result =', '    lp_matrix = matrix-matrix[-1, 0] if _shift else matrix\n    result =')
    new = new.replace('-matrix.T', '-lp_matrix.T')
    new = new.replace("        raise RuntimeError", "        if not _shift:\n            return certify_matrix(matrix, _method='highs-ipm', _shift=True)\n        raise RuntimeError")
    verify_centered_change(old, new, 'certify_matrix')


def test_reviewed_upgrade_binds_sources_and_protects_game(tmp_path, monkeypatch):
    from stl.solver import leap_build as b
    from stl.solver.leap_resume import _framed_digest
    from stl.solver.leap_recover import verify_reviewed_change
    old = tmp_path/'old'; new = tmp_path/'new'
    oracle = 'src/stl/solver/leap_oracle.py'; game = 'src/stl/engine/game.py'
    for root in (old,new):
        for name, text in [(oracle,'GAP = 1e-6\n'),(game,'CLOCK = 60\n')]:
            path = root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)
    names = [oracle,game]
    old_hash = _framed_digest(old,names)
    b.write_json(old/'manifest.json',{'files':{n:b.file_hash(old/n)for n in names},'builder_sha256':old_hash})
    (new/oracle).write_text('GAP = 1e-6\n# Native fallback added.\n')
    monkeypatch.setattr(b,'builder_hash',lambda:_framed_digest(new,names))
    review = {'old_builder_sha256':old_hash,'new_builder_sha256':b.builder_hash(),
              'changes':{oracle:{'before':b.file_hash(old/oracle),'after':b.file_hash(new/oracle)}}}
    verify_reviewed_change(old,new,review)
    (new/oracle).write_text('GAP = 1e-3\n')
    with pytest.raises(ValueError,match='review'):
        verify_reviewed_change(old,new,review)
    (new/oracle).write_text('GAP = 1e-6\n# Native fallback added.\n')
    (new/game).write_text('CLOCK = 61\n')
    review['new_builder_sha256'] = b.builder_hash()
    review['changes'][game] = {'before':b.file_hash(old/game),'after':b.file_hash(new/game)}
    with pytest.raises(ValueError,match='protected'):
        verify_reviewed_change(old,new,review)

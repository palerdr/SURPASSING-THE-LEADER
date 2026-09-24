"""We retain a certified prefix and partial key across the centered LP retry."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from stl.solver import leap_build as build
from stl.solver.leap_resume import _framed_digest, snapshot_digest, verify_prefix_tables


def verify_reviewed_change(snapshot, root, review):
    """We bind an explicit numerical upgrade to exact old and new source bytes."""
    allowed = {'src/stl/solver/'+name for name in
               ('leap_oracle.py','leap_audit.py','leap_lp.py','leap_support.py','leap_build.py','leap_recover.py')}
    allowed |= {'src/crates/stl_solver/src/'+name for name in ('leap.rs','leap_packing.rs','lib.rs')}
    if set(review['changes'])-allowed:
        raise ValueError('review changes a protected game or dependency file')
    old_files = json.loads((snapshot/'manifest.json').read_text())['files']
    if review['old_builder_sha256'] != snapshot_digest(snapshot) or review['new_builder_sha256'] != build.builder_hash():
        raise ValueError('review builder identity mismatch')
    names = set(old_files) | {name for name in allowed if (root/name).exists()}
    if _framed_digest(root,names) != review['new_builder_sha256']:
        raise ValueError('review omits a builder source file')
    changes = {}
    for name in names:
        before = old_files.get(name); after = build.file_hash(root/name)
        if before != after:
            if name not in allowed:
                raise ValueError(f'protected source changed: {name}')
            changes[name] = {'before':before,'after':after}
    if changes != review['changes']:
        raise ValueError('review source bytes differ')


def verify_centered_change(old, new, function_name):
    """We restrict source changes to a retry after the existing LP rejection."""
    expected = ast.parse(old)
    function = next(node for node in expected.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    if [arg.arg for arg in function.args.kwonlyargs] != ['_method']:
        raise ValueError('unexpected source before centered retry')
    function.args.kwonlyargs.append(ast.arg(arg='_shift'))
    function.args.kw_defaults.append(ast.Constant(value=False))
    shift = 'f' if function_name == 'solve_lp' else 'matrix[-1, 0]'
    calls = retries = 0
    for index, node in enumerate(function.body):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'linprog':
            function.body.insert(index, ast.parse(f'lp_matrix = matrix-{shift} if _shift else matrix').body[0])
            for child in ast.walk(node.value):
                if isinstance(child, ast.Attribute) and child.attr == 'T' and isinstance(child.value, ast.Name) and child.value.id == 'matrix':
                    child.value.id = 'lp_matrix'; calls += 1
            break
    for node in function.body:
        if (isinstance(node, ast.If) and isinstance(node.body[-1], ast.Raise)
                and isinstance(node.body[-1].exc, ast.Call)
                and isinstance(node.body[-1].exc.func, ast.Name)
                and node.body[-1].exc.func.id == 'RuntimeError'):
            arguments = ', '.join(arg.arg for arg in function.args.args)
            node.body.insert(-1, ast.parse(f"if not _shift:\n    return {function_name}({arguments}, _method='highs-ipm', _shift=True)").body[0])
            retries += 1
    if calls != 1 or retries != 2 or ast.dump(expected) != ast.dump(ast.parse(new)):
        raise ValueError('source change exceeds the centered retry')


def partial_membership(table, bitmap):
    """We accept finite partial values only at reachable cells."""
    cols = table.shape[1]-1
    if table.dtype != np.float64 or table.shape[0] != len(bitmap) or not np.all(table[:, -1] == -1.):
        raise ValueError('partial table schema mismatch')
    retained = 0
    for lo in range(0, len(bitmap), 64):
        block = table[lo:lo+64, :cols]
        reached = np.unpackbits(bitmap[lo:lo+64], axis=1, count=cols, bitorder='little').astype(bool)
        finite = np.isfinite(block)
        if np.isinf(block).any() or np.any(finite & ~reached) or np.any(np.abs(block[finite]) > 1+1e-9):
            raise ValueError('partial table membership or value mismatch')
        retained += int(finite.sum())
    return retained


def recover_key(key, bitmap, store, dth, fallback):
    """We rerun the kernel and reuse the partial table's certified LP values."""
    table = np.load(store.path(key), mmap_mode='r+')
    if table.shape != (build.N, build.key_columns(key)+1):
        raise ValueError('partial table shape mismatch')
    retained = partial_membership(table, bitmap)
    p = build.profiles()
    recovered = 0

    class Store:
        def create(self, requested):
            if requested != key:
                raise ValueError('partial recovery key mismatch')
            return table

        def __getattr__(self, name):
            return getattr(store, name)

    class Fallback:
        def __getattr__(self, name):
            return getattr(fallback, name)

        def solve(self, success, fail, success_table, fail_table, pcs, pds, window):
            nonlocal recovered
            columns = p.idx0[pds] if key[0] == 'REV' else pds
            values = np.array(table[pcs, columns], copy=True)
            cached = np.isfinite(values); recovered += int(cached.sum())
            missing = ~cached
            solved = reused = 0
            if missing.any():
                values[missing], solved, reused = fallback.solve(success, fail, success_table,
                        fail_table, pcs[missing], pds[missing], window)
            return values, solved, reused+int(cached.sum())

    proxy = Store()
    record = build.sweep_key(key, bitmap, proxy, dth, min_clock=720, full=True, fallback=Fallback())
    store.write_seconds = proxy.write_seconds
    record['lp_solves'] += recovered
    record['cached_failures'] -= recovered
    record['recovered_lp_solves'] = recovered
    record['retained_states'] = retained
    record['lp_seconds_scope'] = 'recovery pass only; previous partial LP time was not checkpointed'
    return record


def continue_centered(source, fresh_reach, destination, dth_path, *, native_review=None):
    """We archive prior source identities before recovering the partial key."""
    source = Path(source); fresh_reach = Path(fresh_reach); destination = Path(destination)
    if destination.exists():
        raise ValueError('continuation destination must not exist')
    root = Path(__file__).resolve().parents[3]
    checkpoint = json.loads((source/'checkpoint.json').read_text())
    partial = json.loads((source/'partial.json').read_text())
    snapshot = source/'source-snapshot'; old_hash = snapshot_digest(snapshot)
    if (old_hash != checkpoint['identity']['builder_sha256'] or partial['identity'] != checkpoint['identity']
            or partial['parent_checkpoint_sha256'] != build.file_hash(source/'checkpoint.json')):
        raise ValueError('partial source or checkpoint identity mismatch')
    if checkpoint['identity'].get('mode') != 'full' or build.file_hash(dth_path) != checkpoint['identity']['dth_sha256']:
        raise ValueError('partial DTH identity mismatch')
    key = tuple(partial['key']); partial_path = source/'tables'/f'{build.key_name(key)}.npy'
    if build.file_hash(partial_path) != partial['table_sha256']:
        raise ValueError('partial table hash mismatch')
    old_files = json.loads((snapshot/'manifest.json').read_text())['files']
    retries = {'src/stl/solver/leap_oracle.py': 'solve_lp', 'src/stl/solver/leap_audit.py': 'certify_matrix'}
    current_hash = build.builder_hash()
    if native_review is not None:
        review = json.loads(Path(native_review).read_text())
        verify_reviewed_change(snapshot,root,review)
    else:
        for name in old_files:
            old = (snapshot/name).read_bytes(); new = (root/name).read_bytes()
            if name in retries:
                verify_centered_change(old.decode(), new.decode(), retries[name])
            elif old != new:
                raise ValueError(f'unrelated source change: {name}')
        names = set(old_files) | {str(Path(__file__).resolve().relative_to(root))}
        if _framed_digest(root, names) != current_hash:
            raise ValueError('unrelated builder source addition')
    build.validate_files(source, checkpoint['files'])
    reach = build._load_reach(fresh_reach)
    if reach.by_type != build.TOTALS or reach.window_count != 470336555:
        raise ValueError('fresh reachability coverage mismatch')
    for name, digest in json.loads((fresh_reach/'manifest.json').read_text())['files'].items():
        if checkpoint['files'].get(f'reachability/{name}') != digest:
            raise ValueError(f'fresh reachability differs from prefix: {name}')
    remaining = set(reach.counts)-{tuple(r['key']) for r in checkpoint['records']}
    if key != max(remaining, key=build.key_clock):
        raise ValueError('partial key must follow the completed prefix')
    table = np.load(partial_path, mmap_mode='r')
    if table.shape != (build.N, build.key_columns(key)+1) or partial_membership(table, reach.load(key)) != partial['retained_states']:
        raise ValueError('partial retained count mismatch')
    print('Verifying completed prefix and partial table.', flush=True)
    states = verify_prefix_tables(source, checkpoint['records'], reach)
    destination.mkdir(parents=True)
    files = {}

    def retain(path, name):
        target = destination/name; target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target); files[name] = build.file_hash(target)

    for name in checkpoint['files']:
        if not name.startswith('reachability/'):
            retain(source/name, name)
    for path in fresh_reach.iterdir():
        if path.is_file():
            retain(path, f'reachability/{path.name}')
    archive = 'provenance/native-packing' if native_review is not None else 'provenance/centered-retry'
    for name in [*old_files, 'manifest.json']:
        retain(snapshot/name, f'{archive}/source/{name}')
    for name in ('checkpoint.json', 'partial.json'):
        retain(source/name, f'{archive}/{name}')
    if native_review is not None:
        retain(Path(native_review),f'{archive}/review.json')
        # We archive the new source as well as the prefix's source identity.
        names = set(old_files) | set(review['changes'])
        current_files = {}
        for name in names:
            target = destination/'source-snapshot'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(root/name,target)
            current_files[name] = build.file_hash(target)
            files[str(target.relative_to(destination))] = current_files[name]
        build.write_json(destination/'source-snapshot/manifest.json',
                         {'builder_sha256':current_hash,'files':current_files})
        files['source-snapshot/manifest.json'] = build.file_hash(destination/'source-snapshot/manifest.json')
    retain(partial_path, f'{archive}/partial-table.npy')
    # We copy the partial table because the next sweep writes into it.
    shutil.copyfile(partial_path, destination/'tables'/partial_path.name)
    provenance = {'prefix_builder_sha256': old_hash, 'continuation_builder_sha256': current_hash,
                  'prefix_states': states, 'partial': partial,
                  'source_change': 'centered IPM retry after both prior methods reject; certificate uses the original matrix',
                  'validation': 'restricted AST change, file hashes, regenerated reachability, prefix membership and finite-value audit'}
    if native_review is not None:
        provenance['source_change'] = 'reviewed native packing fallback and offset retries; original 1e-6 certificate'
        provenance['validation'] = 'exact reviewed source hashes, unchanged protected files, regenerated reachability, complete prefix membership audit'
    build.write_json(destination/f'{archive}/continuation.json', provenance)
    files[f'{archive}/continuation.json'] = build.file_hash(destination/f'{archive}/continuation.json')
    checkpoint['identity']['builder_sha256'] = current_hash
    # The continuation's run_full keys use the current residue path; each
    # prefix record keeps its own builder hash.
    checkpoint['identity']['residue'] = build.RESIDUE
    for record in checkpoint['records']:
        record.setdefault('builder_sha256', old_hash)
    checkpoint['files'] = files
    build.write_json(destination/'checkpoint.json', checkpoint)
    from stl.solver.leap_lp import Fallback, NativeFallback
    store = build.TableStore(destination/'tables')
    for name, value in checkpoint['io'].items():
        setattr(store, name, value)
    started = time.perf_counter()
    with (NativeFallback() if native_review is not None else Fallback()) as fallback:
        record = recover_key(key, reach.load(key), store, np.load(dth_path, mmap_mode='r'), fallback)
    if record['recovered_lp_solves'] != partial['recovered_lp_solves']:
        raise ValueError('recovered LP count differs from the recorded partial work')
    record['sweep_seconds'] = time.perf_counter()-started+partial['previous_work_wall_seconds']
    record['recovered_work_wall_seconds'] = partial['previous_work_wall_seconds']
    record['retained_builder_sha256'] = old_hash
    checkpoint['records'].append(record)
    files[f'tables/{partial_path.name}'] = build.file_hash(store.path(key))
    build.commit_full_checkpoint(destination, checkpoint['identity'], checkpoint['records'], files,
                                 store, build.key_clock(key), checkpoint['elapsed_seconds']+record['sweep_seconds'])
    print(json.dumps(record), flush=True)
    return build.run_full(dth_path, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('fresh_reach', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--dth-path', type=Path, default=Path('src/dth_compact/artifacts/V.npy'))
    parser.add_argument('--native-review', type=Path)
    args = parser.parse_args()
    continue_centered(args.source, args.fresh_reach, args.destination, args.dth_path, native_review=args.native_review)


if __name__ == '__main__':
    main()

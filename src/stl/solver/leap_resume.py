"""We retain certified prefix values across the HiGHS IPM retry upgrade."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from stl.solver import leap_build as build
from stl.solver.leap_audit import audit_packed_key


def verify_retry_change(old, new, function_name='solve_lp'):
    """We permit only an IPM retry after the original solver rejects a result."""
    expected = ast.parse(old)
    function = next(node for node in expected.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    if function.args.kwonlyargs:
        raise ValueError('unexpected source before retry upgrade')
    function.args.kwonlyargs.append(ast.arg(arg='_method'))
    function.args.kw_defaults.append(ast.Constant(value='highs'))
    methods = 0
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'linprog':
            for keyword in node.keywords:
                if keyword.arg == 'method' and isinstance(keyword.value, ast.Constant) and keyword.value.value == 'highs':
                    keyword.value = ast.Name(id='_method', ctx=ast.Load()); methods += 1
    retries = 0
    for node in function.body:
        if (isinstance(node, ast.If) and len(node.body) == 1 and isinstance(node.body[0], ast.Raise)
                and isinstance(node.body[0].exc, ast.Call) and isinstance(node.body[0].exc.func, ast.Name)
                and node.body[0].exc.func.id == 'RuntimeError'):
            arguments = ', '.join(arg.arg for arg in function.args.args)
            node.body.insert(0, ast.parse(f"if _method == 'highs':\n    return {function_name}({arguments}, _method='highs-ipm')").body[0])
            retries += 1
    if methods != 1 or retries != 2 or ast.dump(expected) != ast.dump(ast.parse(new)):
        raise ValueError('source change exceeds the approved numerical retry')


def _framed_digest(root, names):
    digest = hashlib.sha256()
    for name in sorted(names):
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('source path must stay inside the snapshot')
        encoded = name.encode(); data = (Path(root)/name).read_bytes()
        digest.update(len(encoded).to_bytes(8, 'big')); digest.update(encoded)
        digest.update(len(data).to_bytes(8, 'big')); digest.update(data)
    return digest.hexdigest()


def snapshot_digest(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    build.validate_files(directory, manifest['files'])
    digest = _framed_digest(directory, manifest['files'])
    if digest != manifest['builder_sha256']:
        raise ValueError('source snapshot hash mismatch')
    return digest


def verify_prefix_tables(directory, records, reach):
    directory = Path(directory); total = 0; seen = set()
    for record in records:
        key = tuple(record['key']); cols = build.key_columns(key)
        if key in seen or record['states'] != reach.counts[key]:
            raise ValueError('prefix record coverage mismatch')
        seen.add(key)
        path = directory/'tables'/f'{build.key_name(key)}.npy'
        reached = reach.load(key)
        if not path.exists():
            audit_packed_key(directory/'tables', key, reached, record['states'])
        else:
            table = np.load(path, mmap_mode='r')
            if table.dtype != np.float64 or table.shape != (build.N, cols+1) or not np.all(table[:, -1] == -1.):
                raise ValueError('prefix table schema mismatch')
            for lo in range(0, build.N, 64):
                block = table[lo:lo+64, :cols]
                expected = np.unpackbits(reached[lo:lo+64], axis=1, count=cols, bitorder='little').astype(bool)
                if not np.array_equal(np.isfinite(block), expected) or np.isinf(block).any():
                    raise ValueError('prefix table membership mismatch')
                if np.any(np.abs(block[expected]) > 1+1e-9):
                    raise ValueError('prefix utility outside range')
        total += record['states']
    return total


def continue_after_ipm(source, fresh_reach, destination):
    """We create a continuation artifact without changing the original prefix."""
    source = Path(source); destination = Path(destination); fresh_reach = Path(fresh_reach)
    if destination.exists():
        raise ValueError('continuation destination must not exist')
    root = Path(__file__).resolve().parents[3]
    checkpoint = json.loads((source/'checkpoint.json').read_text())
    snapshot = source/'source-snapshot'; old_hash = snapshot_digest(snapshot)
    if old_hash != checkpoint['identity']['builder_sha256'] or checkpoint['identity'].get('mode') != 'full':
        raise ValueError('prefix source identity mismatch')
    old_files = json.loads((snapshot/'manifest.json').read_text())['files']
    retries = {'src/stl/solver/leap_oracle.py': 'solve_lp',
               'src/stl/solver/leap_audit.py': 'certify_matrix'}
    for name in old_files:
        old = (snapshot/name).read_bytes(); new = (root/name).read_bytes()
        if name in retries:
            verify_retry_change(old.decode(), new.decode(), retries[name])
        elif old != new:
            raise ValueError(f'unrelated source change: {name}')
    current_hash = build.builder_hash()
    names = set(old_files) | {str(Path(__file__).resolve().relative_to(root))}
    if _framed_digest(root, names) != current_hash:
        raise ValueError('unrelated builder source addition')
    build.validate_files(source, checkpoint['files'])
    reach = build._load_reach(fresh_reach)
    if reach.by_type != build.TOTALS or reach.window_count != 470336555:
        raise ValueError('fresh reachability coverage mismatch')
    fresh_files = json.loads((fresh_reach/'manifest.json').read_text())['files']
    for name, digest in fresh_files.items():
        if checkpoint['files'].get(f'reachability/{name}') != digest:
            raise ValueError(f'fresh reachability differs from prefix: {name}')
    print('Verifying prefix table membership and finite values.', flush=True)
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
    for name in [*old_files, 'manifest.json']:
        retain(snapshot/name, f'provenance/source/{name}')
    retain(source/'checkpoint.json', 'provenance/prefix-checkpoint.json')
    retain(source/'reachability/manifest.json', 'provenance/prefix-reachability-manifest.json')
    provenance = {'prefix_builder_sha256': old_hash, 'continuation_builder_sha256': current_hash,
                  'prefix_states': states, 'prefix_keys': len(checkpoint['records']),
                  'source_change': 'HiGHS IPM retry after the original LP rejection; original successful returns remain unchanged',
                  'validation': 'source AST restriction, file hashes, regenerated reachability, complete prefix membership and finite-value audit'}
    build.write_json(destination/'provenance/continuation.json', provenance)
    files['provenance/continuation.json'] = build.file_hash(destination/'provenance/continuation.json')
    checkpoint['identity']['builder_sha256'] = current_hash
    for record in checkpoint['records']:
        record['builder_sha256'] = old_hash
    checkpoint['files'] = files
    build.write_json(destination/'checkpoint.json', checkpoint)
    print(json.dumps(provenance, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('fresh_reach', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    continue_after_ipm(args.source, args.fresh_reach, args.destination)


if __name__ == '__main__':
    main()

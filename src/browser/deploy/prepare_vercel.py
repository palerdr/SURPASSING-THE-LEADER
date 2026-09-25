"""Prepare the browser and certified artifact for a prebuilt Vercel deployment."""

from __future__ import annotations

import json
import argparse
import py_compile
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from arena.presentation.scene_art import PANEL_ROOT, SceneArt
from arena.presentation.sprites import encode_png
from browser.deploy.manifest import RUNTIME_FILES
from dth.agent import CompleteDTHAgent

# The bundle, the client, and the Vercel project link, by repository path.
BUNDLE = "src/browser/build/vercel"
WEBCLIENT = "src/browser/webclient"
VERCEL_LINK = "src/browser/.vercel/project.json"
# The packages the function installs. Their versions come from the browser
# project's lock, which pins the environment that the browser tests run in.
# The root uv.lock still enters the bundle, because the DTH digest labels it.
BROWSER_LOCK = "src/browser/uv.lock"
BUNDLE_PACKAGES = ("fastapi", "numpy", "scipy", "httpx", "uvicorn")


def add_bytecode(target: Path) -> int:
    """Compile every Python file the built function maps and map the results.

    Vercel ships no bytecode and forbids writing it, so each new process
    compiled numpy, scipy, and FastAPI from source before its first answer.
    `unchecked-hash` files load whatever modification time the bundle gives
    the sources. Run this after `vercel build` and before `vercel deploy`.
    """
    runtime = (3, 13)
    if sys.version_info[:2] != runtime:
        raise SystemExit(f"bytecode must come from Python {runtime[0]}.{runtime[1]}")
    written = 0
    for config in (target / ".vercel/output/functions").glob("*.func/.vc-config.json"):
        settings = json.loads(config.read_text())
        mapped = settings["filePathMap"]
        for name, source in list(mapped.items()):
            if not name.endswith(".py") or name.startswith("_vendor/pip/"):
                continue
            tag = f"{Path(name).stem}.{sys.implementation.cache_tag}.pyc"
            compiled = (target / source).parent / "__pycache__" / tag
            try:
                py_compile.compile(
                    str(target / source),
                    cfile=str(compiled),
                    dfile=f"/var/task/{name}",
                    doraise=True,
                    invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
                )
            except py_compile.PyCompileError:
                # Vendored packages carry template and Python 2 files that never import.
                continue
            mapped[str(Path(name).parent / "__pycache__" / tag)] = str(
                compiled.relative_to(target)
            )
            written += 1
        config.write_text(json.dumps(settings, indent=2))
    return written


def copy_runtime_sources(root: Path, target: Path) -> None:
    """Copy each file of the runtime manifest to ``runtime/`` in the bundle.

    The copy keeps each file's path from the repository root, so the DTH
    digest labels ``src/dth/...`` and ``uv.lock`` still match at cold start.
    """
    for entry in RUNTIME_FILES:
        destination = target / "runtime" / entry.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / entry.path, destination)


def locked_versions(lock: Path, names: tuple[str, ...] = BUNDLE_PACKAGES) -> dict[str, str]:
    """Return the one version that ``lock`` pins for each package in ``names``."""
    found: dict[str, set[str]] = {name: set() for name in names}
    for package in tomllib.loads(lock.read_text(encoding="utf-8")).get("package", []):
        if package.get("name") in found:
            found[package["name"]].add(str(package["version"]))
    for name, versions in found.items():
        if len(versions) != 1:
            raise SystemExit(f"{lock} must pin one version of {name}; it pins {sorted(versions)}")
    return {name: versions.pop() for name, versions in found.items()}


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path, default=root / "src/dth/artifacts/complete_full_v1"
    )
    parser.add_argument(
        "--bytecode",
        action="store_true",
        help="after `vercel build`: add compiled bytecode to the built function",
    )
    options = parser.parse_args()
    if options.bytecode:
        count = add_bytecode(root / BUNDLE)
        print(f"Added {count} bytecode files to the built function.")
        return
    agent = CompleteDTHAgent(options.artifact)
    opening = agent.decide((0, 0, 0, 0))
    print(
        f"Certified artifact: opening value={opening.value}, gap={opening.saddle_gap}"
    )
    target = root / BUNDLE
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    subprocess.run(
        ["npm", "--prefix", str(root / WEBCLIENT), "run", "build"],
        check=True,
    )
    shutil.copytree(root / WEBCLIENT / "dist", target / "public")
    # The manifest holds the root uv.lock, so runtime/uv.lock comes with it.
    copy_runtime_sources(root, target)
    artifact = target / "runtime/src/dth/artifacts/complete_fast_v1"
    artifact.mkdir(parents=True)
    for name in ("tablebase.json", "value.npy", "solver_kind.npy"):
        shutil.copy2(options.artifact / name, artifact / name)
    for (character, pose), frames in SceneArt.load().poses.items():
        folder = target / "public/art" / character / pose
        folder.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames):
            (folder / f"{index}.png").write_bytes(encode_png(frame))
    panel = target / "public/art/panel/stl_rules"
    panel.parent.mkdir(parents=True)
    shutil.copy2(PANEL_ROOT / "stl_rules.png", panel)
    (target / "app.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "root = Path(__file__).resolve().parent\n"
        'sys.path.insert(0, str(root / "runtime/src"))\n'
        "from browser.deploy.production import create_production_app\n"
        'app = create_production_app(root / "runtime/src/dth/artifacts/complete_fast_v1")\n'
    )
    pins = locked_versions(root / BROWSER_LOCK)
    dependencies = ", ".join(f'"{name}=={version}"' for name, version in pins.items())
    (target / "pyproject.toml").write_text(
        '[project]\nname = "stl-browser"\nversion = "0.1.0"\n'
        'requires-python = ">=3.13,<3.14"\n'
        f"dependencies = [{dependencies}]\n"
    )
    config = {
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "framework": "fastapi",
        "functions": {"app.py": {"maxDuration": 60, "includeFiles": "runtime/**"}},
        "headers": [
            {
                "source": "/art/panel/stl_rules",
                "headers": [{"key": "Content-Type", "value": "image/png"}],
            }
        ],
    }
    (target / "vercel.json").write_text(json.dumps(config, indent=2) + "\n")
    link = root / VERCEL_LINK
    if link.exists():
        (target / ".vercel").mkdir()
        shutil.copy2(link, target / ".vercel/project.json")
    print(
        f"Prepared {target}. Use vercel build, then this command with --bytecode, "
        "then vercel deploy --prebuilt --archive=tgz."
    )


if __name__ == "__main__":
    main()

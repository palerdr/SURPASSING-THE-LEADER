"""Prepare the browser and certified artifact for a prebuilt Vercel deployment."""

from __future__ import annotations

import json
import argparse
import shutil
import subprocess
from pathlib import Path

from arena.sprites import encode_png
from arena.tui import SceneArt
from dth.agent import CompleteDTHAgent


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path, default=root / "src/dth/artifacts/complete_full_v1"
    )
    options = parser.parse_args()
    agent = CompleteDTHAgent(options.artifact)
    opening = agent.decide((0, 0, 0, 0))
    print(
        f"Certified artifact: opening value={opening.value}, gap={opening.saddle_gap}"
    )
    target = root / "src/arena/web/build/vercel"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    subprocess.run(
        ["npm", "--prefix", str(root / "src/arena/webclient"), "run", "build"],
        check=True,
    )
    shutil.copytree(root / "src/arena/webclient/dist", target / "public")
    files = [
        "arena/__init__.py",
        "arena/agent.py",
        "arena/contracts.py",
        "arena/dth_adapter.py",
        "arena/session.py",
        "arena/tui.py",
        "arena/sprites.py",
        "arena/web/__init__.py",
        "arena/web/app.py",
        "arena/web/schema.py",
        "arena/web/hosted.py",
        "arena/web/ledger.py",
        "arena/web/names.py",
        "arena/web/production.py",
        "dth/__init__.py",
        "dth/agent.py",
        "dth/solver.py",
        "dth/packed.py",
        "dth/support_solver.py",
        "dth/complete_tablebase.py",
        "dth/fast_kernel.py",
        "dth/fast_kernel.c",
        "stl/__init__.py",
    ]
    files.extend(
        str(p.relative_to(root / "src")) for p in (root / "src/stl/engine").glob("*.py")
    )
    for name in files:
        destination = target / "runtime/src" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "src" / name, destination)
    shutil.copy2(root / "uv.lock", target / "runtime/uv.lock")
    artifact = target / "runtime/src/dth/artifacts/complete_fast_v1"
    artifact.mkdir(parents=True)
    for name in ("tablebase.json", "value.npy", "solver_kind.npy"):
        shutil.copy2(options.artifact / name, artifact / name)
    for (character, pose), frames in SceneArt.load(root / "art/sprites").poses.items():
        folder = target / "public/art" / character / pose
        folder.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames):
            (folder / f"{index}.png").write_bytes(encode_png(frame))
    panel = target / "public/art/panel/stl_rules"
    panel.parent.mkdir(parents=True)
    shutil.copy2(root / "art/panels/stl_rules.png", panel)
    (target / "app.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "root = Path(__file__).resolve().parent\n"
        'sys.path.insert(0, str(root / "runtime/src"))\n'
        "from arena.web.production import create_production_app\n"
        'app = create_production_app(root / "runtime/src/dth/artifacts/complete_fast_v1")\n'
    )
    (target / "pyproject.toml").write_text(
        '[project]\nname = "stl-browser"\nversion = "0.1.0"\n'
        'requires-python = ">=3.13,<3.14"\n'
        'dependencies = ["fastapi==0.141.1", "numpy==2.5.0", '
        '"scipy==1.18.0", "httpx==0.28.1", "uvicorn==0.52.1"]\n'
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
    link = root / ".vercel/project.json"
    if link.exists():
        (target / ".vercel").mkdir()
        shutil.copy2(link, target / ".vercel/project.json")
    print(
        f"Prepared {target}. Use vercel build, then vercel deploy --prebuilt --archive=tgz."
    )


if __name__ == "__main__":
    main()

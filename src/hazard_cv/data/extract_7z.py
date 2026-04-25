"""Extract .7z ETH3D archives (py7zr preferred; falls back to 7z CLI)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def extract_7z_archive(archive: Path, out_dir: Path, overwrite: bool = True) -> None:
    """Extract archive into out_dir (creates subfolders as inside the archive)."""
    archive = Path(archive).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        raise FileNotFoundError(archive)

    seven = shutil.which("7z") or shutil.which("7za")
    if seven:
        cmd = [seven, "x", str(archive), f"-o{out_dir}", "-y"]
        if overwrite:
            cmd.append("-aoa")
        subprocess.run(cmd, check=True)
        return

    try:
        import py7zr
    except ImportError as e:
        raise RuntimeError(
            "Need either `7z` on PATH or `pip install py7zr` to extract .7z files."
        ) from e

    with py7zr.SevenZipFile(archive, mode="r") as z:
        z.extractall(path=out_dir)

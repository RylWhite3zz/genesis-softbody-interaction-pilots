"""Paths and write isolation for this experiment only."""

from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
ASSET_ROOT = Path(
    os.environ.get("SOFTBODY_ASSET_ROOT", REPOSITORY_ROOT / "assets" / "representative")
).expanduser().resolve()
ROBOT_URDF = Path(
    os.environ.get("PIPER_URDF", REPOSITORY_ROOT / "robot_assets" / "piper_with_gripper.urdf")
).expanduser().resolve()
GENESIS_PYTHON = Path(os.environ.get("GENESIS_PYTHON", sys.executable)).expanduser().resolve()
REPORT_ROOT = PROJECT_ROOT / "reports"
VIDEO_ROOT = PROJECT_ROOT / "videos"
CACHE_ROOT = PROJECT_ROOT / ".cache"
WORK_ROOT = PROJECT_ROOT / "work"


def require_within_project(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"write path must stay inside {PROJECT_ROOT}: {resolved}") from exc
    return resolved


def configure_writable_runtime() -> None:
    cache_paths = {
        "NUMBA_CACHE_DIR": CACHE_ROOT / "numba",
        "GS_CACHE_FILE_PATH": CACHE_ROOT / "genesis",
        "XDG_CACHE_HOME": CACHE_ROOT / "xdg",
        "MPLCONFIGDIR": CACHE_ROOT / "matplotlib",
        "CUDA_CACHE_PATH": CACHE_ROOT / "cuda",
        "QD_CACHE_DIR": CACHE_ROOT / "quadrants",
        "QD_OFFLINE_CACHE_FILE_PATH": CACHE_ROOT / "quadrants",
        "TORCH_EXTENSIONS_DIR": CACHE_ROOT / "torch_extensions",
        "TRITON_CACHE_DIR": CACHE_ROOT / "triton",
        "TMPDIR": CACHE_ROOT / "tmp",
    }
    for name, path in cache_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    tempfile.tempdir = str(CACHE_ROOT / "tmp")


def isolate_genesis_particle_sampler() -> dict[str, str]:
    """Redirect the stock PBS sampler's executable and temporary output."""

    import genesis.utils.particle as particle_utils

    original_misc = particle_utils.miu
    installed_source = Path(original_misc.get_src_dir()).resolve()
    runtime_source = CACHE_ROOT / "genesis_runtime"
    runtime_ext = runtime_source / "ext"
    runtime_ext.mkdir(parents=True, exist_ok=True)
    source_binary = installed_source / "ext" / "VolumeSampling"
    runtime_binary = runtime_ext / "VolumeSampling"
    if not source_binary.is_file():
        raise FileNotFoundError(source_binary)
    if not runtime_binary.is_file() or runtime_binary.stat().st_size != source_binary.stat().st_size:
        shutil.copyfile(source_binary, runtime_binary)
        runtime_binary.chmod(runtime_binary.stat().st_mode | stat.S_IXUSR)

    class _MiscProxy:
        def get_src_dir(self) -> str:
            return str(runtime_source)

        def __getattr__(self, name: str) -> Any:
            return getattr(original_misc, name)

    particle_utils.miu = _MiscProxy()
    return {
        "installed_source_read_only": str(installed_source),
        "runtime_source": str(runtime_source),
        "pbs_binary": str(runtime_binary),
        "temporary_directory": tempfile.gettempdir(),
    }

"""Startup utilities for validating and preparing the local CosyVoice runtime."""

import logging
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Sequence

from app.config import AppSettings, COSYVOICE_ROOT_DIR, MATCHA_TTS_ROOT_DIR

_logger = logging.getLogger(__name__)


def _configure_local_python_path() -> None:
    additions: list[str] = []
    for path in (COSYVOICE_ROOT_DIR, MATCHA_TTS_ROOT_DIR):
        if not path.is_dir():
            continue

        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.append(resolved)
        additions.append(resolved)

    if not additions:
        return

    current_entries = [
        entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry
    ]

    # Keep worker subprocess imports aligned with the main process.
    changed = False
    for entry in additions:
        if entry in current_entries:
            continue
        current_entries.append(entry)
        changed = True

    if changed:
        os.environ["PYTHONPATH"] = os.pathsep.join(current_entries)


def _cosyvoice_codebase_exists() -> bool:
    return (
        COSYVOICE_ROOT_DIR.is_dir()
        and (COSYVOICE_ROOT_DIR / "README.md").is_file()
        and (COSYVOICE_ROOT_DIR / "cosyvoice").is_dir()
    )


def _model_dir_exists(model_dir: str | Path) -> bool:
    return Path(model_dir).is_dir()


def _ttsfrd_dir_exists(ttsfrd_path: Path) -> bool:
    return ttsfrd_path.is_dir()


def _prepare_ttsfrd_resource(ttsfrd_path: Path) -> Path | None:
    def _safe_extractall(zip_ref: zipfile.ZipFile, destination: Path) -> None:
        resolved_destination = destination.resolve()
        for member in zip_ref.infolist():
            member_path = (destination / member.filename).resolve()
            if not member_path.is_relative_to(resolved_destination):
                raise ValueError(f"unsafe archive path detected: {member.filename}")
        zip_ref.extractall(destination)

    zip_path = ttsfrd_path / "resource.zip"
    resource_path = ttsfrd_path / "resource"

    if resource_path.is_dir():
        return resource_path

    if not zip_path.is_file():
        _logger.warning("Expected ttsfrd resource archive not found at %s.", zip_path)
        return None

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            _safe_extractall(zip_ref, ttsfrd_path)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        _logger.warning(
            "Failed to extract ttsfrd resources from %s: %s",
            zip_path,
            exc,
        )
        return None

    if not resource_path.is_dir():
        _logger.warning(
            "TTSFRD resource directory missing after extraction: %s", resource_path
        )
        return None

    _logger.info("Prepared ttsfrd resources at %s", resource_path)
    return resource_path


def _find_best_wheel(prefix: str, wheel_files: Sequence[Path]) -> Path | None:
    major = sys.version_info.major
    minor = sys.version_info.minor
    tags_to_check = [
        f"cp{major}{minor}",
        f"py{major}{minor}",
        f"cp{major}",
        f"py{major}",
    ]

    candidates = sorted(
        (wheel for wheel in wheel_files if wheel.name.startswith(prefix)),
        reverse=True,
    )
    for tag in tags_to_check:
        for wheel in candidates:
            if tag in wheel.name:
                return wheel
    return None


def ensure_ttsfrd_installed(ttsfrd_path: Path) -> bool:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        _logger.warning("ttsfrd installation is only supported on Linux x86_64.")
        return False

    if not _ttsfrd_dir_exists(ttsfrd_path):
        _logger.warning("TTSFRD directory does not exist: %s", ttsfrd_path)
        return False

    resource_path = _prepare_ttsfrd_resource(ttsfrd_path)
    if resource_path is None:
        return False

    try:
        __import__("ttsfrd")
        _logger.info("ttsfrd is already installed.")
        return True
    except ImportError:
        _logger.info("ttsfrd is not installed, attempting local wheel installation.")

    wheel_files = list(ttsfrd_path.glob("*.whl"))
    deps_wheel = _find_best_wheel("ttsfrd_dependency-", wheel_files)
    core_wheel = _find_best_wheel("ttsfrd-", wheel_files)
    if deps_wheel is None or core_wheel is None:
        _logger.warning("Suitable ttsfrd wheels were not found in %s.", ttsfrd_path)
        return False

    for wheel_file in (deps_wheel, core_wheel):
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", str(wheel_file)],
                check=True,
            )
            _logger.info("Installed ttsfrd component from %s", wheel_file)
        except (OSError, subprocess.CalledProcessError) as exc:
            _logger.warning("Failed to install ttsfrd from %s: %s", wheel_file, exc)
            return False

    return True


def prepare_runtime(settings: AppSettings) -> None:
    """Prepare the process environment before the app imports runtime-heavy modules."""
    _configure_local_python_path()

    if not _cosyvoice_codebase_exists():
        raise RuntimeError(
            f"Bundled CosyVoice codebase not found at {COSYVOICE_ROOT_DIR}."
        )

    if not _model_dir_exists(settings.model_path):
        raise RuntimeError(f"Model directory not found: {settings.model_path}.")

    if settings.ttsfrd_path.exists() and not ensure_ttsfrd_installed(
        settings.ttsfrd_path
    ):
        _logger.warning(
            "Failed to prepare ttsfrd. CosyVoice will fall back to other text frontends if available."
        )

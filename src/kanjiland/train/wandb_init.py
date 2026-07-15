"""W&B run initialization (CLAUDE.md rule 4).

Every training run — debug runs included — logs to W&B project
``kanjiland``. This module captures the reproducibility metadata the human
will inevitably want six months later: git SHA and dirty flag, the exact
config file that produced the run, seed, hostname, CUDA and torch
versions, and (when available) GPU model.

Offline mode: set ``WANDB_MODE=offline`` in the environment. This is
respected transparently by the underlying ``wandb.init`` call, so CI and
sandboxed training don't require special handling here.
"""

from __future__ import annotations

import getpass
import platform
import socket
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import wandb
import yaml

DEFAULT_PROJECT = "kanjiland"


def init_run(
    config_path: str | Path,
    *,
    project: str = DEFAULT_PROJECT,
    run_name: str | None = None,
    extra_tags: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> wandb.sdk.wandb_run.Run:
    """Load the YAML config at ``config_path`` and start a W&B run.

    Returns the wandb Run so the caller can log metrics. The full config
    is uploaded as an artifact so we can always reproduce the run.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    with config_path.open() as f:
        config = yaml.safe_load(f)

    wb_cfg = config.get("wandb", {})
    project = wb_cfg.get("project", project)
    tags = list(wb_cfg.get("tags", []))
    if extra_tags:
        tags.extend(extra_tags)
    name = run_name or config.get("run_name")

    metadata_payload = _collect_metadata(config_path)
    if extra_metadata:
        metadata_payload.update(extra_metadata)

    run = wandb.init(
        project=project,
        name=name,
        config=config,
        tags=tags,
        settings=wandb.Settings(_disable_stats=False),
    )
    # metadata goes into a separate namespace so hyperparameter sweeps
    # don't get polluted with hostnames and git SHAs.
    run.config.update({"_run": metadata_payload}, allow_val_change=True)

    artifact = wandb.Artifact(
        name=f"config-{config_path.stem}",
        type="config",
        metadata={"path": str(config_path)},
    )
    artifact.add_file(str(config_path))
    run.log_artifact(artifact)
    return run


def _collect_metadata(config_path: Path) -> dict[str, Any]:
    return {
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "git_branch": _git_branch(),
        "config_path": str(config_path.resolve()),
        "hostname": socket.gethostname(),
        "user": _safe(getpass.getuser),
        "python_version": platform.python_version(),
        "torch_version": _pkg_version("torch"),
        "cuda_available": _cuda_available(),
        "cuda_device": _cuda_device_name(),
        "package_versions": _kanjiland_version(),
    }


def _git_sha() -> str | None:
    return _run_git(["rev-parse", "HEAD"])


def _git_branch() -> str | None:
    return _run_git(["rev-parse", "--abbrev-ref", "HEAD"])


def _git_dirty() -> bool | None:
    status = _run_git(["status", "--porcelain"])
    if status is None:
        return None
    return bool(status.strip())


def _run_git(args: list[str]) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _kanjiland_version() -> str | None:
    return _pkg_version("kanjiland")


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _cuda_device_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None

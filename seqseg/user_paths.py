"""Persisted user paths for nnU-Net and common SeqSeg train/run defaults.

Stored at ``~/.seqseg/paths.yaml`` (override with ``SEQSEG_PATHS_FILE``).

Environment variables always win over the file when both are set.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

import yaml

NNUNET_KEYS = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")

_FILE_TO_ENV = {
    "nnunet_raw": "nnUNet_raw",
    "nnunet_preprocessed": "nnUNet_preprocessed",
    "nnunet_results": "nnUNet_results",
}

_DEFAULT_NNUNET_ROOT = Path.home() / "nnunet_data"


def default_paths_file() -> Path:
    override = os.environ.get("SEQSEG_PATHS_FILE")
    if override:
        return Path(os.path.expanduser(override)).resolve()
    return (Path.home() / ".seqseg" / "paths.yaml").resolve()


def _expand(path: Optional[str]) -> Optional[str]:
    if path is None or path == "":
        return None
    return str(Path(os.path.expanduser(str(path))).resolve())


def load_paths(path_file: Optional[Path] = None) -> Dict[str, Any]:
    """Load path config; missing file → empty dict."""
    pf = path_file or default_paths_file()
    if not pf.is_file():
        return {}
    with open(pf, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid SeqSeg paths file (expected mapping): {pf}")
    return data


def save_paths(data: Mapping[str, Any], path_file: Optional[Path] = None) -> Path:
    """Write path config (creates parent dirs). Returns file path."""
    pf = path_file or default_paths_file()
    pf.parent.mkdir(parents=True, exist_ok=True)
    clean: Dict[str, Any] = {}
    for key in (
        "nnunet_raw",
        "nnunet_preprocessed",
        "nnunet_results",
        "data_dir",
        "outdir",
    ):
        val = data.get(key)
        if val is not None and val != "":
            clean[key] = _expand(str(val))
    with open(pf, "w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, default_flow_style=False, sort_keys=False)
    return pf


def clear_paths(path_file: Optional[Path] = None) -> Optional[Path]:
    """Delete the paths file if it exists. Returns path if removed."""
    pf = path_file or default_paths_file()
    if pf.is_file():
        pf.unlink()
        return pf
    return None


def default_nnunet_layout(root: Optional[str] = None) -> Dict[str, str]:
    """Standard nnU-Net trio under a root directory."""
    base = Path(os.path.expanduser(root)) if root else _DEFAULT_NNUNET_ROOT
    base = base.resolve()
    return {
        "nnunet_raw": str(base / "nnUNet_raw"),
        "nnunet_preprocessed": str(base / "nnUNet_preprocessed"),
        "nnunet_results": str(base / "nnUNet_results"),
    }


def ensure_nnunet_dirs(paths: Mapping[str, Any]) -> List[str]:
    """Create nnU-Net directories if configured. Returns created/existing paths."""
    created: List[str] = []
    for key in ("nnunet_raw", "nnunet_preprocessed", "nnunet_results"):
        p = paths.get(key)
        if not p:
            continue
        path = Path(os.path.expanduser(str(p)))
        path.mkdir(parents=True, exist_ok=True)
        created.append(str(path.resolve()))
    return created


def merge_path_updates(
    current: Mapping[str, Any],
    *,
    nnunet_root: Optional[str] = None,
    nnunet_raw: Optional[str] = None,
    nnunet_preprocessed: Optional[str] = None,
    nnunet_results: Optional[str] = None,
    data_dir: Optional[str] = None,
    outdir: Optional[str] = None,
) -> Dict[str, Any]:
    """Return updated paths dict from CLI-style fields."""
    out = dict(current)
    if nnunet_root is not None:
        out.update(default_nnunet_layout(nnunet_root))
    if nnunet_raw is not None:
        out["nnunet_raw"] = _expand(nnunet_raw)
    if nnunet_preprocessed is not None:
        out["nnunet_preprocessed"] = _expand(nnunet_preprocessed)
    if nnunet_results is not None:
        out["nnunet_results"] = _expand(nnunet_results)
    if data_dir is not None:
        out["data_dir"] = _expand(data_dir)
    if outdir is not None:
        out["outdir"] = _expand(outdir)
    return out


def apply_nnunet_env(
    env: Optional[MutableMapping[str, str]] = None,
    path_file: Optional[Path] = None,
    *,
    create_dirs: bool = False,
) -> MutableMapping[str, str]:
    """
    Fill missing ``nnUNet_*`` keys in ``env`` from the paths file.

    Existing environment values are left unchanged.
    """
    target: MutableMapping[str, str] = os.environ if env is None else env
    stored = load_paths(path_file)
    if create_dirs:
        ensure_nnunet_dirs(stored)
    for file_key, env_key in _FILE_TO_ENV.items():
        if target.get(env_key):
            continue
        val = stored.get(file_key)
        if val:
            target[env_key] = _expand(str(val))  # type: ignore[assignment]
    return target


def resolve_path(
    key: str,
    explicit: Optional[str] = None,
    path_file: Optional[Path] = None,
) -> Optional[str]:
    """
    Resolve a user path.

    Priority: explicit CLI value → env (for nnU-Net keys) → paths file.
    """
    if explicit is not None and explicit != "":
        return _expand(explicit)

    env_key = _FILE_TO_ENV.get(key)
    if env_key and os.environ.get(env_key):
        return _expand(os.environ[env_key])

    stored = load_paths(path_file)
    val = stored.get(key)
    return _expand(str(val)) if val else None


def format_paths_report(path_file: Optional[Path] = None) -> str:
    """Human-readable status of file + effective nnU-Net env."""
    pf = path_file or default_paths_file()
    stored = load_paths(pf)
    lines = [f"Paths file: {pf}" + (" (missing)" if not pf.is_file() else "")]
    if stored:
        for key in (
            "nnunet_raw",
            "nnunet_preprocessed",
            "nnunet_results",
            "data_dir",
            "outdir",
        ):
            if key in stored:
                lines.append(f"  {key}: {stored[key]}")
    else:
        lines.append("  (no saved paths)")

    lines.append("Effective nnU-Net env (env var wins over file):")
    for file_key, env_key in _FILE_TO_ENV.items():
        env_val = os.environ.get(env_key)
        file_val = stored.get(file_key)
        if env_val:
            lines.append(f"  {env_key}: {env_val}  [from environment]")
        elif file_val:
            lines.append(f"  {env_key}: {_expand(str(file_val))}  [from paths file]")
        else:
            lines.append(f"  {env_key}: (unset)")
    return "\n".join(lines)


def shell_exports(path_file: Optional[Path] = None) -> str:
    """Print ``export`` lines for the effective nnU-Net trio (for ``eval``)."""
    apply_nnunet_env()
    lines: List[str] = []
    for env_key in NNUNET_KEYS:
        val = os.environ.get(env_key)
        if val:
            lines.append(f'export {env_key}="{val}"')
    return "\n".join(lines)

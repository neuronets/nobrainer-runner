"""YAML profile loader for nobrainer-runner.

Profiles are stored in ``~/.nobrainer-runner/profiles/<name>.yaml`` and
describe the compute backend (Slurm, AWS Batch, GCP Batch) together with
default job parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_PROFILES_DIR = Path.home() / ".nobrainer-runner" / "profiles"

_REQUIRED_COMMON = {"backend"}
_REQUIRED_SLURM = {"partition"}
_REQUIRED_AWS = {"instance_type"}
_REQUIRED_GCP = {"instance_type"}


def load_profile(name: str, profiles_dir: str | Path | None = None) -> dict[str, Any]:
    """Load a compute profile by name.

    Parameters
    ----------
    name : str
        Profile filename without the ``.yaml`` extension.
    profiles_dir : path or None
        Override the default profile directory
        ``~/.nobrainer-runner/profiles/``.

    Returns
    -------
    dict
        Parsed profile data.

    Raises
    ------
    FileNotFoundError
        If the profile file does not exist.
    ValueError
        If required fields are missing.
    """
    base_dir = Path(profiles_dir) if profiles_dir is not None else _PROFILES_DIR
    path = base_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Profile '{name}' not found at {path}. "
            f"Create it or check your profiles directory."
        )

    with path.open() as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    _validate(data, name)
    return data


def _validate(data: dict[str, Any], name: str) -> None:
    """Raise ``ValueError`` if required fields are absent."""
    missing_common = _REQUIRED_COMMON - set(data)
    if missing_common:
        raise ValueError(
            f"Profile '{name}' is missing required field(s): {missing_common}"
        )

    backend = data["backend"].lower()
    defaults = data.get("defaults", {})

    if backend == "slurm":
        missing = _REQUIRED_SLURM - set(defaults)
    elif backend == "aws":
        missing = _REQUIRED_AWS - set(defaults)
    elif backend == "gcp":
        missing = _REQUIRED_GCP - set(defaults)
    else:
        raise ValueError(
            f"Profile '{name}': unknown backend '{backend}'. "
            "Must be one of: slurm, aws, gcp."
        )

    if missing:
        raise ValueError(
            f"Profile '{name}' (backend={backend}) is missing "
            f"'defaults.{missing.pop()}' field."
        )

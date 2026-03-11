"""Slurm backend for nobrainer-runner.

Renders and submits sbatch scripts.  Supports dry-run mode (print only),
job status polling, cancellation, and output retrieval.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Any

_SBATCH_TEMPLATE = textwrap.dedent(
    """\
    #!/bin/bash
    #SBATCH --job-name={job_name}
    #SBATCH --partition={partition}
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --gpus={gpus}
    #SBATCH --time={time}
    #SBATCH --output={log_dir}/{job_name}_%j.out
    #SBATCH --error={log_dir}/{job_name}_%j.err

    {command}
    """
)


def submit(
    profile: dict[str, Any],
    command: str,
    gpus: int = 1,
    job_name: str = "nobrainer-job",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Submit a command to a Slurm cluster.

    Parameters
    ----------
    profile : dict
        Parsed profile (must contain ``backend: slurm`` and
        ``defaults.partition``).
    command : str
        Shell command to execute inside the sbatch script.
    gpus : int
        Number of GPUs to request.
    job_name : str
        Slurm job name (also used as log prefix).
    dry_run : bool
        If ``True``, print the sbatch script instead of submitting.

    Returns
    -------
    dict
        ``{"job_id": str, "dry_run": bool, "script": str}``
    """
    defaults = profile.get("defaults", {})
    partition = defaults["partition"]
    time_limit = defaults.get("time", "04:00:00")
    log_dir = defaults.get("log_dir", str(Path.home() / "nobrainer-logs"))
    os.makedirs(log_dir, exist_ok=True)

    script = _SBATCH_TEMPLATE.format(
        job_name=job_name,
        partition=partition,
        gpus=gpus,
        time=time_limit,
        log_dir=log_dir,
        command=command,
    )

    if dry_run:
        return {"job_id": None, "dry_run": True, "script": script}

    result = subprocess.run(
        ["sbatch", "--parsable"],
        input=script,
        capture_output=True,
        text=True,
        check=True,
    )
    job_id = result.stdout.strip()
    return {"job_id": job_id, "dry_run": False, "script": script}


def status(job_id: str) -> dict[str, Any]:
    """Query the state of a Slurm job.

    Parameters
    ----------
    job_id : str
        Slurm job ID returned by :func:`submit`.

    Returns
    -------
    dict
        ``{"job_id": str, "status": str, "failure_reason": str | None}``

    Notes
    -----
    When the job state is ``PREEMPTED``, the returned dict includes
    ``"failure_reason": "preempted"`` (satisfies FR-014a).
    """
    result = subprocess.run(
        ["squeue", "-j", str(job_id), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    state: str = "UNKNOWN"
    failure_reason: str | None = None

    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            jobs = data.get("jobs", [])
            if jobs:
                state = jobs[0].get("job_state", "UNKNOWN")
        except (json.JSONDecodeError, KeyError):
            pass
    else:
        # squeue returns non-zero when job is no longer in the queue
        state = "COMPLETED"

    if state == "PREEMPTED":
        failure_reason = "preempted"
        state = "FAILED"

    return {"job_id": str(job_id), "status": state, "failure_reason": failure_reason}


def cancel(job_id: str) -> dict[str, Any]:
    """Cancel a Slurm job.

    Returns
    -------
    dict
        ``{"job_id": str, "cancelled": bool}``
    """
    result = subprocess.run(
        ["scancel", str(job_id)],
        capture_output=True,
        text=True,
        check=False,
    )
    return {"job_id": str(job_id), "cancelled": result.returncode == 0}


def results(
    job_id: str,
    log_dir: str | Path | None = None,
    job_name: str = "nobrainer-job",
) -> dict[str, Any]:
    """Read stdout/stderr from Slurm log files.

    Parameters
    ----------
    job_id : str
        Slurm job ID.
    log_dir : path or None
        Directory containing Slurm log files.  Defaults to
        ``~/nobrainer-logs``.
    job_name : str
        Job name used as log file prefix.

    Returns
    -------
    dict
        ``{"stdout": str, "stderr": str, "log_dir": str}``
    """
    if log_dir is None:
        log_dir = Path.home() / "nobrainer-logs"
    log_dir = Path(log_dir)

    stdout_path = log_dir / f"{job_name}_{job_id}.out"
    stderr_path = log_dir / f"{job_name}_{job_id}.err"

    stdout = stdout_path.read_text() if stdout_path.exists() else ""
    stderr = stderr_path.read_text() if stderr_path.exists() else ""

    return {"stdout": stdout, "stderr": stderr, "log_dir": str(log_dir)}

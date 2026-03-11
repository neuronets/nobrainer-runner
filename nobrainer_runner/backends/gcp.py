"""GCP Batch backend for nobrainer-runner.

Requires ``google-cloud-batch``
(install with ``pip install nobrainer-runner[gcp]``).
"""

from __future__ import annotations

from typing import Any


def _client():
    try:
        from google.cloud import batch_v1  # type: ignore[import-untyped]

        return batch_v1.BatchServiceClient(), batch_v1
    except ImportError as exc:
        raise ImportError(
            "google-cloud-batch is required for the GCP backend. "
            "Install it with: pip install nobrainer-runner[gcp]"
        ) from exc


def submit(
    profile: dict[str, Any],
    command: str,
    gpus: int = 1,
    job_name: str = "nobrainer-job",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Submit a job to GCP Batch.

    Parameters
    ----------
    profile : dict
        Parsed profile with ``defaults.project``, ``defaults.region``,
        ``defaults.instance_type``.
    command : str
        Shell command to run in the container.
    gpus : int
        Number of GPUs per task.
    job_name : str
        GCP Batch job ID.
    dry_run : bool
        If ``True``, return the would-be job spec without submitting.

    Returns
    -------
    dict
        ``{"job_id": str | None, "dry_run": bool}``
    """
    defaults = profile.get("defaults", {})
    project = defaults.get("project", "")
    region = defaults.get("region", "us-central1")

    job_spec = {
        "name": job_name,
        "project": project,
        "region": region,
        "command": command,
        "gpus": gpus,
    }

    if dry_run:
        return {"job_id": None, "dry_run": True, "spec": job_spec}

    client, batch_v1 = _client()
    parent = f"projects/{project}/locations/{region}"
    runnable = batch_v1.Runnable()
    runnable.container = batch_v1.Runnable.Container()
    runnable.container.commands = ["bash", "-c", command]

    task = batch_v1.TaskSpec()
    task.runnables = [runnable]

    group = batch_v1.TaskGroup()
    group.task_spec = task

    job = batch_v1.Job()
    job.task_groups = [group]

    request = batch_v1.CreateJobRequest(parent=parent, job=job, job_id=job_name)
    response = client.create_job(request)
    return {"job_id": response.name, "dry_run": False}


def status(job_id: str, project: str = "", region: str = "us-central1") -> dict[str, Any]:
    """Poll the state of a GCP Batch job.

    Returns
    -------
    dict
        ``{"job_id": str, "status": str, "failure_reason": str | None}``
    """
    client, batch_v1 = _client()
    try:
        job = client.get_job(name=job_id)
        state = batch_v1.JobStatus.State(job.status.state).name
        return {"job_id": job_id, "status": state, "failure_reason": None}
    except Exception as exc:
        return {"job_id": job_id, "status": "UNKNOWN", "failure_reason": str(exc)}


def cancel(job_id: str) -> dict[str, Any]:
    """Cancel a GCP Batch job.

    Returns
    -------
    dict
        ``{"job_id": str, "cancelled": bool}``
    """
    client, _ = _client()
    try:
        client.delete_job(name=job_id)
        return {"job_id": job_id, "cancelled": True}
    except Exception:
        return {"job_id": job_id, "cancelled": False}


def results(job_id: str, **kwargs: Any) -> dict[str, Any]:
    """Read Cloud Logging entries for a GCP Batch job.

    Returns
    -------
    dict
        ``{"stdout": str, "stderr": str}``
    """
    try:
        from google.cloud import logging as cloud_logging  # type: ignore[import-untyped]

        log_client = cloud_logging.Client()
        entries = log_client.list_entries(filter_=f'resource.labels.job_id="{job_id}"')
        lines = [entry.payload for entry in entries if isinstance(entry.payload, str)]
        return {"stdout": "\n".join(lines), "stderr": ""}
    except Exception as exc:
        return {"stdout": "", "stderr": str(exc)}

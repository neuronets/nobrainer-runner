"""AWS Batch backend for nobrainer-runner.

Requires ``boto3`` (install with ``pip install nobrainer-runner[aws]``).
"""

from __future__ import annotations

from typing import Any


def _client():
    try:
        import boto3  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for the AWS backend. "
            "Install it with: pip install nobrainer-runner[aws]"
        ) from exc
    return boto3.client("batch")


def submit(
    profile: dict[str, Any],
    command: str,
    gpus: int = 1,
    job_name: str = "nobrainer-job",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Submit a job to AWS Batch.

    Parameters
    ----------
    profile : dict
        Parsed profile with ``defaults.instance_type``,
        ``defaults.job_queue``, and ``defaults.job_definition``.
    command : str
        Shell command to pass as the container command override.
    gpus : int
        Number of GPUs to request (applied via ``resourceRequirements``).
    job_name : str
        AWS Batch job name.
    dry_run : bool
        If ``True``, return the would-be request payload without submitting.

    Returns
    -------
    dict
        ``{"job_id": str | None, "dry_run": bool}``
    """
    defaults = profile.get("defaults", {})
    job_queue = defaults.get("job_queue", "nobrainer-queue")
    job_definition = defaults.get("job_definition", "nobrainer-job-def")

    container_overrides: dict[str, Any] = {
        "command": ["bash", "-c", command],
        "resourceRequirements": [{"type": "GPU", "value": str(gpus)}],
    }

    payload = {
        "jobName": job_name,
        "jobQueue": job_queue,
        "jobDefinition": job_definition,
        "containerOverrides": container_overrides,
    }

    if dry_run:
        return {"job_id": None, "dry_run": True, "payload": payload}

    response = _client().submit_job(**payload)
    return {"job_id": response["jobId"], "dry_run": False}


def status(job_id: str) -> dict[str, Any]:
    """Query the status of an AWS Batch job.

    Returns
    -------
    dict
        ``{"job_id": str, "status": str, "failure_reason": str | None}``
    """
    response = _client().describe_jobs(jobs=[job_id])
    jobs = response.get("jobs", [])
    if not jobs:
        return {"job_id": job_id, "status": "UNKNOWN", "failure_reason": None}
    job = jobs[0]
    state = job.get("status", "UNKNOWN")
    failure_reason = job.get("statusReason") if state == "FAILED" else None
    return {"job_id": job_id, "status": state, "failure_reason": failure_reason}


def cancel(job_id: str) -> dict[str, Any]:
    """Cancel an AWS Batch job.

    Returns
    -------
    dict
        ``{"job_id": str, "cancelled": bool}``
    """
    try:
        _client().terminate_job(jobId=job_id, reason="Cancelled by nobrainer-runner")
        return {"job_id": job_id, "cancelled": True}
    except Exception:
        return {"job_id": job_id, "cancelled": False}


def results(job_id: str, **kwargs: Any) -> dict[str, Any]:
    """Download stdout/stderr from CloudWatch for an AWS Batch job.

    Returns
    -------
    dict
        ``{"stdout": str, "stderr": str}``
    """
    try:
        import boto3  # type: ignore[import-untyped]

        logs_client = boto3.client("logs")
        job_info = _client().describe_jobs(jobs=[job_id])
        jobs = job_info.get("jobs", [])
        if not jobs:
            return {"stdout": "", "stderr": ""}
        log_stream = jobs[0].get("container", {}).get("logStreamName", "")
        if not log_stream:
            return {"stdout": "", "stderr": ""}
        events = logs_client.get_log_events(
            logGroupName="/aws/batch/job",
            logStreamName=log_stream,
            startFromHead=True,
        )
        stdout = "\n".join(e["message"] for e in events.get("events", []))
        return {"stdout": stdout, "stderr": ""}
    except Exception as exc:
        return {"stdout": "", "stderr": str(exc)}

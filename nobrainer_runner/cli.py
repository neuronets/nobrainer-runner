"""Click CLI for nobrainer-runner.

Entry point: ``nobrainer-runner``
Commands: submit, status, results, cancel
"""

from __future__ import annotations

import json
import sys

import click

from .profiles import load_profile

_option_kwds: dict = {"show_default": True}


@click.group()
@click.version_option(message="%(prog)s %(version)s")
def cli() -> None:
    """nobrainer-runner: GPU job dispatch for Slurm, AWS Batch, and GCP Batch."""


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("command")
@click.option(
    "-p", "--profile", required=True, help="Profile name (from ~/.nobrainer-runner/profiles/)."
)
@click.option("--gpus", type=int, default=1, help="Number of GPUs.", **_option_kwds)
@click.option(
    "--job-name", default="nobrainer-job", help="Job name.", **_option_kwds
)
@click.option("--dry-run", is_flag=True, help="Print the job spec without submitting.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "text"]),
    default="text",
    help="Output format.",
    **_option_kwds,
)
def submit(
    *,
    command: str,
    profile: str,
    gpus: int,
    job_name: str,
    dry_run: bool,
    output_format: str,
) -> None:
    """Submit COMMAND to the compute backend specified by PROFILE."""
    try:
        prof = load_profile(profile)
    except (FileNotFoundError, ValueError) as exc:
        click.echo(click.style(f"ERROR: {exc}", fg="red"), err=True)
        sys.exit(1)

    backend = prof["backend"].lower()
    try:
        if backend == "slurm":
            from .backends.slurm import submit as _submit
        elif backend == "aws":
            from .backends.aws import submit as _submit  # type: ignore[assignment]
        elif backend == "gcp":
            from .backends.gcp import submit as _submit  # type: ignore[assignment]
        else:
            click.echo(click.style(f"ERROR: unknown backend '{backend}'", fg="red"), err=True)
            sys.exit(1)
    except ImportError as exc:
        click.echo(click.style(f"ERROR: {exc}", fg="red"), err=True)
        sys.exit(1)

    result = _submit(prof, command, gpus=gpus, job_name=job_name, dry_run=dry_run)

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
    else:
        if dry_run:
            click.echo(result.get("script", ""))
        else:
            click.echo(f"Submitted job: {result['job_id']}")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("job_id")
@click.option(
    "-p", "--profile", required=True, help="Profile name (backend is inferred from profile)."
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "text"]),
    default="text",
    **_option_kwds,
)
def status(*, job_id: str, profile: str, output_format: str) -> None:
    """Check the status of JOB_ID."""
    try:
        prof = load_profile(profile)
    except (FileNotFoundError, ValueError) as exc:
        click.echo(click.style(f"ERROR: {exc}", fg="red"), err=True)
        sys.exit(1)

    backend = prof["backend"].lower()
    if backend == "slurm":
        from .backends.slurm import status as _status
    elif backend == "aws":
        from .backends.aws import status as _status  # type: ignore[assignment]
    elif backend == "gcp":
        from .backends.gcp import status as _status  # type: ignore[assignment]
    else:
        click.echo(click.style(f"ERROR: unknown backend '{backend}'", fg="red"), err=True)
        sys.exit(1)

    result = _status(job_id)
    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"Job {job_id}: {result['status']}")
        if result.get("failure_reason"):
            click.echo(f"  Reason: {result['failure_reason']}")


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("job_id")
@click.option("-p", "--profile", required=True, help="Profile name.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "text"]),
    default="text",
    **_option_kwds,
)
def cancel(*, job_id: str, profile: str, output_format: str) -> None:
    """Cancel JOB_ID."""
    try:
        prof = load_profile(profile)
    except (FileNotFoundError, ValueError) as exc:
        click.echo(click.style(f"ERROR: {exc}", fg="red"), err=True)
        sys.exit(1)

    backend = prof["backend"].lower()
    if backend == "slurm":
        from .backends.slurm import cancel as _cancel
    elif backend == "aws":
        from .backends.aws import cancel as _cancel  # type: ignore[assignment]
    elif backend == "gcp":
        from .backends.gcp import cancel as _cancel  # type: ignore[assignment]
    else:
        click.echo(click.style(f"ERROR: unknown backend '{backend}'", fg="red"), err=True)
        sys.exit(1)

    result = _cancel(job_id)
    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
    else:
        if result["cancelled"]:
            click.echo(f"Cancelled job {job_id}.")
        else:
            click.echo(f"Failed to cancel job {job_id}.", err=True)
            sys.exit(1)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("report_log", type=click.Path(exists=True))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "text"]),
    default="text",
    **_option_kwds,
)
def results(*, report_log: str, output_format: str) -> None:
    """Parse pytest REPORT_LOG and display pass/fail/coverage summary."""
    from .results import parse_results

    try:
        data = parse_results(report_log)
    except FileNotFoundError as exc:
        click.echo(click.style(f"ERROR: {exc}", fg="red"), err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(data["summary"])
        if data["tests_failed"] > 0 or data["tests_error"] > 0:
            sys.exit(1)


if __name__ == "__main__":
    cli()

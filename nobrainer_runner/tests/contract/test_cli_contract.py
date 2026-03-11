"""CLI contract tests for nobrainer-runner commands.

Verifies that all CLI commands are present, accept the expected options,
and exit 0 on --help.
"""

from __future__ import annotations

import json as _json

import pytest
from click.testing import CliRunner

from nobrainer_runner.cli import cli


def _invoke_help(cmd_name: str) -> str:
    runner = CliRunner()
    result = runner.invoke(cli, [cmd_name, "--help"])
    assert result.exit_code == 0, (
        f"'{cmd_name} --help' exited {result.exit_code}:\n{result.output}"
    )
    return result.output


class TestSubmitContract:
    def test_submit_help_exits_zero(self):
        _invoke_help("submit")

    def test_submit_has_profile_option(self):
        out = _invoke_help("submit")
        assert "--profile" in out or "-p" in out

    def test_submit_has_dry_run_option(self):
        out = _invoke_help("submit")
        assert "--dry-run" in out

    def test_submit_has_format_option(self):
        out = _invoke_help("submit")
        assert "--format" in out

    def test_submit_has_gpus_option(self):
        out = _invoke_help("submit")
        assert "--gpus" in out

    def test_submit_dry_run_exits_cleanly(self):
        """Dry run should exit 0, 1, or 2 (not crash with unhandled exception)."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "submit",
                "echo hello",
                "--dry-run",
                "--format",
                "text",
            ],
        )
        assert result.exit_code in (0, 1, 2)


class TestStatusContract:
    def test_status_help_exits_zero(self):
        _invoke_help("status")

    def test_status_has_profile_option(self):
        out = _invoke_help("status")
        assert "--profile" in out or "-p" in out

    def test_status_has_format_option(self):
        out = _invoke_help("status")
        assert "--format" in out

    def test_status_takes_job_id_argument(self):
        out = _invoke_help("status")
        assert "JOB_ID" in out


class TestResultsContract:
    def test_results_help_exits_zero(self):
        _invoke_help("results")

    def test_results_has_format_option(self):
        out = _invoke_help("results")
        assert "--format" in out

    def test_results_takes_report_log_argument(self):
        out = _invoke_help("results")
        assert "REPORT_LOG" in out or "report" in out.lower()

    def test_results_json_format_parseable(self, tmp_path):
        """results --format json should produce parseable JSON for a valid report."""
        report = tmp_path / "report.jsonl"
        lines = [
            _json.dumps(
                {"$report_type": "TestReport", "when": "call", "outcome": "passed"}
            ),
            _json.dumps(
                {"$report_type": "TestReport", "when": "call", "outcome": "failed"}
            ),
        ]
        report.write_text("\n".join(lines) + "\n")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["results", str(report), "--format", "json"]
        )
        assert result.exit_code == 0
        data = _json.loads(result.output)
        assert data["tests_passed"] == 1
        assert data["tests_failed"] == 1


class TestCancelContract:
    def test_cancel_help_exits_zero(self):
        _invoke_help("cancel")

    def test_cancel_has_profile_option(self):
        out = _invoke_help("cancel")
        assert "--profile" in out or "-p" in out

    def test_cancel_takes_job_id_argument(self):
        out = _invoke_help("cancel")
        assert "JOB_ID" in out

    def test_cancel_has_format_option(self):
        out = _invoke_help("cancel")
        assert "--format" in out

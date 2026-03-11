"""Unit tests for the Click CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from nobrainer_runner.cli import cli


def _write_profile(tmp_path, name: str, data: dict) -> Path:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profiles_dir / f"{name}.yaml"
    profile_path.write_text(yaml.dump(data))
    return profiles_dir


class TestSubmitCommand:
    def test_dry_run_text_output(self, tmp_path):
        profiles_dir = _write_profile(
            tmp_path, "test", {"backend": "slurm", "defaults": {"partition": "gpu"}}
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "submit",
                "--profile", "test",
                "--dry-run",
                "echo hello",
            ],
            env={"HOME": str(tmp_path)},
            catch_exceptions=False,
        )
        with patch("nobrainer_runner.profiles._PROFILES_DIR", profiles_dir):
            result = runner.invoke(
                cli,
                ["submit", "--profile", "test", "--dry-run", "echo hello"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0

    def test_dry_run_json_output(self, tmp_path):
        profiles_dir = _write_profile(
            tmp_path, "json_test", {"backend": "slurm", "defaults": {"partition": "gpu"}}
        )
        runner = CliRunner()
        with patch("nobrainer_runner.profiles._PROFILES_DIR", profiles_dir):
            result = runner.invoke(
                cli,
                ["submit", "--profile", "json_test", "--dry-run", "--format", "json", "echo hi"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["dry_run"] is True

    def test_missing_profile_exits_1(self, tmp_path):
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        runner = CliRunner()
        with patch("nobrainer_runner.profiles._PROFILES_DIR", profiles_dir):
            result = runner.invoke(
                cli,
                ["submit", "--profile", "nonexistent", "echo hi"],
            )
        assert result.exit_code != 0


class TestResultsCommand:
    def _make_log(self, tmp_path, records):
        log = tmp_path / "report.jsonl"
        with log.open("w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        return log

    def test_text_output(self, tmp_path):
        log = self._make_log(
            tmp_path,
            [
                {"when": "call", "outcome": "passed"},
                {"when": "call", "outcome": "passed"},
            ],
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["results", str(log)], catch_exceptions=False)
        assert result.exit_code == 0
        assert "2 passed" in result.output

    def test_json_output(self, tmp_path):
        log = self._make_log(tmp_path, [{"when": "call", "outcome": "passed"}])
        runner = CliRunner()
        result = runner.invoke(
            cli, ["results", "--format", "json", str(log)], catch_exceptions=False
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["tests_passed"] == 1

    def test_exits_1_on_failures(self, tmp_path):
        log = self._make_log(tmp_path, [{"when": "call", "outcome": "failed"}])
        runner = CliRunner()
        result = runner.invoke(cli, ["results", str(log)])
        assert result.exit_code == 1

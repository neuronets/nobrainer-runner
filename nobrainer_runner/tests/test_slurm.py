"""Unit tests for the Slurm backend."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nobrainer_runner.backends.slurm import cancel, results, status, submit

_SLURM_PROFILE = {
    "backend": "slurm",
    "defaults": {
        "partition": "gpu",
        "time": "02:00:00",
        "log_dir": "/tmp/nobrainer-logs",
    },
}


class TestSlurmSubmit:
    def test_dry_run_returns_script(self):
        result = submit(_SLURM_PROFILE, "python train.py", dry_run=True)
        assert result["dry_run"] is True
        assert result["job_id"] is None
        assert "#!/bin/bash" in result["script"]

    def test_dry_run_sbatch_directives(self):
        result = submit(
            _SLURM_PROFILE,
            "python train.py",
            gpus=2,
            job_name="my-test",
            dry_run=True,
        )
        script = result["script"]
        assert "#SBATCH --job-name=my-test" in script
        assert "#SBATCH --partition=gpu" in script
        assert "#SBATCH --gpus=2" in script
        assert "#SBATCH --time=02:00:00" in script

    def test_dry_run_contains_command(self):
        result = submit(_SLURM_PROFILE, "echo hello world", dry_run=True)
        assert "echo hello world" in result["script"]

    def test_submit_calls_sbatch(self, tmp_path):
        profile = dict(_SLURM_PROFILE)
        profile["defaults"] = dict(_SLURM_PROFILE["defaults"])
        profile["defaults"]["log_dir"] = str(tmp_path)
        mock_result = MagicMock()
        mock_result.stdout = "12345\n"
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = submit(profile, "python train.py", dry_run=False)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "sbatch"
        assert "--parsable" in args
        assert result["job_id"] == "12345"

    def test_submit_elapsed_time_fast(self, tmp_path):
        """submit() (mocked) completes in under 5 seconds (SC-005)."""
        profile = dict(_SLURM_PROFILE)
        profile["defaults"] = dict(_SLURM_PROFILE["defaults"])
        profile["defaults"]["log_dir"] = str(tmp_path)
        mock_result = MagicMock()
        mock_result.stdout = "99\n"
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            t0 = time.time()
            submit(profile, "python train.py", dry_run=False)
            elapsed = time.time() - t0
        assert elapsed < 5.0


class TestSlurmStatus:
    def test_running_state(self):
        squeue_output = '{"jobs": [{"job_state": "RUNNING"}]}'
        mock = MagicMock(returncode=0, stdout=squeue_output)
        with patch("subprocess.run", return_value=mock):
            result = status("12345")
        assert result["status"] == "RUNNING"
        assert result["failure_reason"] is None

    def test_preempted_becomes_failed(self):
        squeue_output = '{"jobs": [{"job_state": "PREEMPTED"}]}'
        mock = MagicMock(returncode=0, stdout=squeue_output)
        with patch("subprocess.run", return_value=mock):
            result = status("12345")
        assert result["status"] == "FAILED"
        assert result["failure_reason"] == "preempted"

    def test_non_zero_returncode_means_completed(self):
        mock = MagicMock(returncode=1, stdout="")
        with patch("subprocess.run", return_value=mock):
            result = status("12345")
        assert result["status"] == "COMPLETED"


class TestSlurmResults:
    def test_reads_log_files(self, tmp_path):
        (tmp_path / "nobrainer-job_42.out").write_text("STDOUT content")
        (tmp_path / "nobrainer-job_42.err").write_text("STDERR content")
        result = results("42", log_dir=tmp_path)
        assert result["stdout"] == "STDOUT content"
        assert result["stderr"] == "STDERR content"

    def test_missing_logs_return_empty_strings(self, tmp_path):
        result = results("99", log_dir=tmp_path)
        assert result["stdout"] == ""
        assert result["stderr"] == ""


class TestSlurmCancel:
    def test_cancel_calls_scancel(self):
        mock = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock) as mock_run:
            result = cancel("12345")
        mock_run.assert_called_once()
        assert "scancel" in mock_run.call_args[0][0]
        assert result["cancelled"] is True

    def test_cancel_failure(self):
        mock = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=mock):
            result = cancel("bad-id")
        assert result["cancelled"] is False

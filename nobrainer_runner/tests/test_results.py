"""Unit tests for the results parser."""

from __future__ import annotations

import json

import pytest

from nobrainer_runner.results import parse_results


def _write_jsonl(path, records):
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


class TestParseResults:
    def test_counts_passes_and_failures(self, tmp_path):
        log = tmp_path / "report.jsonl"
        _write_jsonl(
            log,
            [
                {"when": "call", "outcome": "passed"},
                {"when": "call", "outcome": "passed"},
                {"when": "call", "outcome": "failed"},
            ],
        )
        result = parse_results(log)
        assert result["tests_passed"] == 2
        assert result["tests_failed"] == 1
        assert result["tests_error"] == 0

    def test_counts_errors(self, tmp_path):
        log = tmp_path / "report.jsonl"
        _write_jsonl(
            log,
            [
                {"when": "setup", "outcome": "error"},
                {"when": "call", "outcome": "passed"},
            ],
        )
        # Only "call" phase errors count
        result = parse_results(log)
        assert result["tests_passed"] == 1
        assert result["tests_error"] == 0

    def test_summary_string(self, tmp_path):
        log = tmp_path / "report.jsonl"
        _write_jsonl(
            log,
            [
                {"when": "call", "outcome": "passed"},
                {"when": "call", "outcome": "failed"},
            ],
        )
        result = parse_results(log)
        assert "1 passed" in result["summary"]
        assert "1 failed" in result["summary"]

    def test_all_passed_zero_failure(self, tmp_path):
        log = tmp_path / "report.jsonl"
        _write_jsonl(log, [{"when": "call", "outcome": "passed"} for _ in range(5)])
        result = parse_results(log)
        assert result["tests_failed"] == 0
        assert result["tests_error"] == 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_results(tmp_path / "nonexistent.jsonl")

    def test_empty_file(self, tmp_path):
        log = tmp_path / "report.jsonl"
        log.write_text("")
        result = parse_results(log)
        assert result["tests_passed"] == 0
        assert result["tests_failed"] == 0

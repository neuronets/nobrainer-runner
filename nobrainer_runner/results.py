"""Results parser for nobrainer-runner.

Parses pytest JSON output (``--report-log`` format) into a structured
dict matching the contract in ``contracts/nobrainer-runner-cli.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_results(report_log: str | Path) -> dict[str, Any]:
    """Parse a pytest ``--report-log`` JSONL file.

    Each line in the file is a JSON object with a ``when`` field.  We
    collect ``call`` phase outcomes to count passes, failures, and errors.

    Parameters
    ----------
    report_log : path
        Path to the pytest ``--report-log`` JSONL file.

    Returns
    -------
    dict with keys:
        - ``tests_passed`` (int)
        - ``tests_failed`` (int)
        - ``tests_error`` (int)
        - ``coverage_pct`` (float | None) — extracted from terminal output
          line ``TOTAL … XX%`` if present
        - ``summary`` (str) — human-readable one-liner

    Raises
    ------
    FileNotFoundError
        If ``report_log`` does not exist.
    """
    path = Path(report_log)
    if not path.exists():
        raise FileNotFoundError(f"Report log not found: {path}")

    passed = 0
    failed = 0
    error = 0
    coverage_pct: float | None = None

    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            when = record.get("when")
            outcome = record.get("outcome", "")

            if when == "call":
                if outcome == "passed":
                    passed += 1
                elif outcome == "failed":
                    failed += 1
                elif outcome == "error":
                    error += 1

            # Try to parse coverage from terminal reporter lines
            if record.get("$report_type") == "CoverageReport":
                coverage_pct = record.get("coverage_pct")
            elif "longrepr" in record:
                longrepr = str(record.get("longrepr", ""))
                for token in longrepr.split():
                    if token.endswith("%") and token[:-1].replace(".", "", 1).isdigit():
                        try:
                            coverage_pct = float(token[:-1])
                        except ValueError:
                            pass

    summary = f"{passed} passed, {failed} failed, {error} error(s)"
    if coverage_pct is not None:
        summary += f", {coverage_pct:.1f}% coverage"

    return {
        "tests_passed": passed,
        "tests_failed": failed,
        "tests_error": error,
        "coverage_pct": coverage_pct,
        "summary": summary,
    }

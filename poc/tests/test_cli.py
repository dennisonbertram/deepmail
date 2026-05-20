"""CLI surface tests.

Lightweight smoke checks that the click command tree exposes the flags we
document. These tests use Click's CliRunner so they don't actually drive
the loop, hit Gmail, or load a model — just exercise --help output.
"""

from __future__ import annotations

import sys
from pathlib import Path

from click.testing import CliRunner

POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

from pi_email.cli import main  # noqa: E402


def test_run_help_lists_max_per_query() -> None:
    """`pi-email run --help` must advertise the --max-per-query flag.

    This is the per-Gmail-query truncation knob; documenting it on the help
    surface is the contract callers rely on to size production runs.
    """
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0, result.output
    assert "--max-per-query" in result.output

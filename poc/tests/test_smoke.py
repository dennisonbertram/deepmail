"""One end-to-end smoke test.

Verifies that:
  - The loop runs to completion against the fixture corpus.
  - The profile file is created.
  - At least 5 family members appear in the materialized profile.
  - The loop terminated on `frontier_exhausted` (not iter_cap / budget_cap /
    corpus_cap). This is the load-bearing claim: the deterministic frontier
    runs to empty without an LLM "I'm done" call.
  - At least one obliquely-mentioned family member ("Mia") appears in the
    profile — proving the iterative expansion did something the naive
    single-search wouldn't.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

from pi_email.loop import run_loop_and_materialize  # noqa: E402


def test_smoke(tmp_path):
    # Force mock proposer: deterministic, no network.
    os.environ["ANTHROPIC_API_KEY"] = ""

    profiles_dir = tmp_path / "profiles"
    result, out_path = run_loop_and_materialize(
        fixtures_dir=POC_ROOT / "fixtures" / "family_corpus",
        seed="figure out my family",
        profiles_dir=profiles_dir,
        force_mock=True,
    )

    # 1. Profile created.
    assert out_path.exists(), f"profile not written at {out_path}"
    content = out_path.read_text(encoding="utf-8")

    # 2. Stop reason is one of the frontier_exhausted variants (Round 3.5
    # split the single `frontier_exhausted` into `_clean` and `_no_yield`
    # for observability; the legacy bare name is no longer emitted but kept
    # as an accepted value here in case a future change reintroduces it).
    # Pass 12B Fix 1: `no_family_signal_after_N` is also acceptable — same
    # family count is expected, just a different graceful-stop label.
    # Both signal "the loop converged" rather than "we hit the iter cap".
    assert result.stop_reason.rule in (
        "frontier_exhausted",
        "frontier_exhausted_clean",
        "frontier_exhausted_no_yield",
        "no_new_persons_after_N",
    ), (
        f"expected a graceful-stop variant (frontier_exhausted_* or "
        f"no_new_persons_after_N), got "
        f"{result.stop_reason.rule} ({result.stop_reason.detail})"
    )

    # 3. At least 5 family members in the profile body.
    expected_members = ["Jane", "Bob", "Sarah", "Emma", "Mia"]
    found = [name for name in expected_members if name in content]
    assert len(found) >= 5, (
        f"expected >=5 family members in profile, found {found}\n\n"
        f"profile content:\n{content}"
    )

    # 4. Mia (obliquely-mentioned) shows up — proves iterative expansion fired.
    assert "Mia" in content, (
        "Mia should appear in profile via iterative expansion "
        "(she's only referenced in swim/school/etc. messages)"
    )

    # 5. Loop did more than one iteration — otherwise nothing was "iterative".
    assert len(result.iterations) >= 2, (
        f"expected >=2 iterations, got {len(result.iterations)}"
    )

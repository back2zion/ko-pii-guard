"""Long invariant sweeps must not report results from a moving source tree."""

import sys

import pytest

from benchmarks import fuzz_structured_invariants


def test_source_change_discards_invariant_report(monkeypatch, tmp_path):
    hashes = iter(("before", "after"))
    output = tmp_path / "report.json"
    monkeypatch.setattr(fuzz_structured_invariants, "package_hash", lambda: next(hashes))
    monkeypatch.setattr(sys, "argv", ["fuzz", "--iterations", "1",
                                     "--progress-every", "1", "--output", str(output)])
    with pytest.raises(RuntimeError, match="Source or invariant scorer changed"):
        fuzz_structured_invariants.main()
    assert not output.exists()

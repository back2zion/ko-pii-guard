"""Text invariant sweeps must discard reports if code moves during execution."""

import sys

import pytest

from benchmarks import fuzz_masking_invariants


def test_source_change_discards_text_invariant_report(monkeypatch, tmp_path):
    hashes = iter(("before", "after"))
    output = tmp_path / "report.json"
    monkeypatch.setattr(fuzz_masking_invariants, "package_hash", lambda: next(hashes))
    monkeypatch.setattr(sys, "argv", ["fuzz", "--iterations", "1",
                                     "--progress-every", "1", "--output", str(output)])
    with pytest.raises(RuntimeError, match="Source or invariant scorer changed"):
        fuzz_masking_invariants.main()
    assert not output.exists()

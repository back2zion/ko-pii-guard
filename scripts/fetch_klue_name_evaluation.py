"""Explicitly download the pinned official KLUE NER development data and license.

Kept outside the repository by default; no package import triggers a download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "3efd98708a40ff49251fddde35453f8fbb11f536"
FILES = {
    "klue_benchmark/klue-ner-v1.1/klue-ner-v1.1_dev.tsv": (
        "0f4d5e818f7b82d299c3a87856fc40a706f5943207580ddb252397e387050a54"
    ),
    "License.md": "7abe19ec9bb73b36141b999b861d24ad855e808bafe0f81e84cce28556f6c297",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/ko-pii-klue-ner"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        repository="https://github.com/KLUE-benchmark/KLUE",
        revision=REVISION,
        files={},
        license="CC-BY-SA-4.0",
        purpose="external person-boundary stress test, not private-person policy accuracy",
        redistributed_in_repository=False,
    )
    for relative, expected in FILES.items():
        url = f"https://raw.githubusercontent.com/KLUE-benchmark/KLUE/{REVISION}/{relative}"
        destination = args.output / Path(relative).name
        if destination.exists():
            content = destination.read_bytes()
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "ko-pii-guard-research"})
            with urllib.request.urlopen(request, timeout=45) as response:
                content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Pinned source mismatch; refusing to overwrite: {destination.name}")
        if not destination.exists():
            destination.write_bytes(content)
        manifest["files"][destination.name] = dict(url=url, sha256=expected, bytes=len(content))
    metadata = json.dumps(manifest, indent=2) + "\n"
    manifest_path = args.output / "source.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing provenance differs; use a new output directory")
    manifest_path.write_text(metadata)
    print(f"Verified pinned KLUE files and CC-BY-SA-4.0 license in {args.output}")


if __name__ == "__main__":
    main()

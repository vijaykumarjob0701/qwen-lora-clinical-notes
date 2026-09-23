#!/usr/bin/env python3
"""Teaching example: pull synthetic notes from a drop folder, dedupe, version.

This is the *pattern* for an EHR export / file-drop ingest job.
Do not point it at real patient PHI. Real ingest belongs behind a BAA,
access control, and a de-identification pipeline you do not commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def ingest(drop_dir: Path, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = raw_dir / f"notes_{stamp}.jsonl"
    n_kept = n_dup = 0
    with out.open("w", encoding="utf-8") as writer:
        for src in sorted(drop_dir.glob("*.jsonl")):
            for line in src.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                key = hashlib.sha256(line.encode("utf-8")).hexdigest()
                if key in seen:
                    n_dup += 1
                    continue
                seen.add(key)
                writer.write(line.rstrip() + "\n")
                n_kept += 1
    manifest = {
        "created_utc": stamp,
        "source_dir": str(drop_dir),
        "rows_kept": n_kept,
        "rows_deduped": n_dup,
        "sha256": file_fingerprint(out),
    }
    (out.with_suffix(".manifest.json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drop-dir", type=Path, default=Path("data"))
    p.add_argument("--raw-dir", type=Path, default=Path("mlops/var/raw"))
    args = p.parse_args()
    # Copy the bundled synthetic file into a pretend drop folder if needed.
    args.drop_dir.mkdir(parents=True, exist_ok=True)
    out = ingest(args.drop_dir, args.raw_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

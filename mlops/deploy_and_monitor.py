#!/usr/bin/env python3
"""Conditional deploy + post-deploy rollback for the local teaching registry.

On promote, the winning adapter is copied into mlops/var/registry/current
and registry.json is rewritten with metrics, a UTC timestamp, and a git sha.
On gate failure the current registry is left untouched.

These helpers do not call any cloud API. vLLM / kubectl lines are still
printed as the production analog.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def registry_root(registry_uri: str) -> Path:
    uri = registry_uri
    if uri.startswith("file://"):
        uri = uri[len("file://") :]
    return Path(uri)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def next_version(root: Path) -> str:
    versions = root / "versions"
    n = 1
    if versions.is_dir():
        nums: list[int] = []
        for child in versions.iterdir():
            name = child.name
            if len(name) >= 4 and name[0] == "v" and name[1:4].isdigit():
                nums.append(int(name[1:4]))
        if nums:
            n = max(nums) + 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"v{n:03d}-{stamp}"


def deploy_if_promoted(
    report_path: Path,
    adapter_dir: Path,
    registry_uri: str,
    *,
    merged_dir: Path | None = None,
    predictions_path: Path | None = None,
) -> bool:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("promote"):
        print("GATE FAILED — keep the current production model and page the on-call.")
        print(json.dumps(report.get("candidate", {}), indent=2))
        return False

    root = registry_root(registry_uri)
    root.mkdir(parents=True, exist_ok=True)
    current = root / "current"
    previous = root / "previous"

    if current.exists():
        if previous.exists():
            shutil.rmtree(previous)
        shutil.move(str(current), str(previous))

    version = next_version(root)
    version_dir = root / "versions" / version
    if version_dir.exists():
        shutil.rmtree(version_dir)
    shutil.copytree(adapter_dir, version_dir)
    if merged_dir is not None and merged_dir.exists():
        dest_merged = version_dir / "merged"
        if dest_merged.exists():
            shutil.rmtree(dest_merged)
        shutil.copytree(merged_dir, dest_merged)
    if predictions_path is not None and predictions_path.exists():
        shutil.copy2(predictions_path, version_dir / "eval_predictions.jsonl")

    shutil.copytree(version_dir, current)

    meta = {
        "version": version,
        "promoted_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "adapter_path": str(current.resolve()),
        "merged_path": str((current / "merged").resolve()) if (current / "merged").exists() else None,
        "source_adapter": str(adapter_dir),
        "metrics": report,
    }
    payload = json.dumps(meta, indent=2)
    (root / "registry.json").write_text(payload + "\n", encoding="utf-8")
    (current / "registry.json").write_text(payload + "\n", encoding="utf-8")

    print("GATE PASSED — copy adapter into the local teaching registry.")
    print(f"  merge:   python train_lora.py --eval-only --adapter-path {adapter_dir} --merge")
    print(f"  push:    copy {adapter_dir} -> {root}")
    print(f"  current: {current}")
    print("  serve:   vllm serve <merged-model> --max-model-len 4096")
    return True


def monitor_and_maybe_rollback(
    live_scores: Path,
    floor: float = 0.55,
    registry_uri: str | None = None,
) -> None:
    """If live entity-F1 drops below a floor, restore registry/previous locally."""
    scores = json.loads(live_scores.read_text(encoding="utf-8"))
    entity_f1 = float(scores.get("entity_f1", 0.0))
    if entity_f1 >= floor:
        print(f"healthy: live entity_f1={entity_f1:.3f}")
        return

    print(f"DRIFT: live entity_f1={entity_f1:.3f} < {floor:.3f}. Rolling back.")
    print("  kubectl rollout undo deploy/clinical-qwen")
    if not registry_uri:
        return
    root = registry_root(registry_uri)
    current, previous = root / "current", root / "previous"
    if not previous.exists():
        print("  local registry has no previous version; left current untouched.")
        return
    rolled = root / "rolled_back_from_current"
    if rolled.exists():
        shutil.rmtree(rolled)
    if current.exists():
        shutil.move(str(current), str(rolled))
    shutil.move(str(previous), str(current))
    print(f"  restored {current} from previous")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, default=Path("mlops/var/eval_report.json"))
    p.add_argument("--adapter", type=Path, default=Path("outputs/qwen-clinical-lora"))
    p.add_argument("--registry", default="file://mlops/var/registry")
    p.add_argument("--merged", type=Path, default=None)
    p.add_argument("--predictions", type=Path, default=None)
    p.add_argument("--live-scores", type=Path, default=None)
    args = p.parse_args()
    ok = True
    if args.report.exists():
        ok = deploy_if_promoted(
            args.report,
            args.adapter,
            args.registry,
            merged_dir=args.merged,
            predictions_path=args.predictions,
        )
    if args.live_scores:
        monitor_and_maybe_rollback(args.live_scores, registry_uri=args.registry)
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()

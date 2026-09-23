#!/usr/bin/env python3
"""Teaching stubs for conditional deploy + post-deploy rollback.

These functions print the commands you would run. They do not call any
cloud API and they do not start a real server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def deploy_if_promoted(report_path: Path, adapter_dir: Path, registry_uri: str) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("promote"):
        print("GATE FAILED — keep the current production model and page the on-call.")
        print(json.dumps(report["candidate"], indent=2))
        return
    print("GATE PASSED — merge adapter, push registry, reload the serving replica.")
    print(f"  merge:   python train_lora.py --eval-only --adapter-path {adapter_dir} --merge")
    print(f"  push:    copy {adapter_dir} -> {registry_uri}")
    print("  serve:   vllm serve <merged-model> --max-model-len 4096")


def monitor_and_maybe_rollback(live_scores: Path, floor: float = 0.55) -> None:
    """If live entity-F1 drops below a floor, roll back to the previous registry tag."""
    scores = json.loads(live_scores.read_text(encoding="utf-8"))
    entity_f1 = float(scores.get("entity_f1", 0.0))
    if entity_f1 < floor:
        print(f"DRIFT: live entity_f1={entity_f1:.3f} < {floor:.3f}. Rolling back.")
        print("  kubectl rollout undo deploy/clinical-qwen")
        return
    print(f"healthy: live entity_f1={entity_f1:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, default=Path("mlops/var/eval_report.json"))
    p.add_argument("--adapter", type=Path, default=Path("outputs/qwen-clinical-lora"))
    p.add_argument("--registry", default="file://mlops/var/registry/clinical-qwen")
    p.add_argument("--live-scores", type=Path, default=None)
    args = p.parse_args()
    if args.report.exists():
        deploy_if_promoted(args.report, args.adapter, args.registry)
    if args.live_scores:
        monitor_and_maybe_rollback(args.live_scores)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Orchestrate ingest → preprocess → train → eval → gated local deploy.

Synthetic / PHI-free only. One command:

    python -m mlops.pipeline --smoke
    python scripts/run_pipeline.py --smoke

--smoke / --dry-run / PIPELINE_SMOKE=1 skip the 7B download and GPU train.
They still run ingest, chat-template preprocess, stub generation, the real
eval gate, and the local teaching registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from mlops.deploy_and_monitor import deploy_if_promoted
from mlops.generate_eval import generate_with_adapter, write_eval_predictions
from mlops.ingest import ingest
from mlops.preprocess import preprocess

REPO_ROOT = Path(__file__).resolve().parents[1]


def drop_fingerprint(drop_dir: Path, pattern: str = "*.jsonl") -> str:
    digest = hashlib.sha256()
    for path in sorted(drop_dir.glob(pattern)):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def should_retrain(drop_dir: Path, state_path: Path, pattern: str = "*.jsonl") -> bool:
    if not state_path.exists():
        return True
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    return state.get("drop_fingerprint") != drop_fingerprint(drop_dir, pattern)


def write_state(state_path: Path, drop_dir: Path, pattern: str, extra: dict | None = None) -> None:
    payload = {
        "drop_fingerprint": drop_fingerprint(drop_dir, pattern),
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra:
        payload.update(extra)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _is_smoke_adapter(adapter_dir: Path) -> bool:
    cfg = adapter_dir / "adapter_config.json"
    if not cfg.is_file():
        return False
    try:
        return bool(json.loads(cfg.read_text(encoding="utf-8")).get("smoke"))
    except json.JSONDecodeError:
        return False


def smoke_train(adapter_dir: Path, processed: Path, stamp: str) -> Path:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "smoke": True,
        "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        "note": "PIPELINE_SMOKE stub adapter. Not real LoRA weights.",
        "processed": str(processed),
        "created_utc": stamp,
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (adapter_dir / "adapter_marker.txt").write_text(f"smoke adapter {stamp}\n", encoding="utf-8")
    return adapter_dir


def real_train(processed: Path, adapter_dir: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "train_lora.py"),
        "--data",
        str(processed),
        "--output-dir",
        str(adapter_dir),
        "--model-id",
        args.model_id,
    ]
    if args.merge:
        cmd.append("--merge")
    if args.load_in_4bit:
        cmd.append("--load-in-4bit")
    if args.load_in_8bit:
        cmd.append("--load-in-8bit")
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


def _write_baseline(
    processed: Path,
    baseline_path: Path,
    registry_dir: Path,
    args: argparse.Namespace,
    smoke: bool,
) -> None:
    saved = registry_dir / "current" / "eval_predictions.jsonl"
    if saved.is_file():
        shutil.copy2(saved, baseline_path)
        return
    current = registry_dir / "current"
    if smoke or _is_smoke_adapter(current) or not current.exists():
        write_eval_predictions(processed, baseline_path, quality=args.smoke_baseline, split="eval")
        return
    generate_with_adapter(processed, baseline_path, current, args.model_id)


def run_eval_gate(candidate: Path, baseline: Path, report: Path) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "mlops" / "eval_gate.py"),
            "--candidate",
            str(candidate),
            "--baseline",
            str(baseline),
            "--report",
            str(report),
        ],
        cwd=str(REPO_ROOT),
        check=False,
    )


def run_pipeline(args: argparse.Namespace) -> int:
    smoke = bool(args.smoke or args.dry_run)
    drop_dir: Path = args.drop_dir
    var_dir: Path = args.var_dir
    pattern: str = args.drop_pattern
    state_path = var_dir / "pipeline_state.json"
    registry_dir = var_dir / "registry"

    if not args.force and not should_retrain(drop_dir, state_path, pattern):
        print("No new drop files since last run; skip. Use --force to retrain anyway.", flush=True)
        return 0
    if not any(drop_dir.glob(pattern)):
        raise SystemExit(f"No files matching {pattern!r} in {drop_dir}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw_dir = var_dir / "raw"
    processed = var_dir / "processed" / "clinical_sft.jsonl"
    run_dir = var_dir / "runs" / stamp
    adapter_dir = run_dir / "adapter"
    eval_dir = run_dir / "eval"
    candidate_path = eval_dir / "candidate.jsonl"
    baseline_path = eval_dir / "baseline.jsonl"
    report_path = var_dir / "eval_report.json"

    print(f"==> ingest {drop_dir} ({pattern})", flush=True)
    raw = ingest(drop_dir, raw_dir, pattern=pattern, sft_only=True)
    print(f"==> preprocess {raw} -> {processed}", flush=True)
    preprocess(raw, processed)

    if smoke:
        print(f"==> train (smoke stub) {adapter_dir}", flush=True)
        smoke_train(adapter_dir, processed, stamp)
        print(f"==> generate eval JSONL (smoke candidate={args.smoke_candidate})", flush=True)
        write_eval_predictions(processed, candidate_path, quality=args.smoke_candidate, split="eval")
        _write_baseline(processed, baseline_path, registry_dir, args, smoke=True)
    else:
        print(f"==> train (LoRA) {adapter_dir}", flush=True)
        real_train(processed, adapter_dir, args)
        print("==> generate eval JSONL (adapter vs baseline)", flush=True)
        generate_with_adapter(processed, candidate_path, adapter_dir, args.model_id)
        _write_baseline(processed, baseline_path, registry_dir, args, smoke=False)

    print(f"==> eval gate {candidate_path} vs {baseline_path}", flush=True)
    run_eval_gate(candidate_path, baseline_path, report_path)
    if not report_path.is_file():
        raise SystemExit("eval gate did not write a report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    merged = adapter_dir / "merged"
    ok = deploy_if_promoted(
        report_path,
        adapter_dir,
        str(registry_dir),
        merged_dir=merged if merged.exists() else None,
        predictions_path=candidate_path if report.get("promote") else None,
    )
    write_state(
        state_path,
        drop_dir,
        pattern,
        {"promote": bool(report.get("promote")), "run": stamp},
    )
    return 0 if ok else 2


def _env_smoke() -> bool:
    return os.environ.get("PIPELINE_SMOKE", "").strip().lower() in {"1", "true", "yes"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drop-dir", type=Path, default=Path("data"))
    p.add_argument("--drop-pattern", default="*clinical*.jsonl")
    p.add_argument("--var-dir", type=Path, default=Path("mlops/var"))
    p.add_argument("--smoke", action="store_true", help="Stub train/generate (no GPU, no 7B download)")
    p.add_argument("--dry-run", action="store_true", help="Same as --smoke: exercise the flow with stubs")
    p.add_argument("--force", action="store_true", help="Retrain even if the drop folder is unchanged")
    p.add_argument("--watch", action="store_true", help="Poll the drop folder and re-run when files change")
    p.add_argument("--interval", type=float, default=30.0, help="Seconds between --watch polls")
    p.add_argument("--smoke-candidate", choices=("echo", "weak"), default="echo")
    p.add_argument("--smoke-baseline", choices=("echo", "weak"), default="weak")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--merge", action="store_true", help="Also merge LoRA after a real train")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if _env_smoke():
        args.smoke = True
    if args.watch:
        while True:
            run_pipeline(args)
            time.sleep(args.interval)
    raise SystemExit(run_pipeline(args))


if __name__ == "__main__":
    main()

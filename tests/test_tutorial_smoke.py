"""CPU-only smoke tests: data shape, preview, preprocess, eval gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_clinical_jsonl_is_synthetic_and_well_formed():
    path = ROOT / "data" / "example_clinical_sft.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 5
    assert any(r.get("split") == "eval" for r in rows)
    for row in rows:
        assert row.get("synthetic") is True
        roles = [m["role"] for m in row["messages"]]
        assert roles[0] == "system"
        assert "user" in roles and "assistant" in roles


def test_preview_runs_without_transformers():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "train_lora.py"), "--preview"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "<|im_start|>user" in proc.stdout
    assert "synth-001" in proc.stdout


def test_preprocess_and_eval_gate(tmp_path):
    dest = tmp_path / "processed.jsonl"
    subprocess.run(
        [sys.executable, str(ROOT / "mlops/preprocess.py"), "--src", str(ROOT / "data/example_clinical_sft.jsonl"), "--dest", str(dest)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    first = json.loads(dest.read_text(encoding="utf-8").splitlines()[0])
    assert first["text"].startswith("<|im_start|>system")

    report = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "mlops/eval_gate.py"),
            "--candidate",
            str(ROOT / "data/example_eval_candidate.jsonl"),
            "--baseline",
            str(ROOT / "data/example_eval_baseline.jsonl"),
            "--report",
            str(report),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["promote"] is True

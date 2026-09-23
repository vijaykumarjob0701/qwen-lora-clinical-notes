"""CPU-only tests for the end-to-end MLOps pipeline (no GPU, no 7B download)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLINICAL = ROOT / "data" / "example_clinical_sft.jsonl"


def _copy_clinical(dest: Path, name: str = "notes_clinical.jsonl") -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / name
    shutil.copy(CLINICAL, target)
    return target


def _registry_json(registry_dir: Path) -> dict:
    return json.loads((registry_dir / "registry.json").read_text(encoding="utf-8"))


def test_generate_eval_writes_id_prediction_reference(tmp_path):
    from mlops.generate_eval import write_eval_predictions

    dest = tmp_path / "candidate.jsonl"
    n = write_eval_predictions(CLINICAL, dest, quality="echo", split="eval")
    assert n >= 2
    rows = [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert {r["id"] for r in rows} >= {"synth-006", "synth-007"}
    for row in rows:
        assert set(row) >= {"id", "prediction", "reference"}
        assert row["prediction"]
        assert row["reference"]
        assert row["prediction"] == row["reference"]


def test_generate_eval_refuses_nonsynthetic(tmp_path):
    from mlops.generate_eval import write_eval_predictions

    src = tmp_path / "phi.jsonl"
    src.write_text(
        json.dumps(
            {
                "id": "real-001",
                "synthetic": False,
                "split": "eval",
                "messages": [
                    {"role": "system", "content": "x"},
                    {"role": "user", "content": "y"},
                    {"role": "assistant", "content": "z"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="synthetic"):
        write_eval_predictions(src, tmp_path / "out.jsonl", quality="echo")


def test_deploy_promotes_into_local_registry(tmp_path):
    from mlops.deploy_and_monitor import deploy_if_promoted
    from mlops.eval_gate import score_file

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"smoke": true}\n', encoding="utf-8")
    (adapter / "adapter_marker.txt").write_text("stub\n", encoding="utf-8")

    cand = ROOT / "data" / "example_eval_candidate.jsonl"
    base = ROOT / "data" / "example_eval_baseline.jsonl"
    report = tmp_path / "eval_report.json"
    report.write_text(
        json.dumps(
            {
                "candidate": score_file(cand),
                "baseline": score_file(base),
                "promote": True,
                "keys": ["rouge_l", "bleu", "entity_f1"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    registry = tmp_path / "registry"
    ok = deploy_if_promoted(report, adapter, str(registry))
    assert ok is True
    current = registry / "current"
    assert (current / "adapter_config.json").is_file()
    meta = _registry_json(registry)
    assert meta["adapter_path"].endswith("current") or Path(meta["adapter_path"]).name == "current"
    assert "metrics" in meta
    assert meta["metrics"]["promote"] is True
    assert "promoted_utc" in meta
    assert "git_sha" in meta or "version" in meta


def test_deploy_keeps_registry_when_gate_fails(tmp_path):
    from mlops.deploy_and_monitor import deploy_if_promoted

    registry = tmp_path / "registry"
    current = registry / "current"
    current.mkdir(parents=True)
    (current / "winner.txt").write_text("keep-me\n", encoding="utf-8")
    (registry / "registry.json").write_text(
        json.dumps({"version": "already-here", "metrics": {"promote": True}}),
        encoding="utf-8",
    )
    before = (registry / "registry.json").read_text(encoding="utf-8")

    adapter = tmp_path / "loser"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")

    report = tmp_path / "eval_report.json"
    report.write_text(
        json.dumps(
            {
                "candidate": {"rouge_l": 0.1, "bleu": 0.1, "entity_f1": 0.1},
                "baseline": {"rouge_l": 0.9, "bleu": 0.9, "entity_f1": 0.9},
                "promote": False,
                "keys": ["rouge_l", "bleu", "entity_f1"],
            }
        ),
        encoding="utf-8",
    )

    ok = deploy_if_promoted(report, adapter, str(registry))
    assert ok is False
    assert (current / "winner.txt").read_text(encoding="utf-8") == "keep-me\n"
    assert not (current / "adapter_config.json").exists()
    assert (registry / "registry.json").read_text(encoding="utf-8") == before


def test_should_retrain_detects_new_drop_files(tmp_path):
    from mlops.pipeline import should_retrain

    drop = tmp_path / "drop"
    state = tmp_path / "pipeline_state.json"
    _copy_clinical(drop)
    assert should_retrain(drop, state) is True
    # First call must not persist; the orchestrator writes state after a run.
    assert should_retrain(drop, state) is True

    state.write_text(
        json.dumps({"drop_fingerprint": _fingerprint_via_module(drop)}),
        encoding="utf-8",
    )
    assert should_retrain(drop, state) is False

    extra = {
        "id": "synth-new",
        "synthetic": True,
        "split": "train",
        "messages": [
            {"role": "system", "content": "You are a careful clinical documentation assistant."},
            {"role": "user", "content": "Summarize this SYNTHETIC note. Name: Demo."},
            {"role": "assistant", "content": "Demo is a fictional teaching case."},
        ],
    }
    (drop / "batch2_clinical.jsonl").write_text(json.dumps(extra) + "\n", encoding="utf-8")
    assert should_retrain(drop, state) is True


def _fingerprint_via_module(drop: Path) -> str:
    from mlops.pipeline import drop_fingerprint

    return drop_fingerprint(drop)


def _pipeline_cmd(tmp_path: Path, extra: list[str] | None = None) -> list[str]:
    drop = tmp_path / "drop"
    if not drop.exists():
        _copy_clinical(drop)
    var = tmp_path / "var"
    cmd = [
        sys.executable,
        "-m",
        "mlops.pipeline",
        "--smoke",
        "--drop-dir",
        str(drop),
        "--var-dir",
        str(var),
        "--drop-pattern",
        "*.jsonl",
        "--force",
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def test_smoke_pipeline_promotes_and_updates_registry(tmp_path):
    proc = subprocess.run(
        _pipeline_cmd(tmp_path),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    registry = tmp_path / "var" / "registry"
    assert (registry / "current" / "adapter_config.json").is_file()
    meta = _registry_json(registry)
    assert meta["metrics"]["promote"] is True
    report = json.loads((tmp_path / "var" / "eval_report.json").read_text(encoding="utf-8"))
    assert report["promote"] is True
    cand = tmp_path / "var"
    cand_files = list(cand.rglob("candidate.jsonl"))
    assert cand_files, "pipeline must write eval JSONL for the gate"
    first = json.loads(cand_files[0].read_text(encoding="utf-8").splitlines()[0])
    assert set(first) >= {"id", "prediction", "reference"}


def test_smoke_pipeline_does_not_promote_weak_candidate(tmp_path):
    # Seed a winning registry first.
    proc = subprocess.run(_pipeline_cmd(tmp_path), cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    registry = tmp_path / "var" / "registry"
    before = (registry / "registry.json").read_text(encoding="utf-8")
    winner = (registry / "current" / "adapter_config.json").read_text(encoding="utf-8")

    proc = subprocess.run(
        _pipeline_cmd(tmp_path, ["--smoke-candidate", "weak"]),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert (registry / "registry.json").read_text(encoding="utf-8") == before
    assert (registry / "current" / "adapter_config.json").read_text(encoding="utf-8") == winner


def test_smoke_pipeline_skips_when_drop_unchanged(tmp_path):
    first = subprocess.run(_pipeline_cmd(tmp_path), cwd=ROOT, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    runs_after_first = list((tmp_path / "var" / "runs").glob("*"))

    skip_cmd = [c for c in _pipeline_cmd(tmp_path) if c != "--force"]
    second = subprocess.run(skip_cmd, cwd=ROOT, capture_output=True, text=True)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "no new" in (second.stdout + second.stderr).lower()
    assert list((tmp_path / "var" / "runs").glob("*")) == runs_after_first


def test_smoke_pipeline_retrains_when_new_jsonl_dropped(tmp_path):
    first = subprocess.run(_pipeline_cmd(tmp_path), cwd=ROOT, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    first_runs = sorted(p.name for p in (tmp_path / "var" / "runs").iterdir())

    extra = json.loads(CLINICAL.read_text(encoding="utf-8").splitlines()[0])
    extra["id"] = "synth-retrain"
    (tmp_path / "drop" / "round2_clinical.jsonl").write_text(json.dumps(extra) + "\n", encoding="utf-8")

    skip_cmd = [c for c in _pipeline_cmd(tmp_path) if c != "--force"]
    second = subprocess.run(skip_cmd, cwd=ROOT, capture_output=True, text=True)
    # Echo vs the already-promoted echo can tie the gate (exit 2). Retrain still happened.
    assert second.returncode in {0, 2}, second.stdout + second.stderr
    second_runs = sorted(p.name for p in (tmp_path / "var" / "runs").iterdir())
    assert len(second_runs) > len(first_runs)
    state = json.loads((tmp_path / "var" / "pipeline_state.json").read_text(encoding="utf-8"))
    assert state.get("drop_fingerprint")


def test_pipeline_refuses_nonsynthetic_drop(tmp_path):
    drop = tmp_path / "drop"
    drop.mkdir()
    (drop / "leaked.jsonl").write_text(
        json.dumps(
            {
                "id": "phi-001",
                "synthetic": False,
                "split": "train",
                "messages": [
                    {"role": "system", "content": "x"},
                    {"role": "user", "content": "real note"},
                    {"role": "assistant", "content": "y"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlops.pipeline",
            "--smoke",
            "--force",
            "--drop-dir",
            str(drop),
            "--var-dir",
            str(tmp_path / "var"),
            "--drop-pattern",
            "*.jsonl",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "synthetic" in (proc.stdout + proc.stderr).lower()
    assert not (tmp_path / "var" / "registry" / "registry.json").exists()


def test_run_pipeline_script_exists():
    assert (ROOT / "scripts" / "run_pipeline.py").is_file()

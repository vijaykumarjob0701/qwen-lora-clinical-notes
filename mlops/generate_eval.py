#!/usr/bin/env python3
"""Write {id, prediction, reference} JSONL for the evaluation gate.

Smoke / dry-run uses a deterministic stub (echo the gold answer, or a
weak truncation). The GPU path loads the same tokenizer + chat template
as train_lora.py and generates from a LoRA adapter or the bare base model.

Synthetic / PHI-free only — a row with messages and synthetic=false is refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def assistant_reference(row: dict[str, Any]) -> str:
    for msg in row.get("messages") or []:
        if msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return str(row.get("reference") or "")


def stub_predict(reference: str, quality: str = "echo") -> str:
    if quality == "echo":
        return reference
    if quality == "weak":
        words = reference.split()
        return " ".join(words[:6]) if words else ""
    raise ValueError(f"unknown stub quality {quality!r}; use echo or weak")


def _refuse_if_phi(row: dict[str, Any]) -> None:
    if "messages" in row and not row.get("synthetic", False):
        raise SystemExit("Refusing a row without synthetic=true. Do not generate from PHI here.")


def write_eval_predictions(
    src: Path,
    dest: Path,
    *,
    quality: str = "echo",
    split: str | None = "eval",
    require_synthetic: bool = True,
) -> int:
    """Stub generator used by --smoke / --dry-run. Returns the number of rows written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(encoding="utf-8") as inp, dest.open("w", encoding="utf-8") as out:
        for line in inp:
            if not line.strip():
                continue
            row = json.loads(line)
            if require_synthetic:
                _refuse_if_phi(row)
            if split and row.get("split") != split:
                continue
            if "id" not in row:
                continue
            ref = assistant_reference(row)
            record = {
                "id": row["id"],
                "prediction": stub_predict(ref, quality),
                "reference": ref,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def generate_with_adapter(
    src: Path,
    dest: Path,
    adapter_path: Path | None,
    model_id: str,
    max_samples: int = 8,
) -> int:
    """Real generation: base model, optionally with a PEFT adapter."""
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import train_lora  # noqa: WPS433 — reuse the tutorial loader / template

    rows = train_lora.load_jsonl(src)
    for row in rows:
        _refuse_if_phi(row)
    _, eval_rows = train_lora.split_rows(rows)

    args = argparse.Namespace(
        model_id=model_id,
        load_in_4bit=False,
        load_in_8bit=False,
    )
    model, tokenizer = train_lora.load_model_and_tokenizer(args)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))

    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    import torch

    model.eval()
    with dest.open("w", encoding="utf-8") as out:
        for row in eval_rows[:max_samples]:
            prompt_messages = [m for m in row["messages"] if m["role"] != "assistant"]
            prompt = train_lora.render_example(
                prompt_messages, tokenizer=tokenizer, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=160,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            pred = tokenizer.decode(
                gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            record = {
                "id": row.get("id"),
                "prediction": pred,
                "reference": assistant_reference(row),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--quality", choices=("echo", "weak"), default="echo")
    p.add_argument("--split", default="eval")
    p.add_argument("--adapter-path", type=Path, default=None)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--real", action="store_true", help="Load a model (needs GPU / download)")
    args = p.parse_args()
    if args.real:
        generate_with_adapter(args.src, args.dest, args.adapter_path, args.model_id)
    else:
        write_eval_predictions(args.src, args.dest, quality=args.quality, split=args.split)


if __name__ == "__main__":
    main()

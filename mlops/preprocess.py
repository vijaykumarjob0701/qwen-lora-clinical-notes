#!/usr/bin/env python3
"""Turn synthetic notes into versioned chat-template text.

Never commit real PHI. This script only understands the teaching JSONL
schema used in data/example_clinical_sft.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def apply_qwen_template(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    """Same string shape Qwen Instruct uses. Training must call the real tokenizer."""
    parts = [f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages]
    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


def normalize_abbreviations(text: str) -> str:
    # Tiny teaching map — real systems use a reviewed lexicon.
    replacements = {
        "hx": "history",
        "pt ": "patient ",
        "f/u": "follow-up",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def preprocess(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(encoding="utf-8") as inp, dest.open("w", encoding="utf-8") as out:
        for line in inp:
            row = json.loads(line)
            if not row.get("synthetic", False):
                raise SystemExit("Refusing a row without synthetic=true. Do not preprocess PHI here.")
            messages = []
            for msg in row["messages"]:
                messages.append({"role": msg["role"], "content": normalize_abbreviations(msg["content"])})
            record = {
                "id": row.get("id"),
                "synthetic": True,
                "split": row.get("split", "train"),
                "text": apply_qwen_template(messages, add_generation_prompt=False),
                "messages": messages,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} templated rows -> {dest}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=Path("data/example_clinical_sft.jsonl"))
    p.add_argument("--dest", type=Path, default=Path("mlops/var/processed/clinical_sft.jsonl"))
    args = p.parse_args()
    preprocess(args.src, args.dest)


if __name__ == "__main__":
    main()

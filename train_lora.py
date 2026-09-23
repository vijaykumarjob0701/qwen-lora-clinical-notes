#!/usr/bin/env python3
"""Fine-tune a Qwen Instruct model with LoRA on synthetic clinical notes.

This script follows the README tutorial step for step:

  1. Load the same base model + tokenizer you will use later
  2. Read instruction-response pairs (JSONL, synthetic data only)
  3. Apply the Qwen chat template
  4. Tokenize
  5–6. Configure LoRA and attach it with PEFT
  7. Train with TRL's SFTTrainer
  8. Save adapter weights
  9. Optionally merge the adapter into the base model
 10. Evaluate on a held-out split

Golden rule: the tokenizer and chat template used here must be the same
ones you use at eval time and at inference time.

Privacy: the bundled dataset is 100% synthetic. Do not point this script
at real patient notes or any PHI.

Usage (preview does not download a model):

    python train_lora.py --preview

Usage (needs a GPU for 7B-class models; 4-bit optional):

    python train_lora.py --model-id Qwen/Qwen2.5-7B-Instruct --load-in-4bit
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("train_lora")

DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "example_clinical_sft.jsonl"
DEFAULT_SYSTEM = (
    "You are a careful clinical documentation assistant. You only work with "
    "synthetic teaching notes. You do not give medical advice, diagnose "
    "patients, or invent facts that are missing from the note."
)

# Official Qwen2 / Qwen2.5 Instruct special tokens. Used only when the
# tokenizer is not loaded (the --preview path). Training always goes
# through tokenizer.apply_chat_template so it cannot drift.
QWEN_IM_START = "<|im_start|>"
QWEN_IM_END = "<|im_end|>"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face model id")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="JSONL with a messages array per row")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/qwen-clinical-lora"))
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--num-train-epochs", type=float, default=3.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--load-in-4bit", action="store_true", help="QLoRA-style 4-bit load via bitsandbytes")
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--merge", action="store_true", help="Also write a merged full model after training")
    p.add_argument("--preview", action="store_true", help="Print templated examples; do not load a model")
    p.add_argument("--eval-only", action="store_true", help="Load a saved adapter and run held-out generation")
    p.add_argument("--adapter-path", type=Path, default=None, help="Existing adapter for --eval-only")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-eval-samples", type=int, default=4)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "messages" not in row:
                raise ValueError(f"{path}:{line_no} is missing a 'messages' list")
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train, eval_ = [], []
    for row in rows:
        bucket = eval_ if row.get("split") == "eval" else train
        bucket.append(row)
    if not eval_:
        # Tiny fallback so a custom file still gets a held-out example.
        eval_ = train[-1:]
        train = train[:-1] or list(eval_)
    return train, eval_


def fallback_qwen_chat_template(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    """Mirror Qwen Instruct formatting when no tokenizer is loaded.

    Training and real inference must use tokenizer.apply_chat_template
    instead of this function. Preview-only.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"{QWEN_IM_START}{role}\n{content}{QWEN_IM_END}\n")
    if add_generation_prompt:
        parts.append(f"{QWEN_IM_START}assistant\n")
    return "".join(parts)


def render_example(messages: list[dict[str, str]], tokenizer=None, add_generation_prompt: bool = False) -> str:
    if tokenizer is None:
        return fallback_qwen_chat_template(messages, add_generation_prompt=add_generation_prompt)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def preview(rows: list[dict[str, Any]], n: int = 2) -> None:
    print("=" * 72)
    print("PREVIEW — synthetic data only. Same template must be used later.")
    print("=" * 72)
    for row in rows[:n]:
        text = render_example(row["messages"])
        print(f"\n--- {row.get('id', 'row')} ({row.get('split', 'train')}) ---\n")
        print(text)
    print("\nTokenization cannot run in --preview without a tokenizer.")
    print("When you train, tokenizer.apply_chat_template does this step for you.")


def build_text_dataset(rows: list[dict[str, Any]], tokenizer):
    from datasets import Dataset

    texts = [render_example(row["messages"], tokenizer=tokenizer, add_generation_prompt=False) for row in rows]
    return Dataset.from_dict({"text": texts})


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant = None
    if args.load_in_4bit or args.load_in_8bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit and not args.load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.bfloat16 if quant is None else None,
    )
    model.config.use_cache = False
    return model, tokenizer


def attach_lora(model, args: argparse.Namespace):
    from peft import LoraConfig, TaskType, get_peft_model

    # Qwen2 / Qwen2.5 dense layers used by attention + MLP.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def train(args: argparse.Namespace, train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> Path:
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = load_model_and_tokenizer(args)
    model = attach_lora(model, args)
    train_ds = build_text_dataset(train_rows, tokenizer)
    eval_ds = build_text_dataset(eval_rows, tokenizer)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
        report_to=[],
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    LOGGER.info("Saved adapter + tokenizer to %s", args.output_dir)

    if args.merge:
        if args.load_in_4bit or args.load_in_8bit:
            LOGGER.warning("Skip merge: unload 4/8-bit first and reload in bf16 before merging.")
        else:
            merged_dir = args.output_dir / "merged"
            merged = model.merge_and_unload()
            merged.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))
            LOGGER.info("Saved merged model to %s", merged_dir)

    generate_held_out(model, tokenizer, eval_rows, args.max_eval_samples)
    return args.output_dir


def generate_held_out(model, tokenizer, eval_rows: list[dict[str, Any]], limit: int) -> None:
    import torch

    model.eval()
    print("\n" + "=" * 72)
    print("HELD-OUT GENERATION (synthetic only)")
    print("=" * 72)
    for row in eval_rows[:limit]:
        prompt_messages = [m for m in row["messages"] if m["role"] != "assistant"]
        prompt = render_example(prompt_messages, tokenizer=tokenizer, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        print(f"\n--- {row.get('id', 'eval')} ---")
        print(completion.strip())
        gold = next((m["content"] for m in row["messages"] if m["role"] == "assistant"), "")
        print("\n[reference]\n" + gold[:400])


def eval_only(args: argparse.Namespace, eval_rows: list[dict[str, Any]]) -> None:
    from peft import PeftModel

    adapter = args.adapter_path or args.output_dir
    model, tokenizer = load_model_and_tokenizer(args)
    model = PeftModel.from_pretrained(model, str(adapter))
    generate_held_out(model, tokenizer, eval_rows, args.max_eval_samples)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    random.seed(args.seed)

    rows = load_jsonl(args.data)
    if any(not row.get("synthetic", False) for row in rows):
        LOGGER.warning("A row is missing synthetic=true. Refuse real PHI. Check the file.")
    train_rows, eval_rows = split_rows(rows)
    LOGGER.info("Loaded %d train / %d eval rows from %s", len(train_rows), len(eval_rows), args.data)

    if args.preview:
        preview(train_rows + eval_rows)
        return
    if args.eval_only:
        eval_only(args, eval_rows)
        return
    train(args, train_rows, eval_rows)


if __name__ == "__main__":
    main()

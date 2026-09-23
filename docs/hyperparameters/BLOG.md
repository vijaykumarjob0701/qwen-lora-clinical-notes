# The recipe is not the ingredients

You already have a base model. You already have a dataset. Those are the ingredients. This post is about the **recipe settings** — the hyperparameters that decide how the adapter is baked and, later, how a frozen model is sampled.

The long, linkable version (every knob, a GIF, ranges, mistakes, code) is the [README](README.md) in this folder. This page is the narrative you can read in one sitting. The parent tutorial is the [Qwen LoRA clinical-notes README](../../README.md).

## Two dashboards

Training knobs write **weights**. Inference knobs write **nothing**. They only choose the next token from a model that is done learning.

If the SOAP format never appears, you are on the training dashboard (`LoraConfig`, `TrainingArguments`). If the format is there but tonight’s bot rambles, you are on the inference dashboard (`temperature`, `top_p`, `max_new_tokens`, stops). Sliding temperature during a fine-tune will not move the loss. Raising LoRA rank will not make a live reply less repetitive.

## A boring recipe that works

For LoRA SFT on a 7B instruct model (Qwen2.5-class) on a single GPU:

- **LR `2e-4`**, cosine, **warmup 3%**, **AdamW**, **weight decay `0.01`**
- Micro-batch **1**, accumulate **8** (effective 8; 16 if you have room)
- **2 epochs**, then believe the held-out curve
- LoRA **`r=16`**, **`alpha=32`**, **dropout `0.05`**, attention **and** MLP
- **`max_seq_length=1024`** unless your chats are honestly longer

For chat inference after the adapter exists:

- **`temperature=0.7`**, **`top_p=0.9`**, **`top_k=50`** (or off)
- **`max_new_tokens=512`**
- Stop on the **real EOS** for that chat template

For JSON or extraction, drop temperature toward `0` and keep the stop list strict. Do not debug a bad adapter by setting temperature to 1.3.

## What each family of knobs is for

**Step size and schedule.** Learning rate is how big a downhill step you take. Warmup keeps step 0 from using the full step on a random adapter. Weight decay is a small leash on parameter size. AdamW is the default walker; 8-bit AdamW is the same walker in a smaller pair of shoes.

**How much data per update.** Batch size is noise in the gradient. Accumulation fakes a bigger batch without a bigger card. Epochs are how many times you repeat *this* JSONL. More epochs on 150 similar rows is memorization, not wisdom.

**Adapter shape.** Rank is capacity (how rich `ΔW` may be). Alpha is volume (`scale = α / r`). Dropout randomly silences adapter units while training. Target modules are *which* frozen linears get an `A`/`B` pair — and they are **name strings for this checkpoint**. Wrong names → 0% trainable.

**What the loss is allowed to see.** `max_seq_length` truncates **training** tokens. It loves to eat the assistant answer at the end of a long prompt. It is not `max_new_tokens`.

**How the frozen model talks.** Temperature sharpens or flattens the next-token distribution. Top-p keeps a probability-mass nucleus. Top-k keeps a fixed head-count. `max_new_tokens` is a hard length wall. Stop sequences cancel the remaining budget when a marker (`<|im_end|>`, EOS) appears.

## One change per run

Learning rate and LoRA alpha both turn the volume up. Rank, epochs, and dropout are the overfitting triangle. Sequence length and micro-batch are the OOM couple. Temperature and top-p reshape then clip the same distribution — A/B one of them. Max tokens and stops: whichever fires first wins.

Name the run after the pair you changed (`lr2e-4_r16_a32`) so you can read the result later.

## When it breaks

- **NaN** → learning rate (then clip, then dtype).
- **OOM** → max sequence length, then micro-batch; accumulate to keep the effective batch.
- **Flat loss / same as base** → LR too low or target-module names from another family.
- **Train perfect, live bad** → too many epochs or too much rank on too little data.
- **Cut off mid-sentence** → `max_new_tokens` (live) or `max_seq_length` (if the *labels* were chopped).
- **Sure, sure, sure** → temperature / top-p / missing EOS, not “need rank 64.”

Pictures for every knob, the when-to-tune table, and copy-paste `TrainingArguments` / `LoraConfig` / `generate` blocks live in the [README](README.md). Rebuild the GIFs with `python scripts/generate_hyperparameter_gifs.py`.

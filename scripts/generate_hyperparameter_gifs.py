#!/usr/bin/env python3
"""Generate short looping GIFs that show what each LLM hyperparameter does.

Run from the repo root:

    python scripts/generate_hyperparameter_gifs.py
    python scripts/generate_hyperparameter_gifs.py --only temperature
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "hyperparameters" / "gifs"

BG = "#f6f1e8"
INK = "#1c1917"
MUTED = "#78716c"
BLUE = "#2563eb"
ORANGE = "#ea580c"
GREEN = "#059669"
PURPLE = "#7c3aed"
RED = "#dc2626"
TEAL = "#0d9488"
GOLD = "#d97706"
PANEL = "#fffdf8"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#d6d3d1",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 10,
    }
)


def fig_to_image(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_gif(frames: list[Image.Image], name: str, duration_ms: int = 90) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    if not frames:
        raise ValueError(f"no frames for {name}")
    # Quantize with a shared adaptive palette so loops don't flicker.
    first = frames[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=160)
    rest = [im.convert("P", palette=Image.Palette.ADAPTIVE, colors=160) for im in frames[1:]]
    first.save(
        path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"  wrote {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB, {len(frames)} frames)")
    return path


def new_fig(width: float = 8.4, height: float = 4.55):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(BG)
    return fig, ax


def pingpong(values: np.ndarray) -> np.ndarray:
    return np.concatenate([values, values[-2:0:-1]])


# ---------------------------------------------------------------------------
# Training GIFs
# ---------------------------------------------------------------------------


def gif_learning_rate() -> None:
    # L = 0.45 x^2  →  grad = 0.9 x.  Convergence needs lr < 2/0.9 ≈ 2.22.
    xs = np.linspace(-2.6, 2.6, 400)
    ys = 0.45 * xs**2
    configs = [
        ("too high — overshoots", 2.32, -2.15, RED, []),
        ("typical — walks downhill", 0.42, -2.00, GREEN, []),
        ("too low — crawls", 0.045, -1.85, BLUE, []),
    ]
    for cfg in configs:
        x = cfg[2]
        for _ in range(30):
            cfg[4].append(x)
            x = float(np.clip(x - cfg[1] * (0.9 * x), -2.55, 2.55))

    frames = []
    for t in range(30):
        fig, ax = new_fig()
        ax.plot(xs, ys, color=INK, lw=2)
        ax.set_xlim(-2.7, 2.7)
        ax.set_ylim(-0.15, 3.15)
        ax.set_title("Learning rate — step size on the loss bowl")
        ax.set_xlabel("weight")
        ax.set_ylabel("loss")
        ax.axhline(0, color="#e7e5e4", lw=1)
        for label, _lr, _x0, color, traj in configs:
            hist = traj[: t + 1]
            ax.plot(hist, [0.45 * x**2 for x in hist], color=color, lw=1.6, alpha=0.55)
            x = hist[-1]
            ax.scatter([x], [0.45 * x**2], s=70, color=color, zorder=5, label=label)
        ax.legend(loc="upper right", fontsize=8, frameon=True, fancybox=False)
        ax.text(
            0.02,
            0.04,
            "High LR overshoots. Tiny LR crawls. A good LR walks downhill.",
            transform=ax.transAxes,
            fontsize=9,
            color=MUTED,
        )
        frames.append(fig_to_image(fig))
    save_gif(frames, "learning_rate.gif", 95)


def gif_batch_size() -> None:
    rng = np.random.default_rng(7)
    true_g = np.array([1.0, 0.15])
    examples = true_g + rng.normal(0, 0.85, size=(48, 2))
    frames = []
    schedule = [1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32]
    schedule = schedule + schedule[::-1]
    for n in schedule:
        fig, ax = new_fig()
        ax.set_xlim(-2.2, 3.2)
        ax.set_ylim(-2.2, 2.4)
        ax.set_aspect("equal")
        ax.set_title("Batch size — noisy vs stable gradient estimates")
        ax.axhline(0, color="#e7e5e4")
        ax.axvline(0, color="#e7e5e4")
        pick = examples[:n]
        ax.scatter(pick[:, 0], pick[:, 1], s=22, color=ORANGE, alpha=0.55, label="one-example grads")
        mean = pick.mean(axis=0)
        ax.arrow(
            0,
            0,
            true_g[0],
            true_g[1],
            width=0.03,
            head_width=0.14,
            color=GREEN,
            length_includes_head=True,
            label="true gradient",
        )
        ax.arrow(
            0,
            0,
            mean[0],
            mean[1],
            width=0.035,
            head_width=0.16,
            color=BLUE,
            length_includes_head=True,
            label="mini-batch mean",
        )
        ax.legend(loc="upper left", fontsize=8)
        ax.text(
            0.02,
            0.06,
            f"batch size = {n}     (accumulation of k micro-batches of m ≈ one batch of k·m)",
            transform=ax.transAxes,
            fontsize=9,
            color=INK,
        )
        frames.append(fig_to_image(fig))
    save_gif(frames, "batch_size.gif", 140)


def gif_epochs() -> None:
    epochs = np.arange(0, 13)
    train = 1.55 * np.exp(-0.38 * epochs) + 0.08
    val = 1.50 * np.exp(-0.32 * epochs) + 0.07 + 0.018 * np.maximum(0, epochs - 4) ** 1.6
    frames = []
    for t in range(1, len(epochs)):
        fig, ax = new_fig()
        ax.plot(epochs[: t + 1], train[: t + 1], color=BLUE, lw=2.3, label="train loss")
        ax.plot(epochs[: t + 1], val[: t + 1], color=ORANGE, lw=2.3, label="held-out loss")
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 1.75)
        ax.set_xlabel("epoch (one full pass over the dataset)")
        ax.set_ylabel("loss")
        ax.set_title("Epochs — underfit, then useful, then memorization")
        ax.axvspan(0, 2.2, color=BLUE, alpha=0.06)
        ax.axvspan(2.2, 5.3, color=GREEN, alpha=0.10)
        ax.axvspan(5.3, 12, color=RED, alpha=0.06)
        ax.text(0.9, 1.62, "underfit", color=BLUE, fontsize=9)
        ax.text(3.1, 1.62, "sweet spot", color=GREEN, fontsize=9)
        ax.text(8.2, 1.62, "overfit", color=RED, fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.scatter([epochs[t]], [train[t]], color=BLUE, s=40, zorder=5)
        ax.scatter([epochs[t]], [val[t]], color=ORANGE, s=40, zorder=5)
        frames.append(fig_to_image(fig))
    # hold last frame
    frames.extend([frames[-1]] * 4)
    save_gif(frames, "epochs.gif", 160)


def gif_warmup_ratio() -> None:
    steps = np.arange(0, 200)
    peak = 2e-4

    def schedule(warmup_ratio: float) -> np.ndarray:
        warm = max(1, int(warmup_ratio * len(steps)))
        lr = np.zeros_like(steps, dtype=float)
        for i, s in enumerate(steps):
            if s < warm:
                lr[i] = peak * (s / warm)
            else:
                # cosine decay after warmup
                p = (s - warm) / (len(steps) - warm)
                lr[i] = 0.1 * peak + 0.9 * peak * 0.5 * (1 + np.cos(np.pi * p))
        return lr

    with_w = schedule(0.06)
    no_w = schedule(0.0)
    frames = []
    for t in pingpong(np.linspace(8, 199, 28, dtype=int)):
        fig, ax = new_fig()
        ax.plot(steps, no_w * 1e4, color=RED, lw=2, label="warmup_ratio = 0")
        ax.plot(steps, with_w * 1e4, color=GREEN, lw=2.2, label="warmup_ratio = 0.06")
        ax.axvline(t, color=PURPLE, ls="--", lw=1.2)
        ax.scatter([t], [no_w[t] * 1e4], color=RED, s=36, zorder=5)
        ax.scatter([t], [with_w[t] * 1e4], color=GREEN, s=36, zorder=5)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 2.4)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("learning rate  × 10⁴")
        ax.set_title("Warmup ratio — ease into the peak LR, then decay")
        ax.legend(loc="upper right", fontsize=8)
        ax.text(
            0.02,
            0.06,
            "Without warmup, step 0 already takes a full-size jump on random grads.",
            transform=ax.transAxes,
            fontsize=9,
            color=MUTED,
        )
        frames.append(fig_to_image(fig))
    save_gif(frames, "warmup_ratio.gif", 90)


def gif_weight_decay() -> None:
    rng = np.random.default_rng(3)
    n = 18
    w0 = rng.normal(0, 0.35, size=(n, 2))
    grads = rng.normal(0, 0.12, size=(40, n, 2))
    # Also a small consistent drift so weights grow without decay
    drift = rng.normal(0, 0.04, size=(n, 2))
    frames = []
    w_free = w0.copy()
    w_decay = w0.copy()
    for t in range(32):
        g = grads[t % len(grads)]
        w_free = w_free - 0.15 * g + drift
        w_decay = (1 - 0.08) * (w_decay - 0.15 * g)  # decoupled-ish decay toward 0
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.55))
        fig.patch.set_facecolor(BG)
        for ax, pts, title, color in (
            (axes[0], w_free, "weight_decay = 0  — weights wander", ORANGE),
            (axes[1], w_decay, "weight_decay = 0.01  — pulled toward 0", GREEN),
        ):
            ax.set_facecolor(PANEL)
            circ = plt.Circle((0, 0), 1.0, fill=False, ls="--", color="#d6d3d1")
            ax.add_patch(circ)
            ax.scatter(pts[:, 0], pts[:, 1], s=36, color=color, alpha=0.9)
            ax.set_xlim(-2.3, 2.3)
            ax.set_ylim(-2.3, 2.3)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("w₁")
            ax.set_ylabel("w₂")
            ax.text(0.05, 0.08, f"‖W‖ ≈ {np.linalg.norm(pts):.2f}", transform=ax.transAxes, fontsize=9)
        fig.suptitle("Weight decay — a gentle L2 leash on parameter size", fontsize=13, fontweight="bold", y=0.98)
        frames.append(fig_to_image(fig))
    save_gif(frames, "weight_decay.gif", 90)


def gif_optimizer() -> None:
    # Elongated valley: f = 0.08 x^2 + 3.2 y^2
    xs = np.linspace(-3.2, 3.2, 200)
    ys = np.linspace(-1.6, 1.6, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.08 * X**2 + 3.2 * Y**2

    def grad(p):
        return np.array([0.16 * p[0], 6.4 * p[1]])

    # SGD
    sgd = [np.array([-2.8, 1.15])]
    for _ in range(40):
        sgd.append(sgd[-1] - 0.085 * grad(sgd[-1]))

    # Adam
    adam = [np.array([-2.8, 1.15])]
    m = np.zeros(2)
    v = np.zeros(2)
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.18
    for t in range(1, 41):
        g = grad(adam[-1])
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        adam.append(adam[-1] - lr * mhat / (np.sqrt(vhat) + eps))

    frames = []
    for t in range(2, 36):
        fig, ax = new_fig()
        ax.contour(X, Y, Z, levels=14, colors="#a8a29e", linewidths=0.8)
        s = np.array(sgd[:t])
        a = np.array(adam[:t])
        ax.plot(s[:, 0], s[:, 1], color=ORANGE, lw=2, label="SGD — zigzags in a skinny valley")
        ax.plot(a[:, 0], a[:, 1], color=BLUE, lw=2.2, label="AdamW — adaptive per-coordinate steps")
        ax.scatter(s[-1, 0], s[-1, 1], color=ORANGE, s=50, zorder=5)
        ax.scatter(a[-1, 0], a[-1, 1], color=BLUE, s=50, zorder=5)
        ax.scatter([0], [0], marker="*", s=90, color=GREEN, zorder=6, label="minimum")
        ax.set_title("Optimizer — how the walk downhill is computed")
        ax.set_xlabel("parameter a")
        ax.set_ylabel("parameter b")
        ax.legend(loc="upper right", fontsize=8)
        frames.append(fig_to_image(fig))
    save_gif(frames, "optimizer.gif", 85)


def gif_lora_rank() -> None:
    rng = np.random.default_rng(11)
    # A structured "update" matrix with effective rank ~8
    u = rng.normal(size=(18, 8))
    v = rng.normal(size=(8, 18))
    target = u @ v
    target = target / np.max(np.abs(target))
    u_s, s, vt = np.linalg.svd(target, full_matrices=False)
    ranks = [1, 2, 4, 8, 16]
    # dwell on each rank
    sequence = []
    for r in ranks:
        sequence.extend([r] * 6)
    sequence = sequence + sequence[::-1]

    frames = []
    for r in sequence:
        approx = (u_s[:, :r] * s[:r]) @ vt[:r, :]
        err = np.mean((target - approx) ** 2)
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.55))
        fig.patch.set_facecolor(BG)
        for ax, mat, title in (
            (axes[0], target, "needed update  ΔW"),
            (axes[1], approx, f"LoRA rank r = {r}   (MSE {err:.3f})"),
        ):
            ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
            ax.set_title(title, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor(PANEL)
        fig.suptitle("LoRA rank — adapter capacity (how rich ΔW can be)", fontsize=13, fontweight="bold")
        frames.append(fig_to_image(fig))
    save_gif(frames, "lora_rank.gif", 110)


def gif_lora_alpha() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=8)
    x = x / np.linalg.norm(x)
    r = 8
    A = rng.normal(size=(r, 8)) * 0.25
    B = rng.normal(size=(8, r)) * 0.25
    update = B @ (A @ x)
    Wx = 0.55 * x
    alphas = pingpong(np.linspace(8, 64, 20))
    frames = []
    for alpha in alphas:
        scale = alpha / r
        y = Wx + scale * update
        fig, ax = new_fig()
        ax.set_xlim(-0.4, 9)
        ax.set_ylim(-1.6, 1.8)
        ax.axhline(0, color="#e7e5e4")
        idx = np.arange(8)
        ax.bar(idx - 0.2, Wx, width=0.38, color="#a8a29e", label="frozen W x")
        ax.bar(idx + 0.2, y, width=0.38, color=PURPLE, label=f"W x + (α/r) BAx    α={alpha:.0f}")
        ax.set_xticks(idx)
        ax.set_xticklabels([f"d{i}" for i in idx])
        ax.set_ylabel("activation")
        ax.set_title("LoRA alpha — volume knob  (scale = α / r)")
        ax.legend(loc="upper right", fontsize=8)
        ax.text(
            0.02,
            0.06,
            f"r = {r} so scale = {scale:.2f}.  Doubling α doubles the adapter’s kick.",
            transform=ax.transAxes,
            fontsize=9,
            color=MUTED,
        )
        frames.append(fig_to_image(fig))
    save_gif(frames, "lora_alpha.gif", 90)


def gif_lora_dropout() -> None:
    rng = np.random.default_rng(5)
    rows, cols = 5, 10
    frames = []
    for t in range(24):
        fig, ax = new_fig(8.4, 4.4)
        ax.set_xlim(-0.6, cols + 1.4)
        ax.set_ylim(-1.3, rows + 0.8)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("LoRA dropout — randomly silence adapter units while training")
        infer = t >= 18
        if infer:
            mask = np.ones((rows, cols), dtype=bool)
            caption = "inference: dropout off — full adapter"
            color_on = GREEN
        else:
            mask = rng.random((rows, cols)) > 0.18
            caption = "training: each step drops ~p of adapter activations"
            color_on = BLUE
        for i in range(rows):
            for j in range(cols):
                on = mask[i, j]
                circ = plt.Circle(
                    (j, rows - 1 - i),
                    0.32,
                    facecolor=color_on if on else "#e7e5e4",
                    edgecolor="#a8a29e",
                    lw=0.8,
                    alpha=1.0 if on else 0.7,
                )
                ax.add_patch(circ)
        ax.text(0, -0.75, caption, fontsize=11, color=INK)
        ax.text(cols - 0.2, -0.75, "p = 0.05–0.10 typical", fontsize=9, color=MUTED, ha="right")
        frames.append(fig_to_image(fig))
    save_gif(frames, "lora_dropout.gif", 150)


def gif_target_modules() -> None:
    modules = [
        ("q_proj", "attn"),
        ("k_proj", "attn"),
        ("v_proj", "attn"),
        ("o_proj", "attn"),
        ("gate_proj", "mlp"),
        ("up_proj", "mlp"),
        ("down_proj", "mlp"),
    ]
    modes = [
        ("attention only", {"attn"}),
        ("attention + MLP  (recommended)", {"attn", "mlp"}),
        ("MLP only  (unusual)", {"mlp"}),
    ]
    sequence = []
    for mode in modes:
        sequence.extend([mode] * 10)

    frames = []
    for title, families in sequence:
        fig, ax = new_fig(8.4, 4.5)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")
        ax.set_title("Target modules — which frozen layers get an A/B pair")
        ax.add_patch(FancyBboxPatch((0.4, 0.6), 9.2, 4.6, boxstyle="round,pad=0.15", facecolor=PANEL, edgecolor="#d6d3d1"))
        ax.text(5, 4.8, "one Qwen2.5 decoder block", ha="center", fontsize=11, color=MUTED)
        for i, (name, fam) in enumerate(modules):
            x = 0.8 + (i % 4) * 2.25
            y = 2.9 if fam == "attn" else 1.2
            if fam == "mlp":
                x = 1.6 + (i - 4) * 2.4
            on = fam in families
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    1.9,
                    0.95,
                    boxstyle="round,pad=0.08",
                    facecolor="#dbeafe" if on else "#f5f5f4",
                    edgecolor=BLUE if on else "#d6d3d1",
                    lw=2 if on else 1,
                )
            )
            ax.text(x + 0.95, y + 0.55, name, ha="center", va="center", fontsize=9, color=INK)
            ax.text(
                x + 0.95,
                y + 0.22,
                "LoRA on" if on else "frozen",
                ha="center",
                va="center",
                fontsize=8,
                color=BLUE if on else MUTED,
            )
        n = sum(1 for _, fam in modules if fam in families)
        ax.text(0.5, 0.2, f"{title}   →   {n} adapter pairs per block", fontsize=11, color=INK)
        frames.append(fig_to_image(fig))
    save_gif(frames, "target_modules.gif", 130)


def gif_max_seq_length() -> None:
    tokens = [
        "<s>",
        "Sys",
        "You",
        "are",
        "a",
        "tutor",
        "Usr",
        "Explain",
        "warmup",
        "ratio",
        "to",
        "a",
        "beginner",
        "using",
        "the",
        "loss",
        "curve",
        "and",
        "an",
        "example",
        "Asst",
        "Warmup",
        "is",
        "a",
        "short",
        "ramp",
        "...",
        "eos",
    ]
    n = len(tokens)
    lengths = pingpong(np.array([8, 12, 16, 20, 24, 28]))
    frames = []
    for L in lengths:
        fig, ax = new_fig(8.6, 4.4)
        ax.set_xlim(-0.4, n + 0.2)
        ax.set_ylim(-1.2, 3.4)
        ax.axis("off")
        ax.set_title("Max sequence length — tokens past the budget are dropped")
        for i, tok in enumerate(tokens):
            kept = i < L
            ax.add_patch(
                FancyBboxPatch(
                    (i, 1.1),
                    0.92,
                    1.25,
                    boxstyle="round,pad=0.04",
                    facecolor="#dbeafe" if kept else "#fecaca",
                    edgecolor=BLUE if kept else RED,
                    alpha=1.0 if kept else 0.55,
                )
            )
            ax.text(i + 0.46, 1.72, tok, ha="center", va="center", fontsize=7, color=INK, rotation=70)
        ax.axvline(L, color=RED, lw=2)
        ax.text(L + 0.1, 2.7, f"max_seq_length = {L}", color=RED, fontsize=10)
        ax.text(
            0,
            0.15,
            "Kept for loss  →" if L < n else "Whole chat fits",
            fontsize=10,
            color=BLUE,
        )
        if L < n:
            ax.text(L + 0.15, 0.15, "truncated (often the assistant answer!)", fontsize=10, color=RED)
        frames.append(fig_to_image(fig))
    save_gif(frames, "max_seq_length.gif", 220)


# ---------------------------------------------------------------------------
# Inference GIFs
# ---------------------------------------------------------------------------


def _demo_logits() -> tuple[np.ndarray, list[str]]:
    labels = ["the", "a", "this", "patient", "cat", "quantum", "xyz", "banana"]
    logits = np.array([4.2, 3.4, 2.6, 1.8, 0.4, -0.2, -1.1, -1.6])
    return logits, labels


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def gif_temperature() -> None:
    logits, labels = _demo_logits()
    temps = pingpong(np.array([0.2, 0.4, 0.7, 1.0, 1.3, 1.8]))
    frames = []
    for T in temps:
        p = softmax(logits / T)
        fig, ax = new_fig()
        colors = [BLUE if i < 3 else "#a8a29e" for i in range(len(p))]
        ax.bar(labels, p, color=colors)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("P(next token)")
        ax.set_title(f"Temperature — sharpen or flatten the next-token distribution   T = {T:.1f}")
        if T <= 0.45:
            note = "low T: peaked — safer, more repetitive"
        elif T <= 1.05:
            note = "mid T: default chat — a little variety"
        else:
            note = "high T: flat — creative, easier to go off the rails"
        ax.text(0.02, 0.92, note, transform=ax.transAxes, fontsize=10, color=INK)
        frames.append(fig_to_image(fig))
    save_gif(frames, "temperature.gif", 200)


def gif_top_p() -> None:
    logits, labels = _demo_logits()
    p = softmax(logits / 0.9)
    order = np.argsort(-p)
    ps = p[order]
    labs = [labels[i] for i in order]
    cums = np.cumsum(ps)
    values = pingpong(np.array([0.3, 0.5, 0.7, 0.9, 0.95]))
    frames = []
    for thresh in values:
        # Standard nucleus: add tokens in rank order until mass >= p.
        keep = np.zeros_like(ps, dtype=bool)
        running = 0.0
        for i, mass in enumerate(ps):
            keep[i] = True
            running += mass
            if running >= thresh:
                break
        fig, ax = new_fig()
        colors = [TEAL if k else "#e7e5e4" for k in keep]
        ax.bar(labs, ps, color=colors, edgecolor="#a8a29e")
        ax.set_ylim(0, 0.7)
        ax.set_ylabel("probability (sorted)")
        ax.set_title(f"Top-p (nucleus) — keep the smallest set whose mass ≥ p    p = {thresh:.2f}")
        ax.axhline(0, color="#e7e5e4")
        kept_mass = ps[keep].sum()
        ax.text(
            0.02,
            0.90,
            f"nucleus = {int(keep.sum())} tokens, mass {kept_mass:.2f}   (gray tokens are zeroed)",
            transform=ax.transAxes,
            fontsize=10,
        )
        frames.append(fig_to_image(fig))
    save_gif(frames, "top_p.gif", 220)


def gif_top_k() -> None:
    logits, labels = _demo_logits()
    p = softmax(logits / 0.9)
    order = np.argsort(-p)
    ps = p[order]
    labs = [labels[i] for i in order]
    ks = pingpong(np.array([1, 2, 3, 5, 8]))
    frames = []
    for k in ks:
        fig, ax = new_fig()
        colors = [ORANGE if i < k else "#e7e5e4" for i in range(len(ps))]
        ax.bar(labs, ps, color=colors, edgecolor="#a8a29e")
        ax.set_ylim(0, 0.7)
        ax.set_ylabel("probability (sorted)")
        ax.set_title(f"Top-k — keep only the k most likely tokens    k = {k}")
        ax.text(
            0.02,
            0.90,
            "Unlike top-p, k is a fixed count, not a probability budget.",
            transform=ax.transAxes,
            fontsize=10,
        )
        frames.append(fig_to_image(fig))
    save_gif(frames, "top_k.gif", 220)


def gif_max_tokens() -> None:
    words = ["The", "model", "starts", "writing", "a", "careful", "answer", "about", "warmup", "and", "then"]
    budgets = [4, 6, 8, 11]
    sequence = []
    for b in budgets:
        sequence.extend([b] * (b + 3))
    frames = []
    step = 0
    for b in sequence:
        n_show = min((step % (b + 3)) + 1, b)
        step += 1
        fig, ax = new_fig(8.6, 4.3)
        ax.set_xlim(-0.3, 12)
        ax.set_ylim(0, 4)
        ax.axis("off")
        ax.set_title("max_new_tokens — a hard cap on how long the model may talk")
        for i, w in enumerate(words):
            x = i * 1.05
            if i < n_show:
                ax.add_patch(
                    FancyBboxPatch((x, 1.6), 0.98, 0.7, boxstyle="round,pad=0.05", facecolor="#dbeafe", edgecolor=BLUE)
                )
                ax.text(x + 0.49, 1.95, w, ha="center", va="center", fontsize=8)
            elif i < b:
                ax.add_patch(
                    FancyBboxPatch((x, 1.6), 0.98, 0.7, boxstyle="round,pad=0.05", facecolor="#f5f5f4", edgecolor="#e7e5e4")
                )
            else:
                ax.add_patch(
                    FancyBboxPatch((x, 1.6), 0.98, 0.7, boxstyle="round,pad=0.05", facecolor="#fecaca", edgecolor=RED, alpha=0.45)
                )
        ax.axvline(b * 1.05, color=RED, lw=2)
        ax.text(b * 1.05 + 0.08, 2.7, f"max_new_tokens = {b}", color=RED, fontsize=10)
        if n_show >= b and b < len(words):
            ax.text(0, 0.7, "cut off mid-thought — raise the cap or tighten the prompt", fontsize=10, color=RED)
        else:
            ax.text(0, 0.7, "still generating…", fontsize=10, color=MUTED)
        frames.append(fig_to_image(fig))
    save_gif(frames, "max_tokens.gif", 120)


def gif_stop_sequences() -> None:
    pieces = ["Sure", ",", " here", " is", " a", " haiku", ":\n", "soft", " rain", "\n\n", "Human", ":", " wait"]
    stop_at = pieces.index("\n\n")
    frames = []
    for t in range(1, len(pieces) + 6):
        n = min(t, len(pieces))
        stopped = n > stop_at + 1
        shown = pieces[: stop_at + 1] if stopped else pieces[:n]
        fig, ax = new_fig(8.6, 4.3)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4.2)
        ax.axis("off")
        ax.set_title("Stop sequences — end generation when a marker appears")
        text = "".join(shown)
        ax.add_patch(FancyBboxPatch((0.4, 1.3), 9.2, 2.0, boxstyle="round,pad=0.12", facecolor=PANEL, edgecolor="#d6d3d1"))
        ax.text(0.7, 2.3, text.replace("\n", "↵\n"), fontsize=13, family="DejaVu Sans Mono", va="center")
        ax.text(0.4, 0.7, "stop = ['\\n\\n', '</s>']     max_new_tokens still has budget left", fontsize=10, color=MUTED)
        if stopped:
            ax.text(0.4, 0.25, "hit stop sequence ↵↵  — remaining tokens are not sampled", fontsize=11, color=GREEN)
        frames.append(fig_to_image(fig))
    save_gif(frames, "stop_sequences.gif", 160)


GENERATORS = {
    "learning_rate": gif_learning_rate,
    "batch_size": gif_batch_size,
    "epochs": gif_epochs,
    "warmup_ratio": gif_warmup_ratio,
    "weight_decay": gif_weight_decay,
    "optimizer": gif_optimizer,
    "lora_rank": gif_lora_rank,
    "lora_alpha": gif_lora_alpha,
    "lora_dropout": gif_lora_dropout,
    "target_modules": gif_target_modules,
    "max_seq_length": gif_max_seq_length,
    "temperature": gif_temperature,
    "top_p": gif_top_p,
    "top_k": gif_top_k,
    "max_tokens": gif_max_tokens,
    "stop_sequences": gif_stop_sequences,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate educational GIFs for the hyperparameters guide.")
    parser.add_argument("--only", nargs="*", help="Subset of GIF keys to rebuild")
    parser.add_argument("--list", action="store_true", help="Print available keys")
    args = parser.parse_args()
    if args.list:
        print("\n".join(GENERATORS))
        return
    keys = args.only or list(GENERATORS)
    unknown = [k for k in keys if k not in GENERATORS]
    if unknown:
        raise SystemExit(f"unknown keys: {unknown}\nchoose from: {list(GENERATORS)}")
    print(f"Writing GIFs to {OUT_DIR}")
    for key in keys:
        print(f"• {key}")
        GENERATORS[key]()
    print("done")


if __name__ == "__main__":
    main()

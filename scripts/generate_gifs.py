#!/usr/bin/env python3
"""Generate looping educational GIFs for the tutorial README.

Creates:
  docs/gifs/tokens_splitting.gif
  docs/gifs/lora_adapter.gif
  docs/gifs/loss_curve.gif
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "gifs"
W, H = 880, 360
BG = (248, 250, 252)
INK = (15, 23, 42)
MUTED = (71, 85, 105)
SKY = (14, 165, 233)
EMERALD = (16, 185, 129)
AMBER = (245, 158, 11)
ROSE = (244, 63, 94)
SLATE = (148, 163, 184)
WHITE = (255, 255, 255)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/macos/Inter-SemiBold.ttf" if bold else "/usr/share/fonts/truetype/macos/Inter-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def new_canvas() -> Image.Image:
    return Image.new("RGB", (W, H), BG)


def rounded(draw: ImageDraw.ImageDraw, box, fill, outline=None, radius=12, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def save_gif(frames: list[Image.Image], name: str, duration: int = 180) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    return path


def tokens_splitting() -> Path:
    """Show a clinical phrase being split into subword tokens."""
    title_f = font(22, bold=True)
    body_f = font(18)
    small_f = font(15)
    token_f = font(16, bold=True)

    phrase = "The patient has hypertension."
    # Teaching-sized pieces (not claiming to be a specific tokenizer dump)
    pieces = ["The", "Ġpatient", "Ġhas", "Ġhyper", "tension", "."]
    colors = [SKY, SKY, SKY, AMBER, AMBER, SLATE]

    frames: list[Image.Image] = []
    # pause on the raw sentence, then reveal tokens one by one, then hold
    reveal_steps = list(range(0, len(pieces) + 1)) + [len(pieces)] * 4

    for n_show in reveal_steps:
        img = new_canvas()
        d = ImageDraw.Draw(img)
        d.text((28, 22), "Tokenization: text splits into subwords", fill=INK, font=title_f)
        d.text((28, 62), "Same tokenizer must be used at train, eval, and inference.", fill=MUTED, font=small_f)

        rounded(d, (28, 96, 852, 156), WHITE, outline=(203, 213, 225), radius=10)
        d.text((44, 114), phrase, fill=INK, font=body_f)

        x = 28
        y = 188
        for i, (tok, color) in enumerate(zip(pieces, colors)):
            if i >= n_show:
                break
            tw = token_f.getlength(tok) + 22
            if x + tw > 850:
                x = 28
                y += 54
            rounded(d, (x, y, x + tw, y + 40), color, radius=8)
            d.text((x + 10, y + 10), tok, fill=WHITE, font=token_f)
            x += tw + 10

        caption = "Whole sentence (not yet tokenized)" if n_show == 0 else f"{n_show} token(s)  •  hypertension → hyper + tension"
        d.text((28, 318), caption, fill=MUTED, font=small_f)
        frames.append(img)

    return save_gif(frames, "tokens_splitting.gif", duration=420)


def lora_adapter() -> Path:
    """Show LoRA A/B matrices attaching to frozen attention, then detaching."""
    title_f = font(22, bold=True)
    small_f = font(14)
    label_f = font(15, bold=True)

    # phases: empty, A in, A+B in, hold, detach B, detach A, hold empty
    phases = (
        ["empty"] * 2
        + ["a"] * 2
        + ["ab"] * 5
        + ["a"] * 2
        + ["empty"] * 2
    )

    frames: list[Image.Image] = []
    for phase in phases:
        img = new_canvas()
        d = ImageDraw.Draw(img)
        d.text((28, 18), "LoRA: freeze the base model, attach tiny adapters", fill=INK, font=title_f)
        d.text((28, 50), "W' = W + (alpha / r) * B A     •     only A and B train", fill=MUTED, font=small_f)

        # frozen base stack
        layers = ["q_proj (frozen)", "k_proj (frozen)", "v_proj (frozen)", "o_proj (frozen)"]
        for i, name in enumerate(layers):
            y0 = 88 + i * 58
            rounded(d, (48, y0, 300, y0 + 48), (226, 232, 240), outline=SLATE, radius=8)
            d.text((64, y0 + 14), name, fill=MUTED, font=label_f)

        # adapter column
        attach = phase in {"a", "ab"}
        ax = 430 if attach else 640
        rounded(d, (ax, 88, ax + 150, 136), EMERALD if attach else (203, 213, 225), radius=8)
        d.text((ax + 18, 104), "A  (r × k)", fill=WHITE if attach else MUTED, font=label_f)

        attach_b = phase == "ab"
        bx = 430 if attach_b else 640
        rounded(d, (bx, 204, bx + 150, 252), EMERALD if attach_b else (203, 213, 225), radius=8)
        d.text((bx + 18, 220), "B  (d × r)", fill=WHITE if attach_b else MUTED, font=label_f)

        # arrows into q and v
        if attach:
            d.line((300, 112, ax, 112), fill=EMERALD, width=3)
        if attach_b:
            d.line((300, 228, bx, 228), fill=EMERALD, width=3)

        status = {
            "empty": "Adapters detached  •  base model unchanged",
            "a": "Attaching A…",
            "ab": "Adapters attached  •  ready to train / serve",
        }[phase]
        d.text((28, 322), status, fill=EMERALD if phase == "ab" else MUTED, font=small_f)
        frames.append(img)

    return save_gif(frames, "lora_adapter.gif", duration=280)


def loss_curve() -> Path:
    """Animate train/eval loss decreasing over epochs."""
    title_f = font(22, bold=True)
    small_f = font(14)
    label_f = font(13)

    # synthetic, well-behaved teaching curve
    train = [2.40, 1.85, 1.42, 1.15, 0.98, 0.88]
    evall = [2.48, 1.96, 1.55, 1.28, 1.14, 1.10]
    n = len(train)

    left, right, top, bottom = 90, 830, 80, 290
    frames: list[Image.Image] = []

    def xy(i: int, yv: float) -> tuple[int, int]:
        x = left + int((right - left) * (i / (n - 1)))
        # map loss 0.6–2.6 onto the plot
        y = bottom - int((bottom - top) * ((yv - 0.6) / 2.0))
        return x, y

    for step in range(1, n + 1):
        for sub in (0, 1):  # hold each epoch two frames
            img = new_canvas()
            d = ImageDraw.Draw(img)
            d.text((28, 16), "Training: loss should trend down", fill=INK, font=title_f)
            d.text((28, 46), f"Epoch {step - 1}/{n - 1}   •   if this stays flat, check template, data, or LR", fill=MUTED, font=small_f)

            # axes
            d.line((left, top, left, bottom), fill=SLATE, width=2)
            d.line((left, bottom, right, bottom), fill=SLATE, width=2)
            d.text((36, top - 4), "loss", fill=MUTED, font=label_f)
            d.text((right - 40, bottom + 10), "epoch", fill=MUTED, font=label_f)

            for i in range(n):
                x, _ = xy(i, train[i])
                d.text((x - 4, bottom + 8), str(i), fill=MUTED, font=label_f)

            def draw_series(values: list[float], color, upto: int):
                pts = [xy(i, values[i]) for i in range(upto)]
                if len(pts) >= 2:
                    d.line(pts, fill=color, width=4)
                for p in pts:
                    d.ellipse((p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5), fill=color)

            draw_series(train, SKY, step)
            draw_series(evall, ROSE, step)

            d.ellipse((620, 318, 632, 330), fill=SKY)
            d.text((638, 314), "train loss", fill=INK, font=small_f)
            d.ellipse((750, 318, 762, 330), fill=ROSE)
            d.text((768, 314), "eval loss", fill=INK, font=small_f)
            frames.append(img)
            if sub == 1 and step == n:
                # extra hold on the finished curve
                frames.append(img.copy())
                frames.append(img.copy())

    return save_gif(frames, "loss_curve.gif", duration=320)


def main() -> None:
    paths = [tokens_splitting(), lora_adapter(), loss_curve()]
    for p in paths:
        print(p, p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()

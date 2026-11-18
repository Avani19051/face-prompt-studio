"""Steps 6-7: generate images of the learnt identity from a text prompt.

The trained LoRA is loaded on top of the base Stable Diffusion model. Writing
the trigger phrase (e.g. "sks person") in a prompt makes the U-Net draw that
person; the VAE then decodes the final latent into pixels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

DEFAULT_NEGATIVE = "blurry, deformed, disfigured, extra limbs, bad anatomy, low quality, watermark, text"


def load_meta(lora_dir: str | Path) -> dict:
    meta_path = Path(lora_dir) / "identity.json"
    return json.loads(meta_path.read_text()) if meta_path.exists() else {}


def load_pipeline(lora_dir: str | None, base_model: str | None = None, lora_scale: float = 1.0):
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    from compat import disable_old_torchao_for_peft

    disable_old_torchao_for_peft()
    meta = load_meta(lora_dir) if lora_dir else {}
    base_model = base_model or meta.get("base_model", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    if lora_dir:
        pipe.load_lora_weights(lora_dir, adapter_name="identity")
        pipe.set_adapters(["identity"], adapter_weights=[lora_scale])
    return pipe, meta


def generate(pipe, prompt: str, n: int = 4, seed: int = 0, steps: int = 30, guidance: float = 7.0,
             negative: str = DEFAULT_NEGATIVE, height: int = 512, width: int = 512) -> list[Image.Image]:
    g = torch.Generator(pipe.device).manual_seed(seed)
    return pipe(
        prompt,
        negative_prompt=negative,
        num_images_per_prompt=n,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width,
        generator=g,
    ).images


def grid(images: list[Image.Image], cols: int | None = None) -> Image.Image:
    cols = cols or len(images)
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    canvas = Image.new("RGB", (cols * w, rows * h), "white")
    for i, im in enumerate(images):
        canvas.paste(im, ((i % cols) * w, (i // cols) * h))
    return canvas


def main():
    ap = argparse.ArgumentParser(description="Generate images of the learnt identity.")
    ap.add_argument("--lora_dir", default="outputs/lora")
    ap.add_argument("-p", "--prompt", required=True, help='e.g. "sks person as an astronaut, cinematic"')
    ap.add_argument("-n", "--num_images", type=int, default=4)
    ap.add_argument("-o", "--output", default="outputs/generated")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--lora_scale", type=float, default=1.0)
    args = ap.parse_args()

    pipe, meta = load_pipeline(args.lora_dir, lora_scale=args.lora_scale)
    trigger = meta.get("trigger")
    if trigger and trigger not in args.prompt:
        print(f"note: the prompt does not contain the trigger phrase '{trigger}', so the face will not appear")

    images = generate(pipe, args.prompt, args.num_images, args.seed, args.steps, args.guidance)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stem = "".join(c if c.isalnum() else "_" for c in args.prompt)[:60]
    for i, im in enumerate(images):
        im.save(out / f"{stem}_{args.seed}_{i}.png")
    grid(images).save(out / f"{stem}_{args.seed}_grid.png")
    print(f"saved {len(images)} images to {out}")


if __name__ == "__main__":
    main()

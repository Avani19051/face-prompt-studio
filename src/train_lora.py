"""Steps 3-5: teach Stable Diffusion a new identity with LoRA.

Instead of training an identity encoder (as Face2Diffusion does), we fine-tune
small low-rank adapters (LoRA) inside the U-Net's attention layers on 10-20
aligned photos of one person, all captioned "a photo of <token> <class>".
The base model's weights stay frozen; only ~3 MB of adapter weights are learnt.

Training objective (standard latent-diffusion loss):
    z0      = VAE.encode(image) * scaling_factor
    zt      = sqrt(a_t) * z0 + sqrt(1 - a_t) * eps        (random t, eps ~ N(0, I))
    loss    = || eps - UNet(zt, t, CLIP(caption)) ||^2

The loop is deliberately written out by hand (no Trainer / Accelerate wrapper)
so every step is visible. It runs on a free Colab T4 (16 GB) in fp16.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


class FaceDataset(Dataset):
    """Aligned face crops, lightly augmented. The caption is the same for all."""

    def __init__(self, folder: str, resolution: int, repeats: int = 1):
        self.paths = [p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        if len(self.paths) < 3:
            raise ValueError(f"need at least 3 aligned photos in {folder}, found {len(self.paths)}")
        self.repeats = repeats
        self.tf = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # [0, 1] -> [-1, 1], what the VAE expects
            ]
        )

    def __len__(self):
        return len(self.paths) * self.repeats

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i % len(self.paths)]).convert("RGB"))


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def encode_prompt(tokenizer, text_encoder, prompt: str, device) -> torch.Tensor:
    ids = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    return text_encoder(ids)[0]


def save_lora(unet, out_dir: Path, meta: dict):
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    out_dir.mkdir(parents=True, exist_ok=True)
    lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(out_dir, unet_lora_layers=lora_state, safe_serialization=True)
    (out_dir / "identity.json").write_text(json.dumps(meta, indent=2))


@torch.no_grad()
def preview(args, unet, step: int, out_dir: Path):
    """Render a few validation images with the adapters as they are right now."""
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model, unet=unet, torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    g = torch.Generator("cuda").manual_seed(args.seed)
    with torch.autocast("cuda"):
        imgs = pipe(args.validation_prompt, num_images_per_prompt=4, num_inference_steps=25, generator=g).images
    grid = Image.new("RGB", (imgs[0].width * 4, imgs[0].height))
    for k, im in enumerate(imgs):
        grid.paste(im, (k * im.width, 0))
    out_dir.mkdir(parents=True, exist_ok=True)
    grid.save(out_dir / f"preview_step{step:05d}.png")
    del pipe
    torch.cuda.empty_cache()


def main(args):
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig
    from transformers import CLIPTextModel, CLIPTokenizer

    assert torch.cuda.is_available(), "training needs a GPU (Colab: Runtime > Change runtime type > T4 GPU)"
    device = torch.device("cuda")
    seed_everything(args.seed)
    out_dir = Path(args.output_dir)

    # --- frozen pieces of Stable Diffusion -------------------------------------------
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder").to(device, torch.float16)
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae").to(device, torch.float16)
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet").to(device)
    scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    for m in (text_encoder, vae, unet):
        m.requires_grad_(False)

    # --- add trainable LoRA adapters to the U-Net attention projections ---------------
    unet.add_adapter(
        LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian", target_modules=LORA_TARGETS)
    )
    unet.enable_gradient_checkpointing()
    params = [p for p in unet.parameters() if p.requires_grad]
    print(f"trainable LoRA parameters: {sum(p.numel() for p in params) / 1e6:.2f} M")

    # --- data ------------------------------------------------------------------------
    caption = f"a photo of {args.token} {args.class_word}"
    cond = encode_prompt(tokenizer, text_encoder, caption, device)          # (1, 77, 768)
    ds = FaceDataset(args.instance_dir, args.resolution, repeats=100)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    warmup = max(1, args.max_steps // 20)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, args.max_steps) / args.max_steps))
    )
    scaler = torch.cuda.amp.GradScaler()

    # --- training loop ---------------------------------------------------------------
    unet.train()
    step, bar, data_iter = 0, tqdm(total=args.max_steps, desc="training"), iter(dl)
    while step < args.max_steps:
        try:
            pixels = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            pixels = next(data_iter)
        pixels = pixels.to(device, torch.float16)

        with torch.no_grad():
            latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy = scheduler.add_noise(latents, noise, t)
        target = noise if scheduler.config.prediction_type == "epsilon" else scheduler.get_velocity(latents, noise, t)

        with torch.autocast("cuda", dtype=torch.float16):
            pred = unet(noisy, t, encoder_hidden_states=cond.expand(latents.shape[0], -1, -1)).sample
        loss = F.mse_loss(pred.float(), target.float())

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        lr_sched.step()

        step += 1
        bar.update(1)
        bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_sched.get_last_lr()[0]:.1e}")

        if args.preview_every and step % args.preview_every == 0:
            unet.eval()
            preview(args, unet, step, out_dir / "previews")
            unet.train()

    meta = {
        "base_model": args.base_model,
        "token": args.token,
        "class_word": args.class_word,
        "trigger": f"{args.token} {args.class_word}",
        "rank": args.rank,
        "steps": args.max_steps,
        "num_photos": len(ds.paths),
    }
    save_lora(unet, out_dir, meta)
    print(f"saved LoRA to {out_dir}  (use '{meta['trigger']}' in prompts)")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="LoRA fine-tune Stable Diffusion on one person's face.")
    ap.add_argument("--instance_dir", required=True, help="folder of aligned face crops (from face_prep.py)")
    ap.add_argument("--output_dir", default="outputs/lora")
    ap.add_argument("--base_model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--token", default="sks", help="rare word that will stand for this person")
    ap.add_argument("--class_word", default="person")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--preview_every", type=int, default=200, help="0 disables previews")
    ap.add_argument("--validation_prompt", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if args.validation_prompt is None:
        args.validation_prompt = f"a portrait photo of {args.token} {args.class_word}, studio lighting"
    return args


if __name__ == "__main__":
    main(parse_args())

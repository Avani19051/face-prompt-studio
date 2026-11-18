"""Run every step in one go: align -> train -> generate -> score.

    python run_pipeline.py --photos path/to/photos

Finds the photos (searching sub-folders too), then writes everything to outputs/:
    outputs/lora/                LoRA weights + previews during training
    outputs/generated/           images made WITH the LoRA
    outputs/baseline/            the same prompts WITHOUT the LoRA (for comparison)
    outputs/results.png          one sheet: your photos, then each prompt with scores
    outputs/identity_scores.csv  similarity score per image
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PROMPTS = [
    "a portrait photo of sks person as an astronaut, cinematic lighting, detailed",
    "sks person as a renaissance oil painting, museum lighting",
    "sks person reading a book in a cozy cafe, 35mm film photo",
    "a pencil sketch of sks person, detailed, black and white",
]


def find_photo_folder(root: str) -> Path:
    photos = [p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    if not photos:
        raise SystemExit(f"No .jpg/.png/.webp photos found under {root}. (iPhone .heic photos need converting first.)")
    folders = {}
    for p in photos:
        folders[p.parent] = folders.get(p.parent, 0) + 1
    folder = max(folders, key=folders.get)
    print(f"found {folders[folder]} photos in {folder}")
    return folder


def run(cmd: list[str]):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], check=True, cwd=ROOT)


def label(img, text, height=28):
    from PIL import Image, ImageDraw

    out = Image.new("RGB", (img.width, img.height + height), "white")
    out.paste(img, (0, 0))
    ImageDraw.Draw(out).text((6, img.height + 7), text, fill="black")
    return out


def results_sheet(aligned: Path, rows: list[tuple[str, list, list]], out_path: Path, thumb: int = 256,
                  scored: bool = True):
    from PIL import Image, ImageDraw

    refs = [Image.open(p).convert("RGB").resize((thumb, thumb)) for p in sorted(aligned.iterdir()) if p.suffix.lower() in IMAGE_EXTS][:4]
    lines = [("Your photos (aligned)", [label(r, "real photo") for r in refs])]
    for prompt, imgs, scores in rows:
        tiles = []
        for im, s in zip(imgs, scores):
            tag = "not scored" if not scored else ("no face found" if s is None else f"similarity {s:.2f}")
            tiles.append(label(im.resize((thumb, thumb)), tag))
        lines.append((prompt, tiles))

    cols = max(len(t) for _, t in lines)
    tile_h = thumb + 28
    title_h = 24
    sheet = Image.new("RGB", (cols * thumb, len(lines) * (tile_h + title_h)), "white")
    draw = ImageDraw.Draw(sheet)
    y = 0
    for title, tiles in lines:
        draw.text((6, y + 6), title, fill="black")
        for i, t in enumerate(tiles):
            sheet.paste(t, (i * thumb, y + title_h))
        y += tile_h + title_h
    sheet.save(out_path)
    print(f"saved results sheet -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", required=True, help="folder with 10-20 photos of one person (searched recursively)")
    ap.add_argument("--steps", type=int, default=800, help="training steps")
    ap.add_argument("--images_per_prompt", type=int, default=4)
    ap.add_argument("--skip_training", action="store_true", help="reuse an existing outputs/lora")
    ap.add_argument("--only_score", action="store_true",
                    help="reuse images already in outputs/generated and outputs/baseline; just score them and redraw the sheet")
    ap.add_argument("--skip_align", action="store_true",
                    help="photos are already aligned face crops (e.g. from scripts/get_demo_faces.py)")
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. On Kaggle: Settings > Accelerator > GPU T4. On Colab: Runtime > Change runtime type > T4 GPU.")
    print("GPU:", torch.cuda.get_device_name(0))

    aligned = ROOT / "data/aligned"
    lora = ROOT / "outputs/lora"
    gen_dir, base_dir = ROOT / "outputs/generated", ROOT / "outputs/baseline"

    # Steps 1-2: find, align and crop the faces
    photo_dir = find_photo_folder(args.photos)
    if args.only_score:
        args.skip_training = True
    if args.only_score and aligned.exists():
        pass
    elif args.skip_align:
        import shutil

        aligned.mkdir(parents=True, exist_ok=True)
        for p in photo_dir.iterdir():
            if p.suffix.lower() in IMAGE_EXTS:
                shutil.copy(p, aligned / (p.stem + ".png") if p.suffix.lower() == ".png" else aligned / p.name)
    else:
        run(["src/face_prep.py", "--input", str(photo_dir), "--output", str(aligned)])
    n_aligned = len([p for p in aligned.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if n_aligned < 3:
        raise SystemExit(f"Only {n_aligned} photos had a detectable face. Add clearer, front-facing photos.")

    # Steps 3-4: train the LoRA
    if not args.skip_training:
        run(["src/train_lora.py", "--instance_dir", str(aligned), "--output_dir", str(lora), "--max_steps", str(args.steps)])

    # Steps 5-7: generate with and without the LoRA
    gen_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    if args.only_score:
        from PIL import Image

        with_lora = [[Image.open(gen_dir / f"p{i}_{j}.png").convert("RGB") for j in range(args.images_per_prompt)]
                     for i in range(len(PROMPTS))]
    else:
        from generate import generate, load_pipeline

        pipe, _ = load_pipeline(str(lora))
        with_lora = []
        for i, p in enumerate(PROMPTS):
            print(f"generating: {p}")
            imgs = generate(pipe, p, n=args.images_per_prompt, seed=i)
            for j, im in enumerate(imgs):
                im.save(gen_dir / f"p{i}_{j}.png")
            with_lora.append(imgs)

        pipe.set_adapters(["identity"], adapter_weights=[0.0])  # same model, LoRA switched off
        for i, p in enumerate(PROMPTS):
            for j, im in enumerate(generate(pipe, p, n=args.images_per_prompt, seed=i)):
                im.save(base_dir / f"p{i}_{j}.png")
        del pipe
        torch.cuda.empty_cache()

    # Step 8: identity scores
    rows = [(p, imgs, [None] * len(imgs)) for p, imgs in zip(PROMPTS, with_lora)]
    scored = False
    try:
        from evaluate_identity import IdentityScorer, evaluate

        scorer = IdentityScorer()
        ref = scorer.reference(aligned)
        rows = [(p, imgs, [scorer.score(im, ref) for im in imgs]) for p, imgs in zip(PROMPTS, with_lora)]
        print("\nwith LoRA   :", evaluate(scorer, ref, gen_dir, ROOT / "outputs/identity_scores.csv"))
        print("without LoRA:", evaluate(scorer, ref, base_dir))
        scored = True
    except Exception as e:
        print("identity scoring skipped:", e)

    results_sheet(aligned, rows, ROOT / "outputs/results.png", scored=scored)


if __name__ == "__main__":
    main()

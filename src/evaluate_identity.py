"""Measure how well generated images keep the person's identity.

Each face is embedded with ArcFace (InsightFace `buffalo_l`), a face-recognition
network trained so that photos of the same person land close together. We
average the embeddings of the reference photos and report, for every generated
image, the cosine similarity to that average.

Rough guide for ArcFace cosine similarity:
    > 0.5   very likely the same person
    0.3-0.5 resemblance
    < 0.2   different person

Running the same prompts with and without the LoRA (`--baseline_dir`) shows how
much of the identity comes from fine-tuning.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class IdentityScorer:
    def __init__(self, ctx_id: int = 0):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def embed(self, image: Image.Image | np.ndarray) -> np.ndarray | None:
        rgb = np.array(image.convert("RGB") if isinstance(image, Image.Image) else image)
        faces = self.app.get(rgb[:, :, ::-1])  # insightface expects BGR
        if not faces:
            # Tight crops where the face fills the whole frame confuse the detector.
            # Shrink the face by padding a border around it and try again.
            pad = rgb.shape[0] // 2
            padded = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=127)
            faces = self.app.get(padded[:, :, ::-1])
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding

    def reference(self, folder: str | Path) -> np.ndarray:
        embs = [e for p in _images(folder) if (e := self.embed(Image.open(p))) is not None]
        if not embs:
            raise ValueError(f"no faces found in {folder}")
        mean = np.mean(embs, axis=0)
        return mean / np.linalg.norm(mean)

    def score(self, image, ref: np.ndarray) -> float | None:
        e = self.embed(image)
        return None if e is None else float(np.dot(e, ref))


def _images(folder) -> list[Path]:
    return [p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in IMAGE_EXTS and "grid" not in p.stem]


def summarize(scores: list[float | None]) -> dict:
    found = [s for s in scores if s is not None]
    return {
        "images": len(scores),
        "faces_found": len(found),
        "mean_similarity": float(np.mean(found)) if found else float("nan"),
        "share_above_0.5": float(np.mean([s > 0.5 for s in found])) if found else float("nan"),
    }


def evaluate(scorer: IdentityScorer, ref: np.ndarray, folder, csv_path: Path | None = None) -> dict:
    rows = [(p.name, scorer.score(Image.open(p), ref)) for p in _images(folder)]
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "identity_similarity"])
            w.writerows([(n, "" if s is None else f"{s:.4f}") for n, s in rows])
    return summarize([s for _, s in rows])


def main():
    ap = argparse.ArgumentParser(description="ArcFace identity similarity of generated images.")
    ap.add_argument("--reference_dir", required=True, help="the person's real (aligned) photos")
    ap.add_argument("--generated_dir", required=True, help="images generated with the LoRA")
    ap.add_argument("--baseline_dir", help="same prompts generated without the LoRA (optional)")
    ap.add_argument("--csv", default="outputs/identity_scores.csv")
    args = ap.parse_args()

    scorer = IdentityScorer()
    ref = scorer.reference(args.reference_dir)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    print("with LoRA :", evaluate(scorer, ref, args.generated_dir, Path(args.csv)))
    if args.baseline_dir:
        print("baseline  :", evaluate(scorer, ref, args.baseline_dir))


if __name__ == "__main__":
    main()

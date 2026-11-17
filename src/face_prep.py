"""Step 1 + 2: face detection, landmark extraction and alignment.

For every input photo we
  1. detect 68 facial landmarks with `face_alignment`,
  2. reduce them to 5 key points (two eye centres, nose tip, two mouth corners),
  3. estimate a similarity transform (rotation + uniform scale + translation)
     that moves those 5 points onto a fixed template,
  4. warp the photo with that transform.

The template is the standard 112x112 ArcFace layout. For training images we
scale the template up to the training resolution and add a margin, so the
whole head and some background stay in frame (Stable Diffusion learns better
from head-and-shoulders crops than from tight face crops).

The 5-point template idea follows the ArcFace / Face2Diffusion preprocessing;
the code here is written from scratch.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.transform import SimilarityTransform

# Standard ArcFace 5-point template for a 112x112 crop:
# left eye, right eye, nose tip, left mouth corner, right mouth corner
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def landmarks68_to_5(lmk68: np.ndarray) -> np.ndarray:
    """Reduce the 68-point iBUG layout to the 5 points the template uses."""
    lmk68 = np.asarray(lmk68, dtype=np.float32)
    if lmk68.shape != (68, 2):
        raise ValueError(f"expected (68, 2) landmarks, got {lmk68.shape}")
    eye_a = lmk68[36:42].mean(axis=0)   # image-left eye
    eye_b = lmk68[42:48].mean(axis=0)   # image-right eye
    nose = lmk68[30]
    mouth_a = lmk68[48]
    mouth_b = lmk68[54]
    return np.stack([eye_a, eye_b, nose, mouth_a, mouth_b])


def make_template(size: int, margin: float = 0.0) -> np.ndarray:
    """ArcFace template scaled to `size` pixels.

    `margin` shrinks the face inside the crop: 0.0 = tight 112-style crop,
    0.5 = the face occupies about half of the crop, with room for hair and
    shoulders.
    """
    if not 0.0 <= margin < 1.0:
        raise ValueError("margin must be in [0, 1)")
    scale = size / 112.0 * (1.0 - margin)
    offset = size * margin / 2.0
    return ARCFACE_TEMPLATE_112 * scale + offset


def alignment_matrix(lmk5: np.ndarray, template: np.ndarray) -> np.ndarray:
    """3x3 similarity transform mapping `lmk5` onto `template`."""
    src = np.asarray(lmk5, np.float64)
    dst = np.asarray(template, np.float64)
    if hasattr(SimilarityTransform, "from_estimate"):  # scikit-image >= 0.26
        tform = SimilarityTransform.from_estimate(src, dst)
        ok = bool(tform)
    else:
        tform = SimilarityTransform()
        ok = tform.estimate(src, dst)
    if not ok:
        raise RuntimeError("could not estimate the alignment transform")
    return tform.params


def warp(image: np.ndarray, matrix: np.ndarray, size: int) -> np.ndarray:
    import cv2  # imported lazily so the maths above is testable without OpenCV

    return cv2.warpAffine(
        image,
        matrix[:2],
        (size, size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


@dataclass
class AlignedFace:
    image: np.ndarray        # RGB uint8, size x size
    landmarks5: np.ndarray   # 5 key points in the original photo


class FaceAligner:
    """Wraps the landmark detector so it is loaded once and reused."""

    def __init__(self, device: str | None = None):
        import face_alignment
        import torch

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        landmarks_type = getattr(face_alignment.LandmarksType, "TWO_D", None) or face_alignment.LandmarksType._2D
        self.detector = face_alignment.FaceAlignment(landmarks_type, flip_input=False, device=device)

    def landmarks5(self, rgb: np.ndarray) -> np.ndarray | None:
        faces = self.detector.get_landmarks_from_image(rgb)
        if not faces:
            return None
        # keep the largest face if there are several
        largest = max(faces, key=lambda f: np.ptp(f[:, 0]) * np.ptp(f[:, 1]))
        return landmarks68_to_5(largest)

    def align(self, rgb: np.ndarray, size: int = 512, margin: float = 0.45) -> AlignedFace | None:
        lmk5 = self.landmarks5(rgb)
        if lmk5 is None:
            return None
        matrix = alignment_matrix(lmk5, make_template(size, margin))
        return AlignedFace(warp(rgb, matrix, size), lmk5)


def load_rgb(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def prepare_folder(src: str, dst: str, size: int = 512, margin: float = 0.45) -> int:
    """Align every photo in `src` and write the crops to `dst`. Returns the count."""
    aligner = FaceAligner()
    out_dir = Path(dst)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    for path in sorted(Path(src).iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        face = aligner.align(load_rgb(path), size=size, margin=margin)
        if face is None:
            print(f"  skipped {path.name}: no face found")
            continue
        Image.fromarray(face.image).save(out_dir / f"{path.stem}.png")
        kept += 1
    print(f"aligned {kept} photos -> {out_dir}")
    return kept


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Detect, align and crop faces for training.")
    ap.add_argument("--input", required=True, help="folder of raw photos")
    ap.add_argument("--output", required=True, help="folder for aligned crops")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--margin", type=float, default=0.45)
    args = ap.parse_args()
    prepare_folder(args.input, args.output, args.size, args.margin)

"""Download photos of ONE synthetic person from Microsoft's DigiFace-1M dataset.

DigiFace-1M faces are rendered with computer graphics: the people do not exist,
so the demo trains on nobody's real face. Licence: non-commercial research only
(https://github.com/microsoft/DigiFace1M).

The images are 112x112 face crops that are already aligned, so we only upscale
them to 512x512. Run the pipeline afterwards with --skip_align.

    python scripts/get_demo_faces.py --out data/demo_faces
    python run_pipeline.py --photos data/demo_faces --skip_align
"""
from __future__ import annotations

import argparse
import io
import re
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

URL = "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_0-1999_72_imgs.zip"


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/demo_faces")
    ap.add_argument("--identity", type=int, default=7, help="which synthetic person (0-1999)")
    ap.add_argument("--count", type=int, default=18, help="photos to keep (one accessory set = 18)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--cache", default="/tmp/digiface_p1.zip")
    args = ap.parse_args()

    zf = zipfile.ZipFile(download(URL, Path(args.cache)))
    pattern = re.compile(rf"(^|/){args.identity}/(\d+)\.png$")
    members = sorted(
        (m for m in zf.namelist() if pattern.search(m)),
        key=lambda m: int(pattern.search(m).group(2)),
    )
    if not members:
        raise SystemExit(f"identity {args.identity} not found in the archive")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for m in members[: args.count]:
        img = Image.open(io.BytesIO(zf.read(m))).convert("RGB")
        img.resize((args.size, args.size), Image.LANCZOS).save(out / Path(m).name)
    print(f"saved {min(len(members), args.count)} photos of synthetic person #{args.identity} -> {out}")


if __name__ == "__main__":
    main()

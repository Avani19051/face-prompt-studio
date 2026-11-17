"""End-to-end smoke test on a tiny random-weight Stable Diffusion (CPU, ~30 s, no downloads).

It checks that training runs, saves a LoRA, and that the LoRA loads and changes generated images.
It says nothing about image quality - that needs the real model on a GPU.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("diffusers")
pytest.importorskip("peft")
pytest.importorskip("torchvision")

ROOT = Path(__file__).resolve().parents[1]


def test_train_then_generate(tmp_path):
    from PIL import Image

    subprocess.run([sys.executable, str(ROOT / "tests/make_tiny_sd.py"), str(tmp_path)], check=True)
    faces = tmp_path / "aligned"
    faces.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(faces / f"{i}.png")

    lora = tmp_path / "lora"
    subprocess.run(
        [sys.executable, str(ROOT / "src/train_lora.py"), "--instance_dir", str(faces), "--output_dir", str(lora),
         "--base_model", str(tmp_path / "model"), "--resolution", "64", "--max_steps", "3", "--preview_every", "0",
         "--num_workers", "0", "--rank", "4", "--device", "cpu"],
        check=True,
    )
    assert (lora / "pytorch_lora_weights.safetensors").exists()
    meta = json.loads((lora / "identity.json").read_text())
    assert meta["trigger"] == "sks person"

    sys.path.insert(0, str(ROOT / "src"))
    from generate import generate, load_pipeline

    pipe, _ = load_pipeline(str(lora))
    on = np.asarray(generate(pipe, "sks person", n=1, seed=1, steps=3, height=64, width=64)[0], float)
    pipe.set_adapters(["identity"], adapter_weights=[0.0])
    off = np.asarray(generate(pipe, "sks person", n=1, seed=1, steps=3, height=64, width=64)[0], float)
    assert np.abs(on - off).mean() > 0  # the LoRA is actually applied

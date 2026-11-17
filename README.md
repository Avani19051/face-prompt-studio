# Face Prompt Studio

Teach Stable Diffusion one person's face from 10–20 photos, then put that person into any scene with a text prompt, for example *"sks person as an astronaut, cinematic lighting"*. Generated images are scored with a face-recognition model to measure how well the identity is kept.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Avani19051/face-prompt-studio/blob/main/notebooks/face_prompt_studio_colab.ipynb)

**Stack:** PyTorch · Hugging Face Diffusers · PEFT (LoRA) · Stable Diffusion 1.5 · CLIP · face_alignment · OpenCV · scikit-image · InsightFace (ArcFace) · Gradio

---

## 1. Pipeline at a glance

```
INPUT: 10–20 photos of one person
   │
   ▼
1. Face detection and landmark extraction ── face_alignment
   Detect 68 facial landmarks per photo and reduce them to 5 key points.
   │
   ▼
2. Face alignment and cropping ── OpenCV · NumPy · scikit-image
   Warp each face onto a fixed template; save 512 × 512 head-and-shoulders crops.
   │
   ▼
3. Caption and text conditioning ── CLIP · Transformers
   Every photo gets the caption "a photo of sks person"; CLIP encodes it once.
   │
   ▼
4. LoRA fine-tuning ── PyTorch · PEFT · Diffusers
   Train low-rank adapters in the U-Net's attention layers to denoise this face.
   │
   ▼
OUTPUT of training: a ~13 MB LoRA file that "knows" the person
   │
   ▼
5. Prompting ── any prompt containing "sks person"
   │
   ▼
6. Image generation / denoising ── Stable Diffusion 1.5 · DPM-Solver++ · CUDA
   U-Net (base + LoRA) iteratively denoises a random latent guided by the prompt.
   │
   ▼
7. Decoding and identity check ── VAE · ArcFace
   Decode the latent to an image, then score how similar the face is to the real photos.
   │
   ▼
OUTPUT: identity-preserving generated images + similarity scores
```

## 2. Each step explained: what, why and how

### Step 1 — Face detection and landmark extraction
- **What:** find the face in each photo and locate 68 points (jaw, brows, eyes, nose, mouth).
- **Why:** photos come at different angles, sizes and positions. Landmarks give us a precise, repeatable description of where the face is.
- **How:** `face_alignment` (a 2D landmark network). The 68 points are reduced to 5: the centre of each eye (mean of the 6 eye points), the nose tip and the two mouth corners. If a photo has several faces, the largest is kept. → `src/face_prep.py`

### Step 2 — Face alignment and cropping
- **What:** rotate, scale and shift each photo so the 5 key points land on the same template positions.
- **Why:** if every training image has the eyes in the same place, the model spends its capacity learning *who* the person is, not *where* the face is.
- **How:** estimate a similarity transform (rotation + uniform scale + translation) from the 5 points to the standard ArcFace 112 × 112 template, scaled up to 512 × 512 with a 45% margin so hair and shoulders stay in frame. Warp with `cv2.warpAffine`. Unit tests check that the transform exactly undoes a rotated, scaled and shifted face. → `src/face_prep.py`, `tests/test_face_prep.py`

### Step 3 — Caption and text conditioning
- **What:** pair every photo with the caption `a photo of sks person`.
- **Why:** `sks` is a rare token with almost no prior meaning. Training binds it to this person, while `person` keeps the model's general knowledge of what people look like.
- **How:** the CLIP tokenizer and text encoder turn the caption into a 77 × 768 embedding, computed once because the caption never changes. → `src/train_lora.py`

### Step 4 — LoRA fine-tuning
- **What:** adapt Stable Diffusion to this identity without retraining the 860 M-parameter U-Net.
- **Why:** full fine-tuning needs far more memory and tends to overfit on 15 photos. LoRA learns a small low-rank update `W + B·A` on top of each frozen weight, so it fits on a free T4 GPU. At rank 16 that is 3.2 M trainable parameters (0.4% of the U-Net), saved as a ~13 MB file.
- **How:**
  1. Encode the photo into a latent with the VAE: `z0 = VAE(x) · 0.18215`.
  2. Pick a random timestep `t` and add Gaussian noise: `zt = √ᾱt · z0 + √(1−ᾱt) · ε`.
  3. The U-Net, with LoRA adapters on `to_q`, `to_k`, `to_v`, `to_out` in every attention block, predicts the noise from `zt`, `t` and the caption embedding.
  4. Loss = MSE between predicted and true noise. Only the LoRA weights get gradients.
  5. Rank 16, AdamW at 1e-4, warm-up then cosine decay, fp16 mixed precision, gradient checkpointing, 800 steps by default. A preview grid is saved every 200 steps so you can see the face appear.

  → `src/train_lora.py`

### Step 5 — Prompting
- **What:** write any prompt that contains the trigger phrase `sks person`.
- **Why:** the trigger tells the model *who*; the rest of the prompt sets the scene, style and lighting.
- **Examples:** `sks person as an astronaut`, `oil painting of sks person`, `sks person reading in a cafe, 35mm film`.

### Step 6 — Image generation / denoising
- **What:** turn random noise into a picture that matches the prompt and contains the person.
- **How:** load Stable Diffusion 1.5, attach the LoRA (its strength is adjustable with `--lora_scale`), and run 30 steps of the DPM-Solver++ scheduler with classifier-free guidance (scale 7) and a negative prompt. → `src/generate.py`

### Step 7 — Decoding and identity check
- **What:** decode the final latent into an image, then measure whether it still looks like the person.
- **Why:** "looks similar" should be a number, not an opinion. It also shows what the LoRA adds compared with the base model.
- **How:** the VAE decoder produces the 512 × 512 image. InsightFace's ArcFace network embeds faces so that the same person gives nearby vectors. We average the embeddings of the real photos and report each generated image's cosine similarity to that average (above 0.5 is very likely the same person). Running the same prompts without the LoRA gives a baseline. → `src/evaluate_identity.py`

## 3. Quick start (Colab, free T4 GPU)

Open the notebook with the badge above and run the cells top to bottom. Or from a terminal with a GPU:

```bash
pip install -r requirements.txt

# 1–2: detect, align and crop
python src/face_prep.py --input data/raw --output data/aligned

# 3–4: train the LoRA (~15–25 min on a T4)
python src/train_lora.py --instance_dir data/aligned --output_dir outputs/lora

# 5–7: generate
python src/generate.py --lora_dir outputs/lora -p "a portrait of sks person as an astronaut, cinematic" -n 4

# identity check (optionally against a no-LoRA baseline)
python src/evaluate_identity.py --reference_dir data/aligned --generated_dir outputs/generated

# web demo
python app.py --lora_dir outputs/lora --reference_dir data/aligned --share
```

**Free GPUs:** training needs an NVIDIA GPU with CUDA. Both of these are free and come with PyTorch and CUDA already installed:

- **Google Colab:** use the notebook badge above, then Runtime → Change runtime type → T4 GPU. Free access isn't guaranteed at busy times.
- **Kaggle Notebooks:** about 30 GPU hours a week (a P100 or 2× T4). Upload your photos as a private Kaggle dataset. In a new notebook, turn on *Settings → Accelerator → GPU T4* and *Internet*, then run:

  ```bash
  !git clone -q https://github.com/Avani19051/face-prompt-studio.git && cd face-prompt-studio && pip -q install -r requirements.txt
  !cd face-prompt-studio && python src/face_prep.py --input /kaggle/input/<your-dataset> --output data/aligned \
      && python src/train_lora.py --instance_dir data/aligned --output_dir outputs/lora \
      && python src/generate.py --lora_dir outputs/lora -p "a portrait of sks person as an astronaut" -n 4
  ```

**Run the tests (no GPU needed):** `pytest -q tests`. The smoke test trains and generates with a tiny random-weight Stable Diffusion to check the code paths; image quality needs the real model on a GPU.

**Photo tips:** 10–20 photos, different angles, expressions, lighting and backgrounds, one person per photo, no sunglasses. Variety matters more than quantity.

## 4. Project structure

```
face-prompt-studio/
├── src/
│   ├── face_prep.py          # steps 1–2: landmarks, 5-point alignment, crops
│   ├── train_lora.py         # steps 3–4: captioning + hand-written LoRA training loop
│   ├── generate.py           # steps 5–7: load base + LoRA, generate, save grids
│   └── evaluate_identity.py  # ArcFace identity similarity + baseline comparison
├── app.py                    # Gradio demo with per-image identity scores
├── notebooks/face_prompt_studio_colab.ipynb
├── tests/
│   ├── test_face_prep.py       # alignment, warping and scoring tests (CPU, no downloads)
│   ├── test_pipeline_smoke.py  # train → save → load → generate on a tiny random SD (CPU, ~30 s)
│   └── make_tiny_sd.py         # builds that tiny model
└── requirements.txt
```

## 5. How this compares with Face2Diffusion

This project was inspired by [Face2Diffusion](https://github.com/mapooon/Face2Diffusion) (Shiohara & Yamasaki, CVPR 2024), which puts a face into Stable Diffusion **from a single photo, without fine-tuning**. Face2Diffusion trains a face-identity encoder (MSID) and a mapping network that turns the face into two word embeddings inserted into the CLIP prompt.

| | Face2Diffusion | Face Prompt Studio |
|---|---|---|
| Photos needed | 1 | 10–20 |
| Per-person training | none (encoder pre-trained on a large dataset) | ~20 min LoRA fine-tune |
| Where identity enters | CLIP text embeddings (two pseudo-words) | U-Net attention weights (LoRA) |
| Face preprocessing | 68 → 5 landmarks, align to the 112 × 112 template | same idea, re-implemented, with 512 × 512 margin crops |
| Evaluation | paper benchmarks | ArcFace similarity per image + no-LoRA baseline |

The landmark-to-5-point alignment is the common ArcFace preprocessing that Face2Diffusion also uses. All code in this repository was written for this project; none is copied from Face2Diffusion, which is licensed CC BY-NC-SA 4.0.

## 6. Responsible use

Only train on photos of yourself or of people who have agreed to it. Don't use this to impersonate anyone or to create misleading or intimate images of real people.

## Acknowledgements

- [Face2Diffusion](https://github.com/mapooon/Face2Diffusion): the idea of identity-preserving prompting and the landmark alignment approach
- [DreamBooth](https://dreambooth.github.io/) (Ruiz et al., 2023): rare-token subject fine-tuning
- [LoRA](https://arxiv.org/abs/2106.09685) (Hu et al., 2021) and Hugging Face [Diffusers](https://github.com/huggingface/diffusers) / [PEFT](https://github.com/huggingface/peft)
- [face_alignment](https://github.com/1adrianb/face-alignment) and [InsightFace](https://github.com/deepinsight/insightface)

## License

MIT for the code in this repository. Stable Diffusion weights are covered by their own CreativeML OpenRAIL-M license; InsightFace's pre-trained models are for non-commercial research use.

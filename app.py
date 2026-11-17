"""Gradio demo: type a prompt with the trigger phrase, get images of the person.

    python app.py --lora_dir outputs/lora --reference_dir data/aligned --share
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr  # noqa: E402

from generate import DEFAULT_NEGATIVE, generate, load_pipeline  # noqa: E402

EXAMPLES = [
    "a portrait photo of {t} as an astronaut, cinematic lighting",
    "{t} as a renaissance oil painting, museum lighting",
    "a pixar-style 3d character of {t}, smiling",
    "{t} reading a book in a cozy cafe, film photo, 35mm",
    "a pencil sketch of {t}, detailed, black and white",
]


def build(args):
    pipe, meta = load_pipeline(args.lora_dir, lora_scale=args.lora_scale)
    trigger = meta.get("trigger", "sks person")

    scorer, ref = None, None
    if args.reference_dir:
        try:
            from evaluate_identity import IdentityScorer

            scorer = IdentityScorer()
            ref = scorer.reference(args.reference_dir)
        except Exception as e:  # identity scores are optional
            print("identity scoring disabled:", e)

    def run(prompt, negative, n, steps, guidance, seed):
        if trigger not in prompt:
            gr.Warning(f"Add '{trigger}' to the prompt, otherwise the face won't appear.")
        images = generate(pipe, prompt, int(n), int(seed), int(steps), float(guidance), negative)
        if scorer is None:
            return images
        captions = []
        for im in images:
            s = scorer.score(im, ref)
            captions.append("no face found" if s is None else f"identity similarity {s:.2f}")
        return list(zip(images, captions))

    with gr.Blocks(title="Face Prompt Studio") as demo:
        gr.Markdown(f"# Face Prompt Studio\nWrite **`{trigger}`** in the prompt wherever the person should appear.")
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Prompt", value=EXAMPLES[0].format(t=trigger), lines=2)
                negative = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE)
                with gr.Row():
                    n = gr.Slider(1, 4, value=4, step=1, label="Images")
                    steps = gr.Slider(15, 50, value=30, step=1, label="Steps")
                    guidance = gr.Slider(3, 12, value=7, step=0.5, label="Guidance")
                    seed = gr.Number(value=0, precision=0, label="Seed")
                go = gr.Button("Generate", variant="primary")
                gr.Examples([[e.format(t=trigger)] for e in EXAMPLES], inputs=[prompt])
            gallery = gr.Gallery(label="Results", columns=2, height=640)
        go.click(run, [prompt, negative, n, steps, guidance, seed], gallery)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_dir", default="outputs/lora")
    ap.add_argument("--reference_dir", default=None, help="enables per-image identity scores")
    ap.add_argument("--lora_scale", type=float, default=1.0)
    ap.add_argument("--share", action="store_true", help="public link (needed on Colab)")
    a = ap.parse_args()
    build(a).launch(share=a.share)

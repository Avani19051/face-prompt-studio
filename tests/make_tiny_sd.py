"""Build a tiny random-weight Stable Diffusion pipeline on disk (same architecture types, tiny sizes)."""
import json, os, sys, torch
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))

out = sys.argv[1]
os.makedirs(out + "/tok", exist_ok=True)
chars = list(bytes_to_unicode().values())
vocab = {c: i for i, c in enumerate(chars)}
for c in chars: vocab[c + "</w>"] = len(vocab)
vocab["<|startoftext|>"] = len(vocab); vocab["<|endoftext|>"] = len(vocab)
json.dump(vocab, open(out + "/tok/vocab.json", "w"))
open(out + "/tok/merges.txt", "w").write("#version: 0.2\n")
tok = CLIPTokenizer(out + "/tok/vocab.json", out + "/tok/merges.txt", model_max_length=77)

torch.manual_seed(0)
te = CLIPTextModel(CLIPTextConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=37, num_hidden_layers=2,
                                  num_attention_heads=4, max_position_embeddings=77, bos_token_id=vocab["<|startoftext|>"],
                                  eos_token_id=vocab["<|endoftext|>"], pad_token_id=vocab["<|endoftext|>"]))
unet = UNet2DConditionModel(block_out_channels=(32, 64), layers_per_block=1, sample_size=8, in_channels=4, out_channels=4,
                            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                            cross_attention_dim=32, attention_head_dim=4)
vae = AutoencoderKL(block_out_channels=[32, 64], norm_num_groups=32, in_channels=3, out_channels=3, latent_channels=4,
                    down_block_types=["DownEncoderBlock2D"] * 2, up_block_types=["UpDecoderBlock2D"] * 2, sample_size=64)
sched = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
pipe = StableDiffusionPipeline(vae=vae, text_encoder=te, tokenizer=tok, unet=unet, scheduler=sched,
                               safety_checker=None, feature_extractor=None, requires_safety_checker=False)
pipe.save_pretrained(out + "/model")
print("tiny SD saved")

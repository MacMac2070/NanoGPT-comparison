"""Evaluate the trained checkpoint: train/val loss, perplexity, bits per character.

The loss here is nats per BPE token, which is not comparable to the character
models. It is converted to bits per character using the tokenizer's measured
compression ratio on the validation split.
"""

import math

import torch

import config
import data
from model import GPT

torch.manual_seed(1234)

print(f"device: {config.DEVICE}")
tokenizer, vocab_size, train_data, val_data = data.load_dataset()

ckpt_path = config.CKPT_DIR / "ckpt.pt"
ckpt = torch.load(ckpt_path, weights_only=True, map_location=config.DEVICE)
hp = ckpt["hyperparams"]

model = GPT(
    vocab_size=hp["vocab_size"],
    n_embd=hp["n_embd"],
    n_layer=hp["n_layer"],
    n_head=hp["n_head"],
    block_size=hp["block_size"],
    d_ff=hp["d_ff"],
    dropout=hp["dropout"],
).to(config.DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()


@torch.no_grad()
def loss_on(split):
    losses = torch.zeros(config.eval_iters)
    for i in range(config.eval_iters):
        x, y = data.get_batch(split, train_data, val_data,
                              hp["block_size"], config.batch_size,
                              config.DEVICE)
        _, l = model(x, y)
        losses[i] = l.item()
    return losses.mean().item()


tr = loss_on("train")
va = loss_on("val")
n_params = sum(p.numel() for p in model.parameters())

val_text = data.decode(val_data, tokenizer)
chars_per_token = len(val_text) / len(val_data)
bpc = va / (chars_per_token * math.log(2))
random_bpc = math.log(hp["vocab_size"]) / (chars_per_token * math.log(2))

print()
print("=========== paper_gpt_2 (byte-level BPE) ===========")
print(f"checkpoint iteration : {ckpt['iteration']}")
print(f"parameters           : {n_params:,}")
print(f"vocab size           : {hp['vocab_size']}")
print(f"val split            : {len(val_data):,} tokens = {len(val_text):,} chars")
print(f"compression          : {chars_per_token:.4f} characters per token")
print(f"train loss           : {tr:.4f} nats/token")
print(f"val loss             : {va:.4f} nats/token")
print(f"overfit gap          : {va - tr:+.4f}")
print(f"val perplexity       : {math.exp(va):.3f} per token")
print(f"BITS PER CHARACTER   : {bpc:.4f}")
print(f"random baseline BPC  : {random_bpc:.4f}")
print(f"chars of context     : {hp['block_size'] * chars_per_token:,.0f}")

"""Evaluate the trained checkpoint: train/val loss, perplexity, bits per character."""

import math

import torch

import config
import data
from model import GPT

torch.manual_seed(1234)

print(f"device: {config.DEVICE}")
stoi, itos, vocab_size, train_data, val_data = data.load_dataset()

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

print()
print("=========== paper_gpt_1 (character-level) ===========")
print(f"checkpoint iteration : {ckpt['iteration']}")
print(f"parameters           : {n_params:,}")
print(f"vocab size           : {hp['vocab_size']}")
print(f"train loss           : {tr:.4f} nats/char")
print(f"val loss             : {va:.4f} nats/char")
print(f"overfit gap          : {va - tr:+.4f}")
print(f"val perplexity       : {math.exp(va):.3f} per character")
print(f"BITS PER CHARACTER   : {va / math.log(2):.4f}")
print(f"random baseline BPC  : {math.log(hp['vocab_size']) / math.log(2):.4f}")

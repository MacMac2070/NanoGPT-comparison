"""Evaluate the trained checkpoint: train/val loss, perplexity, bits per character."""

import math
import os

import torch

from model import GPT, GPTConfig

torch.manual_seed(1234)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"device: {device}")

repo_dir = os.path.dirname(os.path.abspath(__file__))
ckpt = torch.load(os.path.join(repo_dir, "ckpt.pt"),
                  weights_only=True, map_location=device)
cfg = ckpt["config"]
stoi = ckpt["tokenizer"]["stoi"]

with open(os.path.join(repo_dir, "input.txt"), "r", encoding="utf-8") as f:
    text = f.read()

full = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
n_train = int(0.9 * len(full))
train_data, val_data = full[:n_train], full[n_train:]

block_size = cfg["block_size"]
batch_size = 64
eval_iters = 200

model = GPT(GPTConfig(**cfg)).to(device)
model.load_state_dict(ckpt["model"])
model.eval()


def get_batch(split):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def loss_on(split):
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch(split)
        _, l = model(x, y)
        losses[i] = l.item()
    return losses.mean().item()


tr = loss_on("train")
va = loss_on("val")
n_params = sum(p.numel() for p in model.parameters())

print()
print("=========== paper_gpt_3 (character-level) ===========")
print(f"parameters           : {n_params:,}")
print(f"vocab size           : {cfg['vocab_size']}")
print(f"train loss           : {tr:.4f} nats/char")
print(f"val loss             : {va:.4f} nats/char")
print(f"overfit gap          : {va - tr:+.4f}")
print(f"val perplexity       : {math.exp(va):.3f} per character")
print(f"BITS PER CHARACTER   : {va / math.log(2):.4f}")
print(f"random baseline BPC  : {math.log(cfg['vocab_size']) / math.log(2):.4f}")

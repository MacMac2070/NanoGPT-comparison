"""Train a character-level GPT on input.txt.

Edit the hyperparameters below, then run:  python train.py
"""

import os
import time

import torch

from model import GPT, GPTConfig

# ----------------------------- hyperparameters ------------------------------
batch_size = 64      # sequences per optimisation step
block_size = 256     # context length in characters
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = True
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
eval_iters = 200
repo_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(repo_dir, "input.txt")
ckpt_path = os.path.join(repo_dir, "ckpt.pt")
# ----------------------------------------------------------------------------

torch.manual_seed(1337)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"using device: {device}")

# Load the corpus and build the character-level tokenizer from it.
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[ch] for ch in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


print(f"corpus: {len(text):,} characters, vocab size {vocab_size}")

# Encode the whole corpus once, then split 90/10 into train and validation.
data = torch.tensor(encode(text), dtype=torch.long)
n_train = int(0.9 * len(data))
train_data = data[:n_train]
val_data = data[n_train:]


def get_batch(split):
    """Sample a random batch of (input, target) character sequences."""
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    """Average the loss over eval_iters random batches for each split."""
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    bias=bias,
)
model = GPT(config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"model: {n_params / 1e6:.2f}M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start = time.time()
for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss(model)
        print(
            f"iter {it:5d} | train loss {losses['train']:.4f} "
            f"| val loss {losses['val']:.4f} | {time.time() - start:.0f}s"
        )

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save everything sampling needs: weights, config and tokenizer metadata.
# Config and tokenizer are stored as plain dicts/lists so torch.load can
# restore them under weights_only=True on any recent torch version.
checkpoint = {
    "model": model.state_dict(),
    "config": {
        "vocab_size": vocab_size,
        "block_size": block_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "dropout": dropout,
        "bias": bias,
    },
    "tokenizer": {
        "chars": chars,
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
    },
}
torch.save(checkpoint, ckpt_path)
print(f"saved checkpoint to {ckpt_path}")

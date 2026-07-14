"""Character-level data pipeline: corpus, tokenizer, split and batching.

The vocabulary is built from the corpus itself (about 82 unique characters
for the cleaned Harry Potter text), never hardcoded. In the paper's terms a
"token" here is a single character, and the embedding table in model.py is
the learned vector for each of them (section 3.4, Embeddings and Softmax).
"""

from pathlib import Path

import torch

import config


def load_text(path: Path) -> str:
    """Read the whole corpus as one UTF-8 string."""
    return Path(path).read_text(encoding="utf-8")


def build_vocab(text: str):
    """Build the character vocabulary from the data.

    Returns (stoi, itos, vocab_size): string-to-int and int-to-string maps.
    Sorting makes the mapping deterministic across runs.
    """
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, len(chars)


def encode(s: str, stoi: dict) -> list[int]:
    """Characters to integer ids."""
    return [stoi[ch] for ch in s]


def decode(ids, itos: dict) -> str:
    """Integer ids back to characters. Accepts a list or a 1-D tensor."""
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return "".join(itos[i] for i in ids)


def train_val_split(data: torch.Tensor, frac: float = 0.9):
    """First 90% for training, final 10% for validation."""
    n = int(frac * len(data))
    return data[:n], data[n:]


def get_batch(split: str, train_data: torch.Tensor, val_data: torch.Tensor,
              block_size: int, batch_size: int, device: str):
    """Sample a random batch of contiguous character chunks.

    x is (batch_size, block_size) of inputs; y is the same chunk shifted one
    position right, so y[b, t] is the character that follows x[b, t]. That
    alignment is what turns "read text, guess next character" into a
    supervised target at every position.
    """
    data = train_data if split == "train" else val_data
    assert len(data) > block_size, (
        f"{split} split ({len(data):,} chars) must be longer than "
        f"block_size ({block_size})"
    )
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    # nn.Embedding and cross_entropy both need int64 ids, on the model's device.
    return x.to(device), y.to(device)


def load_dataset():
    """Convenience wrapper used by train.py: corpus -> vocab -> split."""
    text = load_text(config.DATA_PATH)
    stoi, itos, vocab_size = build_vocab(text)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    train_data, val_data = train_val_split(data)
    return stoi, itos, vocab_size, train_data, val_data


if __name__ == "__main__":
    # Self-test: run `python data.py` to check the pipeline end to end.
    text = load_text(config.DATA_PATH)
    stoi, itos, vocab_size = build_vocab(text)
    print(f"corpus: {config.DATA_PATH}")
    print(f"corpus length: {len(text):,} characters")
    print(f"vocab_size: {vocab_size}")
    print(f"vocab: {''.join(sorted(stoi))!r}")

    # Round trip through the tokenizer.
    sample = text[:50]
    assert decode(encode(sample, stoi), itos) == sample, "encode/decode mismatch"
    print(f"tokenizer round trip OK on: {sample!r}")

    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    train_data, val_data = train_val_split(data)
    assert len(train_data) + len(val_data) == len(text)
    print(f"train: {len(train_data):,}  val: {len(val_data):,} "
          f"(sum {len(train_data) + len(val_data):,})")

    x, y = get_batch("train", train_data, val_data,
                     config.block_size, config.batch_size, "cpu")
    print(f"batch shapes: x {tuple(x.shape)}  y {tuple(y.shape)}  dtype {x.dtype}")

    # Targets must be the inputs shifted one position right.
    assert torch.equal(x[0, 1:], y[0, :-1]), "y is not x shifted by one"
    print("alignment check: y[0][:-1] == x[0][1:] OK")
    print(f"x[0] starts: {decode(x[0][:60], itos)!r}")
    print(f"y[0] starts: {decode(y[0][:60], itos)!r}")
    print("data.py self-test passed")

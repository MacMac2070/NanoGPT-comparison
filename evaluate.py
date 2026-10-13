"""Score every trained model on the same held-out text, in bits per character.

Loss per token cannot be compared across tokenizers: a BPE token covers
several characters, so its loss is naturally higher. This script scores
each model on one shared piece of text (the final 10% of the corpus by
characters, which is the character model's validation split) and divides
the total loss by the number of characters, giving bits per character
(BPC), which is comparable across tokenizers.

Each model reads the text through overlapping windows of its full context
length. Only the second half of each window is scored, so every scored
token sees at least block_size / 2 tokens of context.

Usage, from the repository root:
    python evaluate.py                    # every model with a checkpoint
    python evaluate.py --models paper_gpt_1
    python evaluate.py --max-chars 50000  # quick check on part of the text

Results are printed and written to results/metrics.json.
"""

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "files" / "input.txt"
MODELS = ("paper_gpt_1", "paper_gpt_2", "paper_gpt_3")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_gpt_class(model_dir: Path):
    """Import GPT from one model folder's own model.py."""
    spec = importlib.util.spec_from_file_location(
        f"{model_dir.name}_model", model_dir / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(model_dir))  # model.py may import its own config
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        sys.modules.pop("config", None)
    return module.GPT


def make_codec(ckpt: dict):
    """Return encode(text) -> ids for the tokenizer saved in the checkpoint."""
    if "tokenizer_json" in ckpt:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_str(ckpt["tokenizer_json"])
        return lambda s: tokenizer.encode(s).ids, "BPE"
    stoi = ckpt["stoi"]
    return lambda s: [stoi[ch] for ch in s], "characters"


@torch.no_grad()
def total_nll(model, ids: torch.Tensor, block_size: int, device: str,
              batch_size: int = 32) -> tuple[float, int]:
    """Sum of negative log-likelihood (nats) over every predicted token.

    Windows of block_size + 1 ids step by stride = block_size // 2. The
    first window scores all its targets; later windows score only their last
    `stride` targets, so each target is scored exactly once.
    """
    stride = block_size // 2
    n = len(ids)
    starts = list(range(0, max(n - block_size - 1, 0) + 1, stride))
    if starts[-1] + block_size + 1 < n:
        starts.append(n - block_size - 1)

    nll_sum, scored, covered_to = 0.0, 0, 1  # target index 0 is never predicted
    for b in range(0, len(starts), batch_size):
        chunk = starts[b:b + batch_size]
        x = torch.stack([ids[s:s + block_size] for s in chunk]).to(device)
        y = torch.stack([ids[s + 1:s + block_size + 1] for s in chunk]).to(device)
        logits, _ = model(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                              y.reshape(-1), reduction="none").view(len(chunk), -1)
        for row, s in enumerate(chunk):
            # Target positions in this window are s+1 .. s+block_size.
            first_new = max(covered_to, s + 1) - (s + 1)
            if first_new < block_size:
                nll_sum += nll[row, first_new:].sum().item()
                scored += block_size - first_new
                covered_to = s + block_size + 1
    return nll_sum, scored


def evaluate(name: str, val_text: str, device: str) -> dict | None:
    model_dir = ROOT / name
    ckpt_path = model_dir / "checkpoints" / "ckpt.pt"
    if not ckpt_path.is_file():
        print(f"{name}: no checkpoint at {ckpt_path}, skipped")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = load_gpt_class(model_dir)(**ckpt["hyperparams"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    encode, tokens = make_codec(ckpt)
    ids = torch.tensor(encode(val_text), dtype=torch.long)
    block_size = ckpt["hyperparams"]["block_size"]

    t0 = time.time()
    nll, n_scored = total_nll(model, ids, block_size, device)
    n_params = sum(p.numel() for p in model.parameters())

    result = {
        "model": name,
        "tokens": tokens,
        "vocab_size": ckpt["hyperparams"]["vocab_size"],
        "parameters": n_params,
        "trained_iterations": ckpt.get("iteration"),
        "val_chars": len(val_text),
        "val_tokens": len(ids),
        "chars_per_token": round(len(val_text) / len(ids), 3),
        "loss_per_token": round(nll / n_scored, 4),
        # Normalise by characters covered by the scored tokens (all but the
        # first token, a negligible difference at this length).
        "bpc": round(nll / math.log(2) / (len(val_text) * n_scored / len(ids)), 4),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"{name:12s} | {tokens:10s} | params {n_params:,} | "
          f"loss/token {result['loss_per_token']:.4f} | BPC {result['bpc']:.4f} "
          f"| {result['seconds']}s")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--max-chars", type=int, default=None,
                        help="score only the first N characters of the val text")
    args = parser.parse_args()

    text = DATA_PATH.read_text(encoding="utf-8")
    val_text = text[int(0.9 * len(text)):]  # same split as paper_gpt_1/data.py
    if args.max_chars:
        val_text = val_text[:args.max_chars]
    device = pick_device()
    print(f"device: {device} | shared validation text: {len(val_text):,} characters\n")

    results = [r for r in (evaluate(m, val_text, device) for m in args.models) if r]
    out = ROOT / "results" / "metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwritten to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Generate text from a trained checkpoint.

Examples:
    python sample.py --num_tokens 500
    python sample.py --prompt "Harry looked at" --temperature 0.8 --top_k 40
"""

import argparse
import os

import torch

from model import GPT, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample from a trained character-level GPT"
    )
    default_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt.pt")
    parser.add_argument(
        "--ckpt", type=str, default=default_ckpt, help="path to the checkpoint file"
    )
    parser.add_argument(
        "--prompt", type=str, default="\n", help="starting text for generation"
    )
    parser.add_argument(
        "--num_tokens", type=int, default=500,
        help="number of characters to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="sampling temperature (> 0; lower is more predictable)",
    )
    parser.add_argument(
        "--top_k", type=int, default=None,
        help="if set, sample only from the k most likely characters (0 disables)",
    )
    parser.add_argument("--seed", type=int, default=1337, help="random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.temperature <= 0:
        raise SystemExit("--temperature must be > 0")
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    checkpoint = torch.load(args.ckpt, map_location=device)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    stoi = checkpoint["tokenizer"]["stoi"]
    itos = checkpoint["tokenizer"]["itos"]

    # An empty prompt still needs one token of context to condition on.
    prompt = args.prompt if args.prompt else "\n"
    unknown = sorted({ch for ch in prompt if ch not in stoi})
    if unknown:
        raise SystemExit(
            f"prompt contains characters not in the training vocabulary: {unknown!r}"
        )

    idx = torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=args.num_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print("".join(itos[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()

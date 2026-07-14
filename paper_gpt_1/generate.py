"""Sample text from a trained checkpoint.

Rebuilds the model from the hyperparameters and vocabulary stored inside
the checkpoint, so it keeps working even if config.py is edited after
training.

Usage:
    python generate.py                           # 500 chars from scratch
    python generate.py --prompt "Harry looked"   # continue a prompt
    python generate.py --max_new_tokens 2000 --ckpt checkpoints/ckpt.pt
"""

import argparse
from pathlib import Path

import torch

import config
import data
from model import GPT


def main():
    parser = argparse.ArgumentParser(description="Generate text from the paper GPT")
    parser.add_argument("--prompt", type=str, default="",
                        help="starting text (empty starts from a newline)")
    parser.add_argument("--max_new_tokens", type=int, default=500,
                        help="number of characters to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="sampling temperature: <1 sharper/safer, >1 more random")
    parser.add_argument("--top_k", type=int, default=20,
                        help="sample only from the k most likely characters (0 disables)")
    parser.add_argument("--ckpt", type=str,
                        default=str(config.CKPT_DIR / "ckpt.pt"),
                        help="path to the checkpoint file")
    args = parser.parse_args()

    if not Path(args.ckpt).is_file():
        raise SystemExit(
            f"no checkpoint found at {args.ckpt}; run train.py first"
        )
    ckpt = torch.load(args.ckpt, map_location=config.DEVICE, weights_only=True)
    stoi, itos = ckpt["stoi"], ckpt["itos"]

    model = GPT(**ckpt["hyperparams"]).to(config.DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()  # dropout off for generation

    # An empty prompt starts from a newline, a natural text boundary.
    prompt = args.prompt if args.prompt else "\n"
    unknown = sorted(set(prompt) - set(stoi))
    if unknown:
        raise SystemExit(
            f"prompt contains characters not in the training vocabulary: "
            f"{unknown!r}"
        )

    idx = torch.tensor([data.encode(prompt, stoi)], dtype=torch.long,
                       device=config.DEVICE)
    out = model.generate(idx, max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature,
                         top_k=args.top_k or None)
    print(data.decode(out[0], itos))


if __name__ == "__main__":
    main()

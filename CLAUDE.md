# nanoGPT build - project notes for Claude Code

## What this project is
Three small decoder-only GPTs written from scratch in PyTorch, trained on the
same cleaned Harry Potter `input.txt`. The goal is LEARNING, not just a working
model. The three do not share a source, and that distinction matters:

- `paper_gpt_1` and `paper_gpt_2` are written from "Attention Is All You Need"
  (Vaswani et al. 2017), following the base model of Table 3 as closely as a
  decoder-only setup on one book allows: d_model 512, N 6, h 8, d_ff 2048,
  P_drop 0.1, sinusoidal positions, ReLU, post-norm, Noam warmup. They share a
  byte-identical `model.py`. gpt_2 differs from gpt_1 only in using a byte-level
  BPE tokenizer with `bpe_vocab_size` set to 5000; gpt_1 is character-level
  (vocab ~82, built from the corpus, never hardcoded).
- `paper_gpt_3/char_gpt` is written from Karpathy's "Let's build GPT" video.
  The concepts come from the video and the code was typed out by hand rather
  than copied, so it is not line for line with his. It uses the post-2017
  conventions the video teaches: learned position embeddings, pre-norm blocks,
  GELU, and fused QKV. Where it touches the paper, it reached it through the
  video.

Never work from `karpathy/nanoGPT` or any other finished reference
implementation.

## Two modes - do not mix them up

### Learning mode (default)
The user writes the model themselves. Your job is to explain, review, and handle
plumbing - NOT to write the model or training loop. Lean on:
- `transformer-tutor`  - explain concepts / lines / why the video does something
- `error-explainer`    - explain errors and hint at fixes (never patch them)
- `my-code-reviewer`   - check the user's own code against the paper, no rewrites
- `plumbing-helper`    - data cleaning, plotting, checkpoints, setup only

In this mode, never write attention, the block, the GPT class, or the training
loop for the user. If asked to "just write it", explain and let them type it.

### One-shot mode (only when the user explicitly asks for a full build)
Build the whole thing autonomously, then harden it. Lean on:
- `nanogpt-architect`          - plan the files and build order first
- `transformer-code-reviewer`  - review the model before training (read-only)
- `train-smoke-tester`         - run a tiny config, confirm loss decreases
- `debugger`                   - fix runtime errors / NaN / flat loss

Use one-shot mode when the user wants a fast reference build to compare against
their own - then offer to have `transformer-tutor` walk through the differences.

## Defaults
- Assume learning mode unless the user says "one-shot" or "just build it".
- British spelling, no em dashes in prose.
- Keep the two sources straight. Do not describe gpt_3 as written from the
  paper, and do not describe gpt_1 or gpt_2 as following the video.

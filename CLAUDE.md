# nanoGPT build - project notes for Claude Code

## What this project is
Building a small decoder-only GPT from scratch in PyTorch, following Karpathy's
"Let's build GPT" video. The goal is LEARNING, not just a working model. First
train on tinyshakespeare to match Karpathy's loss numbers, then swap in a cleaned
Harry Potter `input.txt` (char-level, vocab ~82).

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

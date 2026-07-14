# Claude Code sub-agents for building nanoGPT

Two sets of Claude Code sub-agents for building a from-scratch nanoGPT:

- **`oneshot/`** - four agents that let Claude Code build the whole nanoGPT
  autonomously and harden it (plan -> review -> smoke-test -> debug).
- **`learning/`** - four agents for coding along with Karpathy's video yourself.
  The three tutoring agents are **read-only by design**, so they can explain and
  review but physically cannot write your model for you.

## How sub-agents work (quick version)
- They are Markdown files with YAML frontmatter in `.claude/agents/`.
- Claude routes to one automatically by matching your task to its `description`.
- The `tools:` line restricts what each agent can do (read-only = can't edit).
- Files load at session start - restart Claude Code after copying them in
  (or create/edit via the `/agents` command, which takes effect immediately).

## Install

Project scope (recommended - lives with this repo):

```bash
# from your nanoGPT project folder
mkdir -p .claude/agents
cp -R oneshot learning .claude/agents/      # both sets, switch by intent
cp CLAUDE.md .                              # optional: guides the main agent
```

Then restart Claude Code (or run `/agents` to confirm they loaded).

### Option A - keep both sets loaded (recommended)
Copy both `oneshot/` and `learning/` into `.claude/agents/`. Claude scans
subfolders recursively, so all eight load. Switch by intent: for learning, ask
the tutor/reviewer; for a full build, say "one-shot it". The included `CLAUDE.md`
tells the main agent to default to learning mode.

### Option B - one set at a time
If you'd rather not have both present, copy only the folder you want:

```bash
mkdir -p .claude/agents
cp learning/*.md .claude/agents/     # learning mode only
# or
cp oneshot/*.md  .claude/agents/     # one-shot mode only
```

User scope (available in every project) instead of project scope:
use `~/.claude/agents/` in place of `.claude/agents/`.

## Using them
- Claude will delegate automatically when a task matches an agent's description.
- You can also force one: e.g. "Use the my-code-reviewer subagent on my attention
  block", or "Have transformer-tutor explain why we scale by sqrt(d_k)".
- Run `/agents` any time to see what's loaded.

## The agents

### oneshot/ (autonomous build)
| agent | role | tools |
|---|---|---|
| nanogpt-architect | plan files + build order first | read-only |
| transformer-code-reviewer | check the model before training | read-only |
| train-smoke-tester | run a tiny config, confirm loss drops | Bash, Read |
| debugger | fix runtime errors / NaN / flat loss | can edit |

### learning/ (code along yourself)
| agent | role | tools |
|---|---|---|
| transformer-tutor | explain concepts / lines / the video | read-only |
| error-explainer | explain errors + hint (never patch) | read-only |
| my-code-reviewer | review YOUR code, no rewrites | read-only |
| plumbing-helper | data, plots, checkpoints, setup only | can edit (non-model) |

## Recommended flow
1. Learning mode: build nanoGPT yourself with the video, leaning on the
   `learning/` agents. Train tinyshakespeare first, then swap in Harry Potter.
2. Later: switch to one-shot mode for a fast reference build, then ask
   `transformer-tutor` to walk through how it differs from your hand-built one.

# nanoGPT from scratch

Three small language models trained on the same book text. Two of them were
written from the paper *Attention Is All You Need*. The third came from Andrej
Karpathy's *Let's build GPT* video. This is a learning project rather than a
library. The interesting part is what happens when you compare the three.

"nanoGPT" here means Andrej Karpathy's nanoGPT: the small character-level GPT
he builds in his *Let's build GPT* video and publishes at
[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT). These three are my own
versions of that idea, and they did not all come from the same place. gpt_1 and
gpt_2 were written from the 2017 paper, following it as closely as this setup
allowed. gpt_3 came from the video: I followed the concepts as he explains them
and typed the code out myself, so it is not line for line with his. Where gpt_3
touches the paper at all, it reached it through the video rather than the other
way round. I did not work from his published repository for any of the three.

The training text is the Harry Potter books, cleaned into one plain-text file.
It is copyrighted, so it is not included here.

## What happened

The three are scored on bits per character. If you used a model to compress a
piece of text it has never seen before, how many bits would it need for each
character? Lower is better. The measure works even when two models chop text up
differently, which turns out to matter a lot here.

paper_gpt_1 wins at 1.5971 bits per character. paper_gpt_3 finishes close
behind on roughly half the parameters. paper_gpt_2 comes last at 1.7564, which
is a surprise, because it is the largest of the three and the only one with a
proper tokenizer.

It comes last because of a mistake in how I set the runs up, and the mistake is
more interesting than the result. All three trained for 5,000 steps, so on paper
they got the same amount of training. They did not. gpt_2 breaks text into
subword chunks averaging about 3.9 characters, so every step feeds it nearly
four times as much text as the other two. Across 5,000 steps it read its
training data about 57 times over, while gpt_1 and gpt_3 read theirs about 14
times each. It stopped learning the language and started memorising the book,
and its scores say so plainly.

Which means this run cannot tell you whether subword tokenization beats
character tokenization. It was spoiled before it could answer that. What it does
show is that counting training steps is a poor way to compare models that split
text differently. Count passes over the text instead.

## Why three

gpt_1 and gpt_2 are the same model in every respect I could hold still: same
size, same shape, same learning schedule, same text, same number of steps. They
share a byte-identical `model.py`. The one thing that differs is how text
becomes numbers. gpt_1 reads a character at a time. gpt_2 reads subword chunks
from a byte-level BPE tokenizer trained on the corpus itself, with the target
vocabulary set to 5,000. That was the experiment.

One consequence of that single change is worth naming, because it hides behind
an identical number. Both set `block_size = 256`, but for gpt_1 that means 256
characters and for gpt_2 it means 256 tokens, which works out at about 1,004
characters. The same setting therefore hands gpt_2 roughly four times the
context.

gpt_3 came from the video instead of the paper. It is smaller, uses a simpler
training recipe, and follows the conventions that became standard after 2017
rather than the paper's own, so it is not a like-for-like replication of the
other two. It is here as a second route to the same architecture.

## Inside the models

All three are decoder-only transformers: a stack of blocks that each mix
information across positions using masked self-attention, then process every
position on its own with a small feed-forward network. Masked means a position
can only see what came before it, which is what lets the model write the next
character instead of filling in a blank.

gpt_1 and gpt_2 follow the 2017 paper closely. Position is stamped on with
fixed sine and cosine waves, the feed-forward network uses ReLU, attention is
divided by the square root of the head width, and each sub-layer normalises
after the residual add, in the paper's order LayerNorm(x + Sublayer(x)). The
embedding is multiplied by sqrt(512) on the way in, as written in section 3.4.

Where those two depart from the paper, and why:

- Decoder only. The paper's model translates, so it carries an encoder stack, a
  decoder stack, and cross-attention joining the two. These use the decoder half
  on its own, because the task here is continuing text rather than translating
  it.
- 82 characters rather than 37,000 byte pairs. Section 5.1 uses a shared BPE
  vocabulary of roughly 37k. gpt_1 uses the 82 distinct characters in the book,
  and gpt_2 uses BPE but stops at 5,000.
- 5,000 steps rather than 100,000. Each step also sees 16,384 characters against
  the paper's roughly 25,000 tokens, so the whole run is a small fraction of the
  paper's compute.
- The warmup constant is faithful, its effect is not. `warmup_steps` is set to
  the paper's 4,000, but the paper spends those over 100,000 steps, so warmup is
  4% of its training and 80% of mine. Copying the number across produced a
  schedule the paper would not recognise.
- No checkpoint averaging. The paper averages its last several checkpoints
  before evaluating. These save and score a single one.

gpt_3 uses the arrangement Karpathy teaches in the video, which is the one that
became common practice after 2017. Position comes from a learned table rather
than sinusoids, the feed-forward network uses GELU, and normalisation happens
before each sub-layer rather than after:

```
x = x + Attention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

Normalising first leaves an unbroken path from the output back to the input,
which makes deep stacks easier to train. gpt_3 also computes query, key and
value in one matrix multiply covering all heads at once, and shrinks the
residual projections at startup by 1/sqrt(2 x layers) so the signal running
through the stack does not grow with depth.

## The models

Configuration first, measured results below the label row. Everything measured
comes from the same held-out 10% of the text, averaged over 200 batches, using
the checkpoint saved at the end of training.

|                          | paper_gpt_1  | paper_gpt_2     | paper_gpt_3  |
|--------------------------|--------------|-----------------|--------------|
| Text split into          | characters   | subword chunks  | characters   |
| Vocabulary size          | 82           | 5,000           | 82           |
| Parameters               | 18,944,000   | 21,462,016      | 10,777,344   |
| Embedding / layers / heads | 512 / 6 / 8 | 512 / 6 / 8    | 384 / 6 / 6  |
| Feed-forward size        | 2048         | 2048            | 1536         |
| Dropout                  | 0.1          | 0.1             | 0.2          |
| Learning rate            | Noam warmup  | Noam warmup     | constant 3e-4|
| Context window (chars)   | 256          | about 1,004     | 256          |
| Steps / passes over text | 5,000 / 14.5 | 5,000 / 57      | 5,000 / 14.5 |
| Training time            | 4h 28m       | 6h 31m          | 1h 34m       |
| Measured results         |              |                 |              |
| Training loss            | 1.0062 /char | 1.3849 /chunk   | 1.0660 /char |
| Held-out loss            | 1.1070 /char | 4.7736 /chunk   | 1.1460 /char |
| Gap between the two      | +0.1008      | +3.3886         | +0.0800      |
| Perplexity               | 3.025        | 118.342         | 3.146        |
| **Bits per character**   | **1.5971**   | **1.7564**      | **1.6533**   |
| Blind-guess baseline     | 6.3576       | 3.1338          | 6.3576       |
| Bits saved vs baseline   | 4.7605       | 1.3774          | 4.7043       |

The blind-guess baseline row looks wrong at first glance, because gpt_2's
number sits at about half the other two. It is not wrong. That row is what a
model would score by guessing at random, and it differs across the three because
the tokenizer has already done part of the work. gpt_1 picks one of 82
characters, so a blind guess costs 6.3576 bits per character. gpt_2 picks one of
5,000 chunks covering about 3.9 characters each, which is a far easier guess,
worth 3.1338. The row underneath takes that head start away and shows what each
network learned on top of its tokenizer: about 4.7 bits for the two character
models, about 1.4 for gpt_2.

The parameter row makes gpt_2 look like the bigger network, and it is not. It
carries 2.5M more parameters than gpt_1, but all of them sit in the lookup table
that turns tokens into vectors, at exactly (5000 - 82) x 512. The transformer
underneath is identical in both at 18.90M. All three reuse that same table for
the final output layer, so predicting the next token costs nothing extra.

## What the text looks like

Full outputs are in [`samples/`](samples/). Each model was given "Harry looked
at" and asked to keep going, at matched settings and roughly matched length.

gpt_1 spells correctly and punctuates dialogue correctly, then loses the thread:

> "I've got the last trouble to me," said Ron, who was wearing the marble
> swigs on her package

gpt_2 reads much more like real prose:

> Harry and Ron made their way off along the platform, so they went down to a
> changing room, where they were heading, and onto the luggage rack

gpt_3, the smallest, invents words outright:

> an obstance of which were still the buttoned the headmaster, and sharp
> staring in her shoulders, which was an astonish

Ranked by how good they sound, that is the exact reverse of the scores. gpt_2
sounds best because it saw the same sentences 57 times and learned how they go.
That is not a second opinion about gpt_2. It is the same finding as its poor
score, seen from the other side, and it is why the ranking here comes from the
numbers rather than from reading the output.

One check worth recording: the source text is copyrighted and gpt_2 is visibly
memorising, so I went looking for passages copied word for word. The longest
exact match from any of the three is six words, from gpt_2: "mopping her eyes
with her sleeve." gpt_1 and gpt_3 top out at five. All of them are ordinary
English phrases rather than reproduced passages.

## Running it

You need Python 3.10 or newer and PyTorch. gpt_2 also needs the `tokenizers`
library. Each model folder carries its own `requirements.txt`, or you can cover
all three at once:

```
pip install torch tokenizers
```

The scripts choose a device on their own: an Nvidia GPU if one is present, then
Apple Silicon, then the CPU.

Run everything below from the top of the repo. The brackets keep each directory
change inside its own subshell, so you can paste several lines in a row without
ending up in the wrong folder.

### Supplying text

Put a plain-text file at `files/input.txt` for gpt_1 and gpt_2, and at
`paper_gpt_3/char_gpt/input.txt` for gpt_3. Mine is 6.29 MB, which after the
90/10 split leaves about 5.66M characters to train on. That figure is what the
passes-over-text row in the table is counted against. A few megabytes upward is
enough.

`files/clean_hp.py` shows the tidying I did: curly quotes converted to straight
ones, page-break characters removed, oversized first letters rejoined to the
words they belong to, and PDF line wrapping undone so each paragraph sits on one
line.

One thing to know before you swap the text out. gpt_2's tokenizer is not trained
with the model. It is built in a separate pass over the corpus before step 1, by
counting which pairs of symbols occur together most often and gluing the
commonest pair 4,744 times, and it is then frozen for the whole run. What the
model learns is the vector for each of those 5,000 pieces, not the pieces
themselves. The result is saved as `paper_gpt_2/tokenizer/bpe.json`, which ships
with this repo, and `data.py` loads that file whenever it exists. So if you
supply your own text, delete it first, or gpt_2 will train with a vocabulary
built from mine.

### Training

```
(cd paper_gpt_1 && python train.py)
(cd paper_gpt_2 && python train.py)
(cd paper_gpt_3/char_gpt && python train.py)
```

The times in the table are for an M-series MacBook Air. On a Mac,
`caffeinate -i python train.py` stops the machine sleeping partway through.

Checkpoints are written to `checkpoints/ckpt.pt` for gpt_1 and gpt_2, and
`ckpt.pt` for gpt_3. They are not in this repo, because two of them are 217 MB
and 246 MB and GitHub refuses files over 100 MB. Generating and scoring both
need a checkpoint, so train first.

### Generating text

```
(cd paper_gpt_1 && python generate.py --prompt "Harry looked at" --max_new_tokens 1200 --temperature 0.8 --top_k 20)
(cd paper_gpt_2 && python generate.py --prompt "Harry looked at" --max_new_tokens 300 --temperature 0.8 --top_k 20)
(cd paper_gpt_3/char_gpt && python sample.py --prompt "Harry looked at" --num_tokens 1200 --temperature 0.8 --top_k 20 --seed 1)
```

Those are the exact commands behind the files in `samples/`. gpt_2 is asked for
300 tokens rather than 1200 because its tokens cover about 3.9 characters each,
so 300 of them produce roughly as much text as 1200 characters.

The defaults are not the same across the three, which matters if you plan to
compare them. gpt_1 and gpt_2 default to temperature 0.8 with top_k 20. gpt_3
defaults to temperature 1.0 with no top_k filtering at all, so its output looks
noisier out of the box. That is a difference in how the text is sampled, not in
how good the model is. gpt_3 also fixes its random seed at 1337, so runs repeat
exactly unless you pass `--seed`.

### Scoring

```
(cd paper_gpt_1 && python eval_bpc.py)
(cd paper_gpt_2 && python eval_bpc.py)
(cd paper_gpt_3/char_gpt && python eval_bpc.py)
```

Loss on its own cannot compare these three. Loss is measured per token, and
gpt_2's tokens are about four characters long, so its numbers sit on a different
scale from the other two. Dividing by the characters each token covers puts them
back on one scale:

```
bits per character = loss per token / (characters per token x ln 2)
```

Lower is better, and the result means the same thing for all three models.

## Layout

```
paper_gpt_1/            character-level, follows the paper
  config.py             settings
  data.py               loading text, mapping characters to numbers
  model.py              the transformer
  train.py              training loop with Noam warmup
  generate.py           generating text
  eval_bpc.py           loss, perplexity, bits per character
  requirements.txt      torch

paper_gpt_2/            same architecture, subword tokenizer
  ...                   as above, plus:
  tokenizer/bpe.json    byte-level BPE, 5,000 vocabulary entries
  requirements.txt      torch, tokenizers

paper_gpt_3/char_gpt/   built from Karpathy's video, fused QKV
  model.py  train.py  sample.py  eval_bpc.py  requirements.txt

samples/                generated text from all three
files/clean_hp.py       text tidying
Hand-written notes/     seven pages: how words get their value, positional
                        encoding, multi-head attention, the feed-forward
                        network, the encoder and decoder stacks, and the
                        transformer as a whole

.claude/agents/         Claude Code sub-agents, see below
CLAUDE.md               project notes for the main Claude Code agent
LICENSE                 MIT
```

I built this alongside Claude Code, and the sub-agents I used are in
`.claude/agents/`. The four in `oneshot/` plan, review, smoke-test and debug a
build end to end. The four in `learning/` are for coding along yourself instead,
and three of them are read-only on purpose, so they can explain a concept or
review what you wrote but cannot write the model for you. Claude Code picks one
by matching your request against the description at the top of each file, and
loads them when a session starts.

## Caveats

All three scripts save the checkpoint from the final step rather than the best
one. For gpt_1 and gpt_3 that costs almost nothing, since their scores had
levelled off by the end. For gpt_2 it costs a great deal, because its best score
came and went long before step 5,000. Saving the best checkpoint, or simply
running about 1,250 steps to match the others on passes through the text, would
report a far better number than the table shows.

One thing the comparison does get right: all three use the same 90/10 split of a
byte-identical file, so the held-out text really is the same text in every case.

## References

- Vaswani et al., *Attention Is All You Need* (2017),
  https://arxiv.org/abs/1706.03762. The source for gpt_1 and gpt_2.
- Andrej Karpathy, *Let's build GPT*,
  https://www.youtube.com/watch?v=kCc8FmEb1nY. The source for gpt_3.
- Andrej Karpathy, nanoGPT, https://github.com/karpathy/nanoGPT. Listed because
  the name comes from here. I did not work from this code.

## License

MIT, see [LICENSE](LICENSE). It covers the code. The training text is not
included.

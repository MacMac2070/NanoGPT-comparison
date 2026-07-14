#!/usr/bin/env python3
"""
clean_hp.py - turn the raw text extracted from the Harry Potter PDF into a
tidy `input.txt` ready to train a character-level nanoGPT on.

WORKFLOW
--------
1) Extract raw text from the PDF first. Easiest is poppler's pdftotext:

       pdftotext "Harrypotter (ebook).pdf" hp_raw.txt

   (on a Mac:  brew install poppler   to get pdftotext)

   No poppler? Use pypdf instead (pip install pypdf):

       python -c "from pypdf import PdfReader; \
open('hp_raw.txt','w',encoding='utf-8').write(\
''.join(p.extract_text() for p in PdfReader('Harrypotter (ebook).pdf').pages))"

2) Then clean it:

       python clean_hp.py hp_raw.txt input.txt

WHAT IT DOES  (each step is a small, separate function you can read)
    - normalises smart quotes / ellipsis to plain ASCII        (optional)
    - removes page-break form-feed characters  (\\f)
    - rejoins split drop-cap letters at chapter starts   H\\n\\narry -> Harry
    - trims the leading contents list + dedication (up to "CHAPTER ONE")
    - un-wraps PDF hard-wrapping so each paragraph is one continuous line,
      with a single blank line between paragraphs
    - prints before/after size, the character vocabulary, and a sample
"""

import re
import sys

# ---- settings you can tweak -------------------------------------------------
NORMALISE_PUNCTUATION = True    # curly quotes / ellipsis -> plain ASCII
ASCII_DASHES          = False   # True also turns  and  into hyphens
TRIM_FRONT_MATTER     = True    # drop everything before the first "CHAPTER ONE"
REMOVE_BOILERPLATE    = True    # drop per-book copyright / credit / contents pages
# -----------------------------------------------------------------------------

# Paragraphs matching any of these are publisher boilerplate, not story text
# (they sit between the books: copyright pages, credits, contents lists).
BOILERPLATE = re.compile(
    r"©|Warner Bros|Pottermore|Publishing Rights|trademarks?\b|digital edition"
    r"|All rights reserved|ISBN|First published|Scholastic|Bloomsbury"
    r"|Arthur A\. Levine|GrandPr|Olly Moss|cover (?:art|illustration)"
    r"|Illustrations? by|www\.|http",
    re.IGNORECASE,
)


def is_boilerplate(paragraph: str) -> bool:
    """True if a paragraph is a publisher credit / copyright / contents block."""
    if paragraph.upper().startswith("CONTENTS"):
        return True
    return bool(BOILERPLATE.search(paragraph))


def normalise_punctuation(text: str) -> str:
    """Replace typographic 'smart' characters with plain ASCII equivalents."""
    replacements = {
        "\u2019": "'",    #  right single quote -> '
        "\u2018": "'",    #  left single quote  -> '
        "\u201c": '"',    #  left double quote  -> "
        "\u201d": '"',    #  right double quote -> "
        "\u2026": "...",  #  ellipsis           -> ...
        "\u00a0": " ",    # non-breaking space   -> normal space
    }
    if ASCII_DASHES:
        replacements["\u2014"] = "--"   #  em dash
        replacements["\u2013"] = "-"    #  en dash
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def remove_form_feeds(text: str) -> str:
    """PDF page breaks come through as form-feed (\\f) characters. Drop them."""
    return text.replace("\f", "\n")


def fix_drop_caps(text: str) -> str:
    """
    Chapter openings extract as a lone capital, a blank line, then the rest of
    the word:  "H\\n\\narry woke"  ->  "Harry woke"  (also "M\\n\\nr." -> "Mr.").
    A single-letter line followed by a blank line is only ever a drop cap.
    """
    return re.sub(r"(?m)^([A-Z])[ \t]*\n{2,}(?=[A-Za-z])", r"\1", text)


def trim_front_matter(text: str) -> str:
    """Cut the global contents list + dedication that precede the story."""
    m = re.search(r"CHAPTER ONE", text)
    return text[m.start():] if m else text


def unwrap_paragraphs(text: str) -> str:
    """
    The PDF hard-wraps lines every ~80 chars. Join the wrapped lines inside
    each paragraph back into one continuous line, keeping a single blank line
    between paragraphs (and leaving chapter headings on their own lines,
    since they are already separated by blank lines).
    """
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned = []
    for para in paragraphs:
        joined = " ".join(line.strip() for line in para.splitlines() if line.strip())
        if not joined:
            continue
        if REMOVE_BOILERPLATE and is_boilerplate(joined):
            continue
        cleaned.append(joined)
    return "\n\n".join(cleaned) + "\n"


def scrub_controls(text: str) -> str:
    """Remove any stray control characters, keeping normal whitespace."""
    return "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch >= " ")


def clean(raw: str) -> str:
    text = raw
    if NORMALISE_PUNCTUATION:
        text = normalise_punctuation(text)
    text = remove_form_feeds(text)
    text = fix_drop_caps(text)
    if TRIM_FRONT_MATTER:
        text = trim_front_matter(text)
    text = unwrap_paragraphs(text)
    text = scrub_controls(text)
    return text


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: python clean_hp.py <raw_input.txt> <clean_output.txt>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = clean(raw)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    # ---- report so you can sanity-check the result ----
    vocab = sorted(set(cleaned))
    printable_vocab = "".join(c for c in vocab if c.isprintable())
    print(f"raw chars:    {len(raw):>12,}")
    print(f"clean chars:  {len(cleaned):>12,}")
    print(f"words:        {len(cleaned.split()):>12,}")
    print(f"vocab size:   {len(vocab):>12} unique characters")
    print(f"vocabulary:   {printable_vocab}")
    print("-" * 68)
    print("sample (first 500 characters):\n")
    print(cleaned[:500])


if __name__ == "__main__":
    main()

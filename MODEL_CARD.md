---
license: mit
language: ar
tags:
  - arabic
  - poetry
  - gpt
  - nanogpt
datasets:
  - arbml/ashaar
---

# Arabic Poetry GPT — cells A and B

Two small decoder-only transformers trained from scratch on classical Arabic
poetry, as part of a study on tokenisation and model width. Full write-up,
methodology, and limitations: **[link to the GitHub repo]**.

These are **research artefacts, not products.** They are ~7.5M parameters
trained on a single GPU, and they produce metrically loose verse containing
invented words. See "Known failure modes" below and §8.5 of the write-up.

## The models

| | cell A | cell B |
|---|---|---|
| tokeniser | character | BPE, vocab 512 |
| tashkeel | stripped | stripped |
| n_embd / n_layer / n_head | 384 / 4 / 6 | 384 / 4 / 6 |
| block size | 512 | 512 |
| parameters | 7.31M | 7.49M |
| corpus | see write-up §8.2 | Ashaar, 162M chars |
| val bits/char | 2.2404 † | 2.2502 |

† Cell A was trained in an earlier session on a smaller corpus build and its
training configuration was **not recorded**. It is published for completeness
but should not be compared directly against cell B. This is explained in §8.2
of the write-up.

## Usage

```python
import json, torch
from safetensors.torch import load_file
from model import GPT
from tokenizer import ArabicBPE, CharTokenizer

cfg = json.load(open("gpt_B_plain_bpe_ashaar.json", encoding="utf-8"))
m = GPT(cfg["vocab_size"], cfg["n_embd"], cfg["n_head"],
        cfg["n_layer"], cfg["block_size"])
m.load_state_dict(load_file("gpt_B_plain_bpe_ashaar.safetensors"))
m.eval()

tok = (CharTokenizer("".join(cfg["tok_vocab"])) if cfg["kind"] == "char"
       else ArabicBPE(cfg["tok_vocab"], [tuple(x) for x in cfg["tok_merges"]]))
```

**Prompts must be normalised the same way the training text was.** Feeding a raw
string will silently corrupt it — سميرة loses its final ة and becomes سمير, a
different name. Use `normalize_prompt` from the repo, never the raw string.

Cell B accepts a conditioning header, `[metre|theme]`:

```
[الكامل|غزل]
وردة
```

## Known failure modes

- **Invented words.** Generates morphologically well-formed non-words,
  especially at the rhyme (المغوارق, الكفارق). It has learned the shape of a
  rhyme without the lexicon to fill it.
- **إيطاء.** Repeats the rhyme word across nearby lines, a fault in classical
  prosody.
- **Metre is not verified.** No prosodic checker was applied; hemistich lengths
  are visibly uneven.
- **Hard 512-token limit.** Past 512 tokens the model loses sight of the header
  and the line that established the rhyme, so long poems drift.
- Trained on a corpus in which words are sometimes split across the hemistich
  boundary; it reproduces that.

## Licence and attribution

Code MIT. Trained on [arbml/ashaar](https://huggingface.co/datasets/arbml/ashaar)
and [ARBML/MetRec](https://github.com/ARBML/MetRec) (MIT) — please cite ARBML.

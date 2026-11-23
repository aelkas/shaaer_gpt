"""Corpus build, de-duplication, poem-level split, tokenisation, caching.

Changes from the original pipeline:
  * exact-duplicate poems are removed (Ashaar is scraped from several sites);
  * the split is a seeded shuffle of POEMS (95/5), not "the last 10% in dataset
    order", so validation is an exchangeable sample of the same distribution;
  * val/train bait overlap is measured and reported;
  * token ids are cached as uint16 arrays, with a per-token character length
    table so bits-per-character is exact on any subset of tokens.
"""
import bz2
import gzip
import hashlib
import json
import lzma
import math
import os
import random
import re
import time
import unicodedata

import numpy as np

from .tok import ArabicBPE

TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
ARABIC = "ء-غف-يٱ"
ALLOWED = re.compile(rf"[^{ARABIC}ً-ْٰ ]")


def clean(s):
    s = unicodedata.normalize("NFKC", s).replace(TATWEEL, "")
    s = TASHKEEL.sub("", s)
    s = re.sub("[إأآٱ]", "ا", s)
    s = s.replace("ى", "ي").replace("ة", "ه")
    s = ALLOWED.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_prompt(s):
    """Prompts must go through the same normalisation as training text."""
    s = unicodedata.normalize("NFKC", s).replace(TATWEEL, "")
    s = TASHKEEL.sub("", s)
    s = re.sub("[إأآٱ]", "ا", s)
    return s.replace("ى", "ي").replace("ة", "ه")


def short_meter(m):
    return (m or "").replace("بحر ", "").strip() or "غير معروف"


def short_theme(t):
    return (t or "").replace("قصيدة ", "").strip() or "عامه"


def poems_from_ashaar(ds, min_baits=3, max_baits=60):
    """Yield (header, [bait, ...]) per poem. Hemistichs come from the dataset's
    own sadr/ajuz list, so a word split across ' # ' is the source's own
    segmentation (tadwir), not something this code introduces."""
    stats = dict(seen=0, odd=0, too_short=0, truncated=0)
    out = []
    for rec in ds:
        stats["seen"] += 1
        hemis = rec.get("poem verses") or []
        if len(hemis) % 2:
            stats["odd"] += 1
            hemis = hemis[:-1]
        baits = []
        for i in range(0, len(hemis), 2):
            a, b = clean(hemis[i]), clean(hemis[i + 1])
            if len(a.split()) < 2 or len(b.split()) < 2:
                continue
            baits.append(f"{a} # {b}")
        if len(baits) < min_baits:
            stats["too_short"] += 1
            continue
        if len(baits) > max_baits:
            baits = baits[:max_baits]
            stats["truncated"] += 1
        header = f"[{short_meter(rec.get('poem meter'))}|{short_theme(rec.get('poem theme'))}]"
        out.append((header, baits))
    return out, stats


def poem_text(header, baits):
    return header + "\n" + "\n".join(baits)


def compressor_baselines(text):
    raw = text.encode("utf-8")
    n = len(text)
    rows = {"uniform": math.log2(len(set(text)))}
    for name, mod in (("gzip", gzip), ("bzip2", bz2), ("lzma", lzma)):
        rows[name] = len(mod.compress(raw)) * 8 / n
    return rows


def build(out_dir, poems, vocab_size=512, val_frac=0.05, seed=0, log=print):
    """poems: list of (header, baits). Writes train.bin/val.bin (uint16),
    tokenizer.json, meta.json, train_poems.txt, val_poems.txt."""
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        log(f"data already built in {out_dir}")
        return json.load(open(meta_path))

    t0 = time.time()
    # exact-duplicate poems (same baits, ignoring header)
    seen, uniq = set(), []
    for h, b in poems:
        key = hashlib.md5("\n".join(b).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        uniq.append((h, b))
    n_dup = len(poems) - len(uniq)
    log(f"poems: {len(poems):,}  exact duplicates removed: {n_dup:,}")

    rng = random.Random(seed)
    rng.shuffle(uniq)
    n_val = int(len(uniq) * val_frac)
    val, train = uniq[:n_val], uniq[n_val:]

    train_baits = set(b for _, bs in train for b in bs)
    val_baits = [b for _, bs in val for b in bs]
    overlap_raw = sum(b in train_baits for b in val_baits) / max(1, len(val_baits))
    log(f"val baits also present verbatim in train: {100*overlap_raw:.2f}% (before cleaning)")
    # near-duplicates: a val poem sharing any line with train goes to train
    moved = 0
    for _ in range(5):
        keep, move = [], []
        for p in val:
            (move if any(b in train_baits for b in p[1]) else keep).append(p)
        if not move:
            break
        moved += len(move)
        train += move
        for _, bs in move:
            train_baits.update(bs)
        val = keep
    val_baits = [b for _, bs in val for b in bs]
    overlap = sum(b in train_baits for b in val_baits) / max(1, len(val_baits))
    log(f"moved {moved:,} near-duplicate poems from val to train; "
        f"val overlap now {100*overlap:.2f}% ({len(val):,} val poems)")

    train_text = "\n\n".join(poem_text(h, b) for h, b in train) + "\n"
    val_text = "\n\n".join(poem_text(h, b) for h, b in val) + "\n"
    open(os.path.join(out_dir, "train_poems.txt"), "w", encoding="utf-8").write(train_text)
    open(os.path.join(out_dir, "val_poems.txt"), "w", encoding="utf-8").write(val_text)

    log(f"training BPE-{vocab_size} on {len(train_text):,} chars ...")
    t1 = time.time()
    tok = ArabicBPE.train(train_text, vocab_size)
    log(f"  {len(tok.merges)} merges in {time.time()-t1:.0f}s")
    tok.save(os.path.join(out_dir, "tokenizer.json"))

    probe = val_text[:20000]
    assert tok.decode(tok.encode(probe)) == probe, "tokeniser round trip failed"

    ids = {}
    for name, txt in (("train", train_text), ("val", val_text)):
        t1 = time.time()
        a = np.array(tok.encode(txt), dtype=np.uint16)
        a.tofile(os.path.join(out_dir, f"{name}.bin"))
        ids[name] = a
        log(f"  {name}: {len(txt):,} chars -> {len(a):,} tokens "
            f"({len(txt)/len(a):.3f} chars/token, {time.time()-t1:.0f}s)")

    base = compressor_baselines(val_text)
    log("compressors on val text (bits/char): " +
        ", ".join(f"{k} {v:.3f}" for k, v in base.items()))

    headers = {}
    for h, _ in train:
        headers[h] = headers.get(h, 0) + 1
    top_headers = sorted(headers.items(), key=lambda kv: -kv[1])[:20]

    meta = dict(
        vocab_size=tok.vocab_size, n_merges=len(tok.merges),
        n_poems_raw=len(poems), n_duplicates=n_dup,
        n_train_poems=len(train), n_val_poems=len(val),
        train_chars=len(train_text), val_chars=len(val_text),
        train_tokens=int(len(ids["train"])), val_tokens=int(len(ids["val"])),
        val_bait_overlap=overlap, val_bait_overlap_raw=overlap_raw,
        n_moved_to_train=moved, compressors_val=base,
        top_headers=top_headers, seed=seed, val_frac=val_frac,
        build_seconds=time.time() - t0,
    )
    json.dump(meta, open(meta_path, "w"), ensure_ascii=False, indent=1)
    return meta


def load(out_dir):
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    tok = ArabicBPE.load(os.path.join(out_dir, "tokenizer.json"))
    train = np.fromfile(os.path.join(out_dir, "train.bin"), dtype=np.uint16)
    val = np.fromfile(os.path.join(out_dir, "val.bin"), dtype=np.uint16)
    return meta, tok, train, val

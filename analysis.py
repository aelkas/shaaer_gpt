"""What the heads do, and what the model writes.

head_study(model, ...) per (layer, head):
  * ablation: increase in loss when that head's output is zeroed, on all
    tokens and on rhyme tokens only (the last token of each line);
  * attention entropy;
  * rhyme attention: when predicting a line's last token, how much attention
    goes to the last tokens of the poem's previous lines (where the rhyme
    letter lives), relative to uniform attention.

generation_study(model, ...): sample whole poems from [metre|theme] headers and
measure rhyme consistency, repeated rhyme words (ita'), copying, invented words
and hemistich length against the same statistics on real held-out poems.
"""
import math
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from .data import normalize_prompt


def _line_end_positions(x, nl, bad_tokens):
    """For one window x (T,), return list of (q, keys): q predicts the last
    token of a bait line; keys are the last-token positions of previous lines
    of the same poem."""
    T = len(x)
    out = []
    ends = []          # line-final positions in current poem
    for e in range(1, T - 1):
        if x[e + 1] != nl or x[e] == nl:
            if x[e] == nl and e > 0 and x[e - 1] == nl:
                ends = []  # blank line: new poem
            continue
        if x[e] in bad_tokens:     # header line "[...]"
            ends = []
            continue
        q = e - 1
        if ends and q > ends[-1]:
            out.append((q, list(ends)))
        ends.append(e)
    return out


@torch.no_grad()
def head_study(model, data, tok, n_windows=128, batch=16):
    model.eval()
    c = model.cfg
    T = c.block
    nl = tok.stoi["\n"]
    bad = {i for i, t in enumerate(tok.vocab) if "]" in t or "[" in t}
    X, Y = data.val_windows(T, n_windows * T)
    X, Y = X[:n_windows], Y[:n_windows]
    Xc = X.cpu().numpy()
    pairs = [_line_end_positions(Xc[i], nl, bad) for i in range(len(Xc))]
    rhyme_mask = torch.zeros_like(Y, dtype=torch.bool)
    for i, ps in enumerate(pairs):
        for q, _ in ps:
            rhyme_mask[i, q] = True

    L, H = c.n_layer, c.n_head

    def losses(head_mask):
        tot = []
        for i in range(0, len(X), batch):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
                logits, _ = model(X[i:i + batch], head_mask=head_mask)
            l = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                Y[i:i + batch].reshape(-1), reduction="none")
            tot.append(l.view(-1, T))
        l = torch.cat(tot)
        return l.mean().item(), l[rhyme_mask].mean().item()

    ones = torch.ones(L, H, device=X.device)
    base_all, base_rhyme = losses(ones)
    abl_all = np.zeros((L, H))
    abl_rhyme = np.zeros((L, H))
    for l in range(L):
        for h in range(H):
            m = ones.clone()
            m[l, h] = 0
            a, r = losses(m)
            abl_all[l, h] = a - base_all
            abl_rhyme[l, h] = r - base_rhyme

    ent = np.zeros((L, H))
    rmass = np.zeros((L, H))
    rbase = 0.0
    n_ent = 0
    n_q = 0
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
            _, _, atts = model(xb, return_attn=True)
        for l, a in enumerate(atts):  # a: B,H,T,T float
            a = a.float()
            p = a[:, :, 64:, :]
            e = -(p.clamp_min(1e-12).log() * p).sum(-1)  # B,H,T-64
            ent[l] += e.mean(dim=(0, 2)).cpu().numpy() * len(xb)
        n_ent += len(xb)
        for j in range(len(xb)):
            for q, keys in pairs[i + j]:
                for l, a in enumerate(atts):
                    rmass[l] += a[j, :, q, keys].float().sum(-1).cpu().numpy()
                rbase += len(keys) / (q + 1)
                n_q += 1
    ent /= max(1, n_ent)
    rmass /= max(1, n_q)
    rbase /= max(1, n_q)
    model.train()
    return dict(
        base_loss=base_all, base_rhyme_loss=base_rhyme,
        ablation_all=abl_all.tolist(), ablation_rhyme=abl_rhyme.tolist(),
        entropy=ent.tolist(), rhyme_mass=rmass.tolist(),
        rhyme_mass_uniform=rbase, n_rhyme_queries=n_q,
    )


# ------------------------------------------------------------------ generation

def _poem_lines(text):
    """Lines of the first poem after the header, complete lines only."""
    body = text.split("\n", 1)[1] if "\n" in text else ""
    if "\n\n" in body:
        body = body.split("\n\n", 1)[0]
        lines = body.split("\n")
    else:
        lines = body.split("\n")[:-1]  # last line may be cut off
    return [l.strip() for l in lines if l.strip()]


def poem_metrics(lines, train_baits=None, train_words=None):
    good = [l for l in lines if l.count(" # ") == 1]
    m = dict(n_lines=len(lines),
             well_formed=len(good) / max(1, len(lines)))
    if len(good) < 2:
        return m
    last_words = [l.split()[-1] for l in good]
    finals = [w[-1] for w in last_words]
    m["rhyme_consistency"] = Counter(finals).most_common(1)[0][1] / len(finals)
    m["repeated_rhyme_word"] = float(len(set(last_words)) < len(last_words))
    if train_baits is not None:
        m["copied"] = sum(hash(l) in train_baits for l in good) / len(good)
    if train_words is not None:
        words = [w for l in good for w in l.replace("#", " ").split()]
        m["invented_words"] = sum(w not in train_words for w in words) / max(1, len(words))
        m["invented_rhyme_words"] = sum(w not in train_words for w in last_words) / len(last_words)
    hl = [len(h.strip()) for l in good for h in l.split(" # ")]
    m["hemistich_chars"] = float(np.mean(hl))
    m["hemistich_chars_sd_within"] = float(np.std(hl))
    return m


def corpus_sets(train_text):
    baits, words = set(), set()
    for line in train_text.split("\n"):
        if " # " in line:
            baits.add(hash(line.strip()))
            words.update(line.replace("#", " ").split())
    return baits, words


def real_poems_by_header(val_text, headers, k=200, seed=0):
    rng = random.Random(seed)
    by = {h: [] for h in headers}
    for block in val_text.split("\n\n"):
        lines = block.strip().split("\n")
        if lines and lines[0] in by:
            by[lines[0]].append(lines[1:])
    return {h: rng.sample(v, min(k, len(v))) for h, v in by.items()}


def _agg(ms):
    keys = sorted({k for m in ms for k in m})
    out = {}
    for k in keys:
        v = [m[k] for m in ms if k in m]
        out[k] = float(np.mean(v)) if v else float("nan")
    out["n_poems"] = len(ms)
    return out


@torch.no_grad()
def generation_study(model, tok, headers, train_sets, n_per_header=48, n_lines=8,
                     temperature=1.0, top_k=20, max_new=400, seed=0):
    model.eval()
    torch.manual_seed(seed)
    dev = next(model.parameters()).device
    nl = tok.stoi["\n"]
    train_baits, train_words = train_sets
    res, samples = {}, {}
    for h in headers:
        p = tok.encode(normalize_prompt(h) + "\n")
        idx = torch.tensor([p] * n_per_header, device=dev)
        plen = len(p)

        def stop(i):
            return bool(((i[:, plen:] == nl).sum(1) >= n_lines + 1).all())

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = model.generate(idx, max_new, temperature, top_k, stop)
        texts = [tok.decode(r.tolist()) for r in out]
        ms = [poem_metrics(_poem_lines(t)[:n_lines], train_baits, train_words) for t in texts]
        res[h] = _agg(ms)
        samples[h] = texts[:4]
    model.train()
    overall = _agg([dict(v) for v in res.values()])
    return dict(per_header=res, overall=overall, samples=samples,
                settings=dict(temperature=temperature, top_k=top_k, n=n_per_header))


def real_reference(val_text, headers, train_sets, k=200):
    real = real_poems_by_header(val_text, headers, k)
    res = {}
    for h, poems in real.items():
        ms = [poem_metrics(p[:8], *train_sets) for p in poems]
        res[h] = _agg(ms)
    return dict(per_header=res, overall=_agg([dict(v) for v in res.values()]))

"""BPE tokeniser for Arabic poetry (from-scratch, Sennrich-style).

Same algorithm as the original notebook: diacritics are glued to the letter
they follow before BPE (a no-op on tashkeel-stripped text), merges never cross
a space, and the tokeniser is fitted on the training split only.

Changes from the notebook: encoding caches per word (the corpus has ~30M word
occurrences but far fewer distinct words), and the tokeniser saves to JSON.
"""
import json
import re
from collections import Counter

COMBINING = "ؐ-ًؚ-ٰٟۖ-ۭ"
CLUSTER_RE = re.compile(rf"[^{COMBINING}][{COMBINING}]*|[{COMBINING}]+")


def to_clusters(s):
    return CLUSTER_RE.findall(s)


class ArabicBPE:
    def __init__(self, vocab, merges):
        self.vocab = list(vocab)
        self.merges = [tuple(m) for m in merges]
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.ranks = {p: i for i, p in enumerate(self.merges)}
        self._cache = {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    @classmethod
    def train(cls, text, vocab_size=512, verbose=False):
        words = Counter(text.split())
        splits = {w: tuple(to_clusters(w)) for w in words}
        alpha = sorted({c for sp in splits.values() for c in sp})
        vocab = ["\n", " "] + [c for c in alpha if c not in ("\n", " ")]
        merges = []
        pairs, where = Counter(), {}
        for w, f in words.items():
            sp = splits[w]
            for i in range(len(sp) - 1):
                p = (sp[i], sp[i + 1])
                pairs[p] += f
                where.setdefault(p, set()).add(w)
        while len(vocab) < vocab_size and pairs:
            best, cnt = max(pairs.items(), key=lambda kv: kv[1])
            if cnt < 2:
                break
            a, b = best
            new = a + b
            merges.append(best)
            vocab.append(new)
            for w in list(where.get(best, ())):
                sp, f = splits[w], words[w]
                for i in range(len(sp) - 1):
                    p = (sp[i], sp[i + 1])
                    pairs[p] -= f
                    if pairs[p] <= 0:
                        pairs.pop(p, None)
                        where.pop(p, None)
                    elif p in where:
                        where[p].discard(w)
                out, i = [], 0
                while i < len(sp):
                    if i < len(sp) - 1 and sp[i] == a and sp[i + 1] == b:
                        out.append(new)
                        i += 2
                    else:
                        out.append(sp[i])
                        i += 1
                sp = tuple(out)
                splits[w] = sp
                for i in range(len(sp) - 1):
                    p = (sp[i], sp[i + 1])
                    pairs[p] += f
                    where.setdefault(p, set()).add(w)
            pairs.pop(best, None)
            where.pop(best, None)
            if verbose and len(merges) % 100 == 0:
                print(f"  {len(merges)} merges", flush=True)
        return cls(vocab, merges)

    def _word(self, w):
        hit = self._cache.get(w)
        if hit is not None:
            return hit
        sp = to_clusters(w)
        while len(sp) > 1:
            best, br = None, None
            for i in range(len(sp) - 1):
                r = self.ranks.get((sp[i], sp[i + 1]))
                if r is not None and (br is None or r < br):
                    best, br = i, r
            if best is None:
                break
            sp = sp[:best] + [sp[best] + sp[best + 1]] + sp[best + 2:]
        ids = []
        for tok in sp:
            if tok in self.stoi:
                ids.append(self.stoi[tok])
            else:  # unseen cluster: fall back to codepoints, drop unknowns
                ids.extend(self.stoi[ch] for ch in tok if ch in self.stoi)
        self._cache[w] = ids
        return ids

    def encode(self, text):
        nl, sp = self.stoi["\n"], self.stoi[" "]
        ids = []
        for j, line in enumerate(text.split("\n")):
            if j:
                ids.append(nl)
            for k, w in enumerate(line.split(" ")):
                if k:
                    ids.append(sp)
                if w:
                    ids.extend(self._word(w))
        return ids

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    def token_char_lengths(self):
        return [len(t) for t in self.vocab]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "merges": self.merges}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["vocab"], d["merges"])

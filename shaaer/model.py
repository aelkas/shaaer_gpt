"""Configurable decoder-only transformer.

Attention variants
  attn="mha"       standard multi-head attention. Each head has its own W_Q, W_K
                   (d -> qk_dim) and W_V (d -> head_dim). Scores x_t W_Q W_K^T x_s
                   use a bilinear form of rank <= qk_dim.
  attn="bilinear"  W_Q W_K^T replaced by ONE learnable d x d matrix M_h per head:
                   score = (x_t M_h) . x_s. Full rank, d^2 parameters per head
                   instead of 2*d*qk_dim. Values still use W_V (d -> head_dim).

Other switches: pos in {learned, rope, none}, mlp in {relu, gelu, swiglu},
init in {old, scaled}, bias on/off, dropout. "old" reproduces the original
notebook (N(0, 0.02) everywhere, biases, lm_head bias).
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab: int = 512
    block: int = 512
    n_layer: int = 4
    n_embd: int = 384
    n_head: int = 6
    head_dim: int = 0      # 0 -> n_embd // n_head
    qk_dim: int = 0        # 0 -> head_dim (mha only)
    attn: str = "mha"      # mha | bilinear
    pos: str = "rope"      # learned | rope | none
    mlp: str = "gelu"      # relu | gelu | swiglu
    init: str = "scaled"   # old | scaled
    bias: bool = False
    dropout: float = 0.0

    def resolved(self):
        c = GPTConfig(**asdict(self))
        if not c.head_dim:
            c.head_dim = c.n_embd // c.n_head
        if not c.qk_dim:
            c.qk_dim = c.head_dim
        return c


def rope_cache(T, dim, base=10000.0, device=None):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(T, device=device).float()
    f = torch.outer(t, inv)  # T, dim/2
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    # x: B, H, T, D ; rotate pairs (even, odd)
    T = x.shape[-2]
    c, s = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    y1 = x1 * c - x2 * s
    y2 = x1 * s + x2 * c
    return torch.stack((y1, y2), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        d, h = c.n_embd, c.n_head
        if c.attn == "mha":
            self.qkv = nn.Linear(d, h * (2 * c.qk_dim + c.head_dim), bias=c.bias)
            self.qk_dim = c.qk_dim
        elif c.attn == "bilinear":
            self.q_bil = nn.Linear(d, h * d, bias=False)   # rows of M_h
            self.v = nn.Linear(d, h * c.head_dim, bias=c.bias)
            self.qk_dim = d
        else:
            raise ValueError(c.attn)
        self.proj = nn.Linear(h * c.head_dim, d, bias=c.bias)
        self.proj.is_residual_out = True
        self.drop = nn.Dropout(c.dropout)
        if c.pos == "rope":
            cos, sin = rope_cache(c.block, self.qk_dim)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def qkv_heads(self, x):
        B, T, d = x.shape
        c, h = self.c, self.c.n_head
        if c.attn == "mha":
            qkv = self.qkv(x)
            q, k, v = qkv.split([h * c.qk_dim, h * c.qk_dim, h * c.head_dim], dim=-1)
            q = q.view(B, T, h, c.qk_dim).transpose(1, 2)
            k = k.view(B, T, h, c.qk_dim).transpose(1, 2)
        else:
            q = self.q_bil(x).view(B, T, h, d).transpose(1, 2)
            k = x.unsqueeze(1).expand(B, h, T, d)
            v = self.v(x)
        v = v.view(B, T, h, c.head_dim).transpose(1, 2)
        if c.pos == "rope":
            q = apply_rope(q, self.cos, self.sin)
            k = apply_rope(k, self.cos, self.sin)
        return q, k, v

    def forward(self, x, head_mask=None, return_attn=False):
        B, T, _ = x.shape
        q, k, v = self.qkv_heads(x)
        if return_attn or head_mask is not None:
            # explicit path, used only for analysis
            att = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~mask, float("-inf")).softmax(-1)
            y = att.to(v.dtype) @ v
            if head_mask is not None:
                y = y * head_mask.view(1, -1, 1, 1).to(y.dtype)
        else:
            att = None
            y = F.scaled_dot_product_attention(
                q, k.contiguous(), v, is_causal=True,
                dropout_p=self.c.dropout if self.training else 0.0)
        y = y.transpose(1, 2).reshape(B, T, -1)
        y = self.drop(self.proj(y))
        return (y, att) if return_attn else y


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c.n_embd
        self.kind = c.mlp
        if c.mlp == "swiglu":
            hdim = int(round(8 * d / 3 / 64)) * 64  # parameter-matched to 4d
            self.w1 = nn.Linear(d, hdim, bias=c.bias)
            self.w3 = nn.Linear(d, hdim, bias=c.bias)
            self.w2 = nn.Linear(hdim, d, bias=c.bias)
            self.w2.is_residual_out = True
        else:
            self.fc = nn.Linear(d, 4 * d, bias=c.bias)
            self.out = nn.Linear(4 * d, d, bias=c.bias)
            self.out.is_residual_out = True
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        if self.kind == "swiglu":
            return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        act = F.relu if self.kind == "relu" else F.gelu
        return self.drop(self.out(act(self.fc(x))))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd, bias=c.bias)
        self.attn = Attention(c)
        self.ln2 = nn.LayerNorm(c.n_embd, bias=c.bias)
        self.mlp = MLP(c)

    def forward(self, x, head_mask=None, return_attn=False):
        if return_attn:
            a, att = self.attn(self.ln1(x), head_mask, True)
            x = x + a
            return x + self.mlp(self.ln2(x)), att
        x = x + self.attn(self.ln1(x), head_mask)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        c = cfg.resolved()
        self.cfg = c
        self.tok = nn.Embedding(c.vocab, c.n_embd)
        self.pos = nn.Embedding(c.block, c.n_embd) if c.pos == "learned" else None
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd, bias=c.bias)
        self.lm_head = nn.Linear(c.n_embd, c.vocab, bias=(c.init == "old"))
        self.lm_head.weight = self.tok.weight  # tied
        self.apply(self._init)
        if c.init == "scaled":
            for m in self.modules():
                if getattr(m, "is_residual_out", False):
                    nn.init.normal_(m.weight, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok.weight.numel()
            if self.pos is not None:
                n -= self.pos.weight.numel()
        return n

    def flops_per_token(self):
        c = self.cfg
        n = self.n_params(non_embedding=True) + c.vocab * c.n_embd  # lm_head matmul
        attn = c.n_layer * c.block * c.n_head * (c.qk_dim if c.attn == "mha" else c.n_embd)
        return 6 * n + 6 * attn  # fwd+bwd; attention term averaged over causal window

    def forward(self, idx, targets=None, head_mask=None, return_attn=False,
                reduction="mean"):
        B, T = idx.shape
        x = self.tok(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(T, device=idx.device))
        x = self.drop(x)
        atts = []
        for i, blk in enumerate(self.blocks):
            hm = None if head_mask is None else head_mask[i]
            if return_attn:
                x, a = blk(x, hm, True)
                atts.append(a)
            else:
                x = blk(x, hm)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                                   targets.reshape(-1), reduction=reduction)
        if return_attn:
            return logits, loss, atts
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=1.0, top_k=None, stop_fn=None):
        for _ in range(max_new):
            ctx = idx[:, -self.cfg.block:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :].float() / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(logits.softmax(-1), 1)
            idx = torch.cat([idx, nxt], 1)
            if stop_fn is not None and stop_fn(idx):
                break
        return idx

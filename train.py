"""One training run: data on GPU, bf16 autocast, optional torch.compile,
AdamW, warmup + {constant, cosine, wsd} schedule, exact bits-per-character
evaluation on a fixed validation set. Returns a result dict and (optionally)
saves a checkpoint that records its full configuration."""
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from .model import GPT, GPTConfig

LN2 = math.log(2)


def lr_mult(step, total, sched, warmup, decay_frac=0.2, min_ratio=0.1):
    if warmup and step < warmup:
        return (step + 1) / warmup
    if sched == "constant":
        return 1.0
    if sched == "cosine":
        p = (step - warmup) / max(1, total - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    if sched in ("wsd", "wsd_sqrt"):
        start = int(total * (1 - decay_frac))
        if step < start:
            return 1.0
        p = (step - start) / max(1, total - start)
        return (1 - p) if sched == "wsd" else (1 - math.sqrt(p))
    raise ValueError(sched)


class Data:
    """Token arrays on the GPU plus a per-token character-length table."""

    def __init__(self, train, val, char_len, device):
        self.train = torch.from_numpy(train.astype(np.int64)).to(device)
        self.val = torch.from_numpy(val.astype(np.int64)).to(device)
        self.char_len = torch.tensor(char_len, dtype=torch.float64, device=device)
        self.device = device

    def batch(self, B, T, gen):
        ix = torch.randint(len(self.train) - T - 1, (B,), device=self.device, generator=gen)
        idx = ix[:, None] + torch.arange(T + 1, device=self.device)
        w = self.train[idx]
        return w[:, :-1].contiguous(), w[:, 1:].contiguous()

    def val_windows(self, T, max_tokens):
        n = min(len(self.val) - 1, max_tokens) // T
        w = self.val[: n * T + 1]
        x = w[:-1].view(n, T)
        y = w[1:].view(n, T)
        return x, y


@torch.no_grad()
def evaluate(model, data, T, max_tokens, batch=64):
    """Exact bits per character over non-overlapping windows of the val split."""
    model.eval()
    X, Y = data.val_windows(T, max_tokens)
    nats, chars = 0.0, 0.0
    for i in range(0, len(X), batch):
        x, y = X[i:i + batch], Y[i:i + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            _, loss = model(x, y, reduction="sum")
        nats += loss.item()
        chars += data.char_len[y].sum().item()
    model.train()
    return nats / chars / LN2


def make_optimizer(model, lr, recipe):
    if recipe == "old":
        return torch.optim.AdamW(model.parameters(), lr=lr)  # betas .9/.999, wd .01, all params
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": 0.1},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8,
                             fused=torch.cuda.is_available())


def train_run(name, model_cfg: GPTConfig, data, tokens, lr, sched="wsd",
              recipe="new", warmup_frac=0.05, decay_frac=0.2, batch_tokens=131072,
              micro_batch=256, seed=0, eval_every_frac=0.1, eval_tokens=1_000_000,
              final_eval_tokens=8_000_000, compile=True, out_dir=None,
              save_ckpt=False, log=print, extra=None):
    device = data.device
    if compile:
        torch._dynamo.reset()  # each run compiles its own graph; avoid cache-limit fallbacks
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    T = model_cfg.block
    model = GPT(model_cfg).to(device)
    cfg = model.cfg
    n_params = model.n_params()
    n_nonemb = model.n_params(non_embedding=True)
    fpt = model.flops_per_token()

    micro = min(micro_batch, batch_tokens // T)
    accum = max(1, (batch_tokens // T) // micro)
    B_eff = micro * accum
    steps = max(1, tokens // (B_eff * T))
    warmup = 0 if recipe == "old" else max(50, int(warmup_frac * steps))
    clip = None if recipe == "old" else 1.0

    opt = make_optimizer(model, lr, recipe)
    fwd = torch.compile(model) if (compile and device.startswith("cuda")) else model

    hist = []
    every = max(1, int(steps * eval_every_frac))
    t0 = time.time()
    tok_seen = 0
    run_loss = None
    log(f"[{name}] params {n_params/1e6:.2f}M (non-emb {n_nonemb/1e6:.2f}M) "
        f"steps {steps} x {B_eff*T:,} tok, micro {micro}x{accum}, lr {lr:.2e} {sched}")
    for step in range(steps):
        m = lr_mult(step, steps, sched, warmup, decay_frac)
        for g in opt.param_groups:
            g["lr"] = lr * m
        for _ in range(accum):
            x, y = data.batch(micro, T, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                _, loss = fwd(x, y)
            (loss / accum).backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tok_seen += B_eff * T
        lv = loss.item() if (step % 50 == 0 or step == steps - 1) else None
        if lv is not None:
            if not math.isfinite(lv):
                log(f"[{name}] diverged at step {step}")
                hist.append(dict(step=step, tokens=tok_seen, train_loss=lv, diverged=True))
                break
            run_loss = lv if run_loss is None else 0.8 * run_loss + 0.2 * lv
        if (step + 1) % every == 0 and step + 1 < steps:
            vb = evaluate(model, data, T, eval_tokens)
            el = time.time() - t0
            hist.append(dict(step=step + 1, tokens=tok_seen, val_bpc=vb,
                             train_loss=run_loss, lr=lr * m, seconds=el))
            log(f"[{name}] {100*(step+1)/steps:4.0f}%  val {vb:.4f} bpc  "
                f"{tok_seen/el/1e6:.2f}M tok/s  {el/60:.1f}m")
    train_seconds = time.time() - t0
    diverged = bool(hist and hist[-1].get("diverged"))
    final = float("nan") if diverged else evaluate(model, data, T, final_eval_tokens)
    hist.append(dict(step=steps, tokens=tok_seen, val_bpc=final, train_loss=run_loss,
                     seconds=train_seconds))
    res = dict(
        name=name, model=asdict(cfg), tokens=tok_seen, lr=lr, sched=sched,
        recipe=recipe, warmup=warmup, decay_frac=decay_frac, batch_tokens=B_eff * T,
        seed=seed, params=n_params, nonemb_params=n_nonemb, flops_per_token=fpt,
        flops=fpt * tok_seen, epochs=tok_seen / len(data.train), final_bpc=final,
        diverged=diverged, seconds=train_seconds, tok_per_s=tok_seen / train_seconds,
        history=hist, extra=extra or {},
    )
    log(f"[{name}] FINAL {final:.4f} bpc  ({train_seconds/60:.1f} min, "
        f"{res['tok_per_s']/1e6:.2f}M tok/s, {res['epochs']:.2f} epochs)")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        json.dump(res, open(os.path.join(out_dir, f"{name}.json"), "w"), indent=1)
        if save_ckpt:
            torch.save(dict(state_dict=model.state_dict(), cfg=asdict(cfg), result=res),
                       os.path.join(out_dir, f"{name}.pt"))
    del fwd, opt
    return res, model

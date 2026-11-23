"""Follow-up analyses, run after the main suite on its saved checkpoints.

E1  level-k conditionals: for contexts of k previous tokens that occur often in
    held-out text, compare the model's implied P(next | context) (its full-context
    predictions averaged over every occurrence of the context) with the empirical
    P(next | context). Model predictions are averaged on one half of the val
    windows and compared with counts from the other half; the real-vs-real floor
    compares counts from the two halves. Same pass: loss vs position in the
    window, and calibration (top-1 confidence vs accuracy).
E2  distillation: the 7M base model trained on a bigger teacher's full predicted
    distributions instead of one-hot next tokens (pure, and 50/50 with data).
E3  bilinear W_QK vs standard attention at widths 192 and 768.
E4  generation metrics at temperature 0.7 (robustness of the T=1 results).
"""
import json
import math
import os
import time
import traceback
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from . import analysis, plots
from .model import GPT, GPTConfig
from .train import evaluate, lr_mult, make_optimizer

LN2 = math.log(2)


def _js(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


@torch.no_grad()
def level_k_study(model, data, ks=(0, 1, 2, 3, 4), min_count=60, max_ctx=20000, batch=32):
    model.eval()
    T = model.cfg.block
    V = model.cfg.vocab
    X, Y = data.val_windows(T, 10**9)
    Xn, Yn = X.cpu().numpy(), Y.cpu().numpy()
    n = len(Xn)
    half = np.arange(n) % 2  # window parity: A = 0, B = 1

    def ctx_keys(k):
        # key of the k tokens before target position t is x[t-k+1..t]; valid for t >= k-1
        if k == 0:
            return np.zeros_like(Xn, dtype=np.int64), np.ones_like(Xn, dtype=bool)
        key = np.zeros_like(Xn, dtype=np.int64)
        valid = np.zeros_like(Xn, dtype=bool)
        valid[:, k - 1:] = True
        for j in range(k):
            sh = np.zeros_like(Xn)
            sh[:, k - 1:] = Xn[:, k - 1 - j: T - j]
            key = key * V + sh
        return key, valid

    levels = {}
    for k in ks:
        key, valid = ctx_keys(k)
        uk, uc = np.unique(key[valid], return_counts=True)
        order = np.argsort(-uc)[:max_ctx]
        keep = np.sort(uk[order][uc[order] >= min_count])
        ci = np.full(key.shape, -1, dtype=np.int64)
        if len(keep):
            pos = np.searchsorted(keep, key)
            posc = np.minimum(pos, len(keep) - 1)
            hit = valid & (keep[posc] == key)
            ci[hit] = posc[hit]
        emp = np.zeros((2, len(keep), V))
        for h in (0, 1):
            m = (ci >= 0) & (half[:, None] == h)
            np.add.at(emp[h], (ci[m], Yn[m]), 1)
        levels[k] = dict(ci=torch.from_numpy(ci).to(X.device), emp=emp, n_ctx=len(keep),
                         qsum=torch.zeros(2, len(keep), V, dtype=torch.float64, device=X.device))

    pos_loss = torch.zeros(T, dtype=torch.float64, device=X.device)
    nb = 10
    conf_sum = torch.zeros(nb, dtype=torch.float64, device=X.device)
    acc_sum = torch.zeros(nb, dtype=torch.float64, device=X.device)
    cnt_bin = torch.zeros(nb, dtype=torch.float64, device=X.device)
    half_t = torch.from_numpy(half).to(X.device)
    for i in range(0, n, batch):
        x, y = X[i:i + batch], Y[i:i + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            logits, _ = model(x)
        q = logits.float().softmax(-1)  # B,T,V
        lp = torch.log(q.gather(-1, y[..., None]).squeeze(-1).clamp_min(1e-12))
        pos_loss += (-lp).double().sum(0)
        conf, pred = q.max(-1)
        b = (conf * nb).clamp(max=nb - 1e-6).long()
        conf_sum.index_add_(0, b.reshape(-1), conf.reshape(-1).double())
        acc_sum.index_add_(0, b.reshape(-1), (pred == y).reshape(-1).double())
        cnt_bin.index_add_(0, b.reshape(-1), torch.ones_like(conf).reshape(-1).double())
        hb = half_t[i:i + batch]
        for k, L in levels.items():
            if not L["n_ctx"]:
                continue
            c = L["ci"][i:i + batch]
            for h in (0, 1):
                m = (c >= 0) & (hb[:, None] == h)
                if m.any():
                    L["qsum"][h].index_add_(0, c[m], q[m].double())
    out = {"levels": {}}
    for k, L in levels.items():
        if not L["n_ctx"]:
            continue
        qs = L["qsum"].cpu().numpy()
        emp = L["emp"]
        model_js, floor_js, w_tot, cover = 0.0, 0.0, 0.0, 0.0
        for j in range(L["n_ctx"]):
            nA, nB = emp[0, j].sum(), emp[1, j].sum()
            if nA < min_count / 4 or nB < min_count / 4:
                continue
            pA, pB = emp[0, j] / nA, emp[1, j] / nB
            qA = qs[0, j] / qs[0, j].sum()
            qB = qs[1, j] / qs[1, j].sum()
            w = nA + nB
            model_js += w * 0.5 * (_js(qA, pB) + _js(qB, pA))
            floor_js += w * _js(pA, pB)
            w_tot += w
        if w_tot:
            out["levels"][str(k)] = dict(model_js=model_js / w_tot, floor_js=floor_js / w_tot,
                                         n_contexts=L["n_ctx"], coverage=w_tot / Yn.size)
    tokens_per_pos = n
    out["loss_by_position_bits"] = (pos_loss / tokens_per_pos / LN2).cpu().tolist()
    cb = cnt_bin.clamp_min(1)
    out["calibration"] = dict(confidence=(conf_sum / cb).cpu().tolist(),
                              accuracy=(acc_sum / cb).cpu().tolist(),
                              count=cnt_bin.cpu().tolist())
    tot = cnt_bin.sum()
    out["ece"] = float(((conf_sum - acc_sum).abs().sum() / tot).item())
    model.train()
    return out


def distill_run(name, student_cfg, teacher, data, tokens, lr, alpha=1.0, batch_tokens=131072,
                micro=128, seed=0, out_dir=None, log=print, compile=True):
    """alpha=1: pure soft targets from the teacher; alpha=0.5: half data, half teacher."""
    device = data.device
    if compile:
        torch._dynamo.reset()
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    T = student_cfg.block
    model = GPT(student_cfg).to(device)
    teacher.eval()
    accum = max(1, (batch_tokens // T) // micro)
    steps = max(1, tokens // (micro * accum * T))
    warmup = max(50, int(0.05 * steps))
    opt = make_optimizer(model, lr, "new")
    fwd = torch.compile(model) if (compile and device.startswith("cuda")) else model
    tfwd = torch.compile(teacher) if (compile and device.startswith("cuda")) else teacher
    t0 = time.time()
    hist = []
    every = max(1, steps // 10)
    log(f"[{name}] distillation alpha={alpha}, {steps} steps, lr {lr:.2e}")
    for step in range(steps):
        m = lr_mult(step, steps, "wsd", warmup)
        for g in opt.param_groups:
            g["lr"] = lr * m
        for _ in range(accum):
            x, y = data.batch(micro, T, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                with torch.no_grad():
                    tl, _ = tfwd(x)
                sl, _ = fwd(x)
            logp = F.log_softmax(sl.float(), -1)
            soft = -(F.softmax(tl.float(), -1) * logp).sum(-1).mean()
            hard = F.nll_loss(logp.view(-1, logp.size(-1)), y.reshape(-1))
            loss = alpha * soft + (1 - alpha) * hard
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % every == 0 and step + 1 < steps:
            vb = evaluate(model, data, T, 1_000_000)
            hist.append(dict(step=step + 1, tokens=(step + 1) * micro * accum * T, val_bpc=vb,
                             seconds=time.time() - t0))
            log(f"[{name}] {100*(step+1)/steps:4.0f}%  val {vb:.4f} bpc")
    final = evaluate(model, data, T, 8_000_000)
    tok_seen = steps * micro * accum * T
    hist.append(dict(step=steps, tokens=tok_seen, val_bpc=final, seconds=time.time() - t0))
    res = dict(name=name, model=student_cfg.__dict__ | {}, tokens=tok_seen, lr=lr, sched="wsd",
               recipe="distill", alpha=alpha, seed=seed, params=model.n_params(),
               nonemb_params=model.n_params(True), final_bpc=final, diverged=False,
               seconds=time.time() - t0, tok_per_s=tok_seen / (time.time() - t0),
               epochs=tok_seen / len(data.train), history=hist, flops=0,
               flops_per_token=0, batch_tokens=micro * accum * T, extra={})
    log(f"[{name}] FINAL {final:.4f} bpc")
    if out_dir:
        json.dump(res, open(os.path.join(out_dir, f"{name}.json"), "w"), indent=1, default=str)
    return res


def run_extras(S, hours=1.5):
    """S: a Suite already set up (data loaded, lr fit available)."""
    t_end = time.time() + hours * 3600
    out = os.path.join(S.res_dir, "E_results.json")
    E = json.load(open(out)) if os.path.exists(out) else {}

    def save():
        json.dump(E, open(out, "w"), ensure_ascii=False, indent=1)

    runs = plots.load_runs(S.res_dir)

    # E3 bilinear vs standard at other widths (cheap at 192, ~7 min each at 768)
    for d in ((64,) if S.quick else (192, 768)):
        for name, mk in ((f"e3_mha_d{d}", dict(n_embd=d, n_head=max(1, d // 64))),
                         (f"e3_bil1_d{d}", dict(n_embd=d, n_head=1, attn="bilinear"))):
            if time.time() > t_end:
                break
            S.run(name, S.cfg(**mk), S.T_base, S.lr_for(d))

    # teacher = lowest-bpc model with a checkpoint
    cands = []
    for d in (S.ck_drive, S.ck_local):
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".pt"):
                    r = runs.get(f[:-3])
                    if r and plots.ok(r):
                        cands.append((r["final_bpc"], f[:-3]))
    cands.sort()
    teacher_name = cands[0][1] if cands else None
    base_name = "att_h2_s0" if S.quick else "att_h6_s0"

    # E1 level-k conditionals for: original recipe, base, 1-head, bilinear 1-head, teacher
    E.setdefault("levels", {})
    for n in ["ladder0_original", base_name, "att_h1_s0", "att_bil_h1_s0", teacher_name]:
        if not n or n in E["levels"] or time.time() > t_end:
            continue
        m = S.load_model(n)
        if m is None:
            continue
        try:
            E["levels"][n] = level_k_study(m, S.data, min_count=10 if S.quick else 60)
            S.log(f"[E1] {n}: " + ", ".join(
                f"k={k} model {v['model_js']:.4f} / floor {v['floor_js']:.4f}"
                for k, v in E["levels"][n]["levels"].items()) + f", ECE {E['levels'][n]['ece']:.4f}")
        except Exception:
            S.log(f"[E1] {n} failed\n{traceback.format_exc()}")
        del m
        save()

    # E2 distillation into the base configuration
    if teacher_name and time.time() < t_end:
        teacher = S.load_model(teacher_name)
        base = runs.get(base_name)
        if teacher is not None and base is not None:
            for alpha in (1.0, 0.5):
                name = f"e2_distill_a{alpha}"
                if S.done(name) or time.time() > t_end:
                    continue
                try:
                    distill_run(name, S.cfg(), teacher, S.data, S.T_base, base["lr"], alpha=alpha,
                                micro=S.micro if S.micro <= 128 else 128, out_dir=S.res_dir,
                                log=S.log, compile=not S.quick and S.device.startswith("cuda"))
                except Exception:
                    S.log(f"[E2] {name} failed\n{traceback.format_exc()}")
            E["distill_teacher"] = teacher_name
            save()
            del teacher

    # E4 generation at temperature 0.7
    E.setdefault("gen_t07", {})
    sets = S._gen_setup()
    for n in [base_name, teacher_name]:
        if not n or n in E["gen_t07"] or time.time() > t_end:
            continue
        m = S.load_model(n)
        if m is None:
            continue
        E["gen_t07"][n] = analysis.generation_study(m, S.tok, S._headers, sets,
                                                    n_per_header=8 if S.quick else 48,
                                                    temperature=0.7,
                                                    max_new=120 if S.quick else 400)
        del m
        save()
    E["real_reference"] = analysis.real_reference(S._val_text, S._headers, sets)
    save()
    S.figures()
    try:
        fig_levels(E, S.fig_dir)
    except Exception:
        S.log("level figure failed\n" + traceback.format_exc())
    S.log("extras finished")


def fig_levels(E, fig_dir):
    import matplotlib.pyplot as plt
    L = E.get("levels", {})
    if not L:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    floor_done = False
    for n, r in L.items():
        ks = sorted(r["levels"], key=int)
        ax[0].plot([int(k) for k in ks], [r["levels"][k]["model_js"] for k in ks], "o-", label=n)
        if not floor_done:
            ax[0].plot([int(k) for k in ks], [r["levels"][k]["floor_js"] for k in ks], "k--",
                       label="real vs real (sampling floor)")
            floor_done = True
        lp = np.array(r["loss_by_position_bits"])
        ax[1].plot(np.arange(len(lp)), lp, lw=0.8, label=n)
        c = r["calibration"]
        ax[2].plot(c["confidence"], c["accuracy"], "o-", ms=3, label=f"{n} (ECE {r['ece']:.3f})")
    ax[0].set_xlabel("k = tokens of context")
    ax[0].set_ylabel("Jensen-Shannon divergence (bits)")
    ax[0].set_title("Model vs empirical P(next | last k tokens)")
    ax[0].legend(fontsize=6)
    ax[1].set_xlabel("position in the 512-token window")
    ax[1].set_ylabel("loss (bits / token)")
    ax[1].set_title("How much context helps")
    ax[1].set_xscale("log")
    ax[1].legend(fontsize=6)
    ax[2].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[2].set_xlabel("model confidence (top-1 probability)")
    ax[2].set_ylabel("fraction correct")
    ax[2].set_title("Calibration")
    ax[2].legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_levels.png"), dpi=150)
    plt.close()


def main(root, hours=1.5, quick=False, poems_fn=None):
    from .suite import Suite
    S = Suite(root, hours=hours, quick=quick, ckpt_local=None if quick else "/content/ckpt",
              log_name="log.txt")
    S.block = 64 if quick else 512
    S.T_base = 300_000 if quick else 300_000_000
    S.T_lr = 150_000 if quick else 200_000_000
    S.setup_data(poems_fn)
    S.p0_throughput()
    fit = json.load(open(os.path.join(S.res_dir, "P1_lr_fit.json")))
    fit["best"] = {int(k): v for k, v in fit["best"].items()}
    S.lr_fit = fit
    S.log("===== extras")
    run_extras(S, hours)

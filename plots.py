"""Figures and summary.md from whatever results exist (safe to call any time)."""
import json
import math
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3})


def load_runs(res_dir):
    runs = {}
    for f in os.listdir(res_dir):
        if f.endswith(".json") and not f.startswith("P"):
            try:
                r = json.load(open(os.path.join(res_dir, f)))
                runs[r["name"]] = r
            except Exception:
                pass
    return runs


def ok(r):
    return r is not None and not r.get("diverged") and math.isfinite(r.get("final_bpc", float("nan")))


def curve(r):
    h = [x for x in r["history"] if "val_bpc" in x and math.isfinite(x["val_bpc"])]
    return [x["tokens"] / 1e6 for x in h], [x["val_bpc"] for x in h]


def fig_ladder(runs, fig_dir, bzip2):
    names = ["ladder0_original", "ladder1_adamw_clip", "ladder2_tuned_lr_wsd",
             "ladder3_no_dropout", "ladder4_scaled_init_nobias", "ladder5_gelu", "att_h6_s0"]
    labels = ["original\nrecipe", "+AdamW β2=.95\nwd .1, clip", "+tuned lr\nwarmup+WSD",
              "+no dropout", "+scaled init\nno bias", "+GELU", "+RoPE\n(new base)"]
    pts = [(l, runs[n]) for n, l in zip(names, labels) if ok(runs.get(n))]
    if len(pts) < 2:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(range(len(pts)), [r["final_bpc"] for _, r in pts], color="#3b6ea5")
    for i, (_, r) in enumerate(pts):
        ax[0].text(i, r["final_bpc"], f"{r['final_bpc']:.3f}", ha="center", va="bottom", fontsize=8)
    ax[0].set_xticks(range(len(pts)), [l for l, _ in pts], fontsize=7)
    lo = min(r["final_bpc"] for _, r in pts)
    hi = max(r["final_bpc"] for _, r in pts)
    ax[0].set_ylim(lo - 0.15 * (hi - lo + 0.05), hi + 0.15 * (hi - lo + 0.05))
    ax[0].set_ylabel("val bits / char")
    ax[0].set_title("Recipe ladder: one fix at a time (same model size, same tokens)")
    for l, r in pts:
        x, y = curve(r)
        ax[1].plot(x, y, label=l.replace("\n", " "))
    ax[1].set_xlabel("training tokens (M)")
    ax[1].set_ylabel("val bits / char")
    ax[1].set_title("validation curves")
    ax[1].legend(fontsize=7)
    plt.tight_layout()
    p = os.path.join(fig_dir, "fig_ladder.png")
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def fig_lr(runs, fig_dir, fit):
    by = {}
    for n, r in runs.items():
        if n.startswith("lr_d") and ok(r):
            d = r["model"]["n_embd"]
            by.setdefault(d, []).append((r["lr"], r["final_bpc"]))
    if not by:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for d in sorted(by):
        pts = sorted(by[d])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=f"width {d}")
        if fit and str(d) in {str(k) for k in fit.get("best", {})}:
            b = fit["best"].get(str(d), fit["best"].get(d))
            ax.axvline(b, ls=":", lw=0.8, color=ax.lines[-1].get_color())
    ax.set_xscale("log")
    ax.set_xlabel("peak learning rate")
    ax.set_ylabel("val bits / char")
    t = "learning-rate sweep"
    if fit:
        t += f"   best lr ∝ width^{fit['slope']:.2f}"
    ax.set_title(t)
    ax.legend()
    plt.tight_layout()
    p = os.path.join(fig_dir, "fig_lr.png")
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def _seeds(runs, prefix):
    return [r for n, r in runs.items() if re.fullmatch(re.escape(prefix) + r"_s\d+", n) and ok(r)]


def _best_lr(runs, prefix):
    c = [runs.get(f"{prefix}_s0{suf}") for suf in ("", "_lrx0.5", "_lrx2.0")]
    c = [r for r in c if ok(r)]
    return min(c, key=lambda r: r["final_bpc"]) if c else None


def fig_heads(runs, fig_dir):
    hs = sorted({int(m.group(1)) for n in runs
                 for m in [re.fullmatch(r"att_h(\d+)_s\d+", n)] if m and ok(runs[n])})
    if not hs:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    mean, sd = [], []
    for h in hs:
        v = [r["final_bpc"] for r in _seeds(runs, f"att_h{h}")]
        mean.append(np.mean(v))
        sd.append(np.std(v) if len(v) > 1 else 0)
    ax.errorbar(hs, mean, yerr=sd, fmt="o-", capsize=3, color="#3b6ea5",
                label="multi-head, head dim = width / heads (common lr, mean over seeds)")
    for h, m in zip(hs, mean):
        ax.annotate(f"{m:.3f}", (h, m), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7)
    bl = [(h, _best_lr(runs, f"att_h{h}")) for h in hs]
    bl = [(h, r) for h, r in bl if r]
    if bl:
        ax.plot([h for h, _ in bl], [r["final_bpc"] for _, r in bl], "x--", color="#7f8c8d",
                label="same, best of 0.5x / 1x / 2x lr (seed 0)")
    others = [("att_bil_h1", "bilinear W_QK, 1 head (full rank)", "#c0392b"),
              ("att_bil_h6", "bilinear W_QK, 6 heads (full rank)", "#e67e22"),
              ("att_bil_h2", "bilinear W_QK, 2 heads", "#e67e22"),
              ("att_wideqk_h6", "6 heads, Q/K dim 384 (factored full rank)", "#27ae60"),
              ("att_wideqk_h2", "2 heads, Q/K dim = width", "#27ae60")]
    for pre, lab, col in others:
        v = _seeds(runs, pre)
        if v:
            b = _best_lr(runs, pre)
            extra = f", best lr {b['final_bpc']:.3f}" if b else ""
            ax.axhline(np.mean([r["final_bpc"] for r in v]), ls="--", color=col,
                       label=f"{lab}: {np.mean([r['final_bpc'] for r in v]):.3f}{extra} "
                             f"({v[0]['params']/1e6:.1f}M params)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(hs, [str(h) for h in hs])
    ax.set_xlabel("number of attention heads (width fixed)")
    ax.set_ylabel("val bits / char")
    ax.set_title("Is multi-head attention worth it?")
    ax.legend(fontsize=6.5)
    plt.tight_layout()
    p = os.path.join(fig_dir, "fig_heads.png")
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def fig_head_analysis(res_dir, fig_dir):
    f = os.path.join(res_dir, "P4_heads.json")
    if not os.path.exists(f):
        return None
    H = json.load(open(f))
    names = [n for n in H if H[n]]
    if not names:
        return None
    fig, axes = plt.subplots(len(names), 3, figsize=(12, 1.6 + 1.3 * len(names)), squeeze=False)
    for i, n in enumerate(names):
        h = H[n]
        mats = [(np.array(h["ablation_all"]) * 1000, "Δ loss, all tokens (mnats)"),
                (np.array(h["ablation_rhyme"]) * 1000, "Δ loss, line-final tokens (mnats)"),
                (np.array(h["rhyme_mass"]) / max(1e-9, h["rhyme_mass_uniform"]),
                 "attention to previous line ends / uniform")]
        for j, (M, t) in enumerate(mats):
            ax = axes[i, j]
            im = ax.imshow(M, aspect="auto", cmap="magma")
            ax.set_title(f"{n}: {t}", fontsize=7)
            ax.set_ylabel("layer")
            ax.set_xlabel("head")
            ax.set_xticks(range(M.shape[1]))
            ax.set_yticks(range(M.shape[0]))
            ax.tick_params(labelsize=6)
            ax.grid(False)
            if M.shape[1] <= 12:
                for (yy, xx), v in np.ndenumerate(M):
                    ax.text(xx, yy, f"{v:.0f}" if j < 2 else f"{v:.1f}", ha="center",
                            va="center", fontsize=5, color="w" if v < M.mean() else "k")
            plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    p = os.path.join(fig_dir, "fig_head_analysis.png")
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def fig_curves(runs, fig_dir, names, labels, fname, title):
    pts = [(l, runs[n]) for n, l in zip(names, labels) if ok(runs.get(n))]
    if len(pts) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    for l, r in pts:
        x, y = curve(r)
        ax.plot(x, y, label=f"{l}: {r['final_bpc']:.4f}")
    ax.set_xlabel("training tokens (M)")
    ax.set_ylabel("val bits / char")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ys = [r["final_bpc"] for _, r in pts]
    ax.set_ylim(min(ys) - 0.05, min(ys) + 0.6)
    plt.tight_layout()
    p = os.path.join(fig_dir, fname)
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def fig_misc(runs, fig_dir):
    names = [("att_h6_s0", "base (RoPE, GELU, L4 d384)"), ("pos_learned", "learned positions"),
             ("pos_none", "no positions (NoPE)"), ("mlp_relu", "ReLU MLP"),
             ("mlp_swiglu", "SwiGLU MLP"), ("dropout_0.1", "dropout 0.1"),
             ("dw_L2_d512", "2 layers x 512"), ("dw_L8_d256", "8 layers x 256"),
             ("dw_L16_d192", "16 layers x 192"), ("batch_32k", "batch 32k tokens"),
             ("batch_65k", "batch 65k tokens"), ("batch_262k", "batch 262k tokens")]
    pts = [(l, runs[n]) for n, l in names if ok(runs.get(n))]
    if len(pts) < 2:
        return None
    base = pts[0][1]["final_bpc"] if pts[0][0].startswith("base") else None
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.35 * len(pts)))
    ys = range(len(pts))
    vals = [r["final_bpc"] for _, r in pts]
    ax.barh(ys, vals, color=["#3b6ea5" if i else "#555" for i in range(len(pts))])
    for y, (l, r) in zip(ys, pts):
        ax.text(r["final_bpc"], y, f" {r['final_bpc']:.4f}  ({r['params']/1e6:.1f}M)",
                va="center", fontsize=7)
    ax.set_yticks(list(ys), [l for l, _ in pts], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(min(vals) - 0.05, max(vals) + 0.08)
    if base:
        ax.axvline(base, ls=":", color="k", lw=0.8)
    ax.set_xlabel("val bits / char (lower is better)")
    ax.set_title("Architecture and batch-size ablations (300M tokens each)")
    plt.tight_layout()
    p = os.path.join(fig_dir, "fig_misc.png")
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def scaling_fit(runs):
    by = {}
    for n, r in runs.items():
        if n.startswith("sc_") and ok(r):
            by.setdefault(r["extra"]["budget"], []).append(r)
    fits = []
    for C in sorted(by):
        rs = sorted(by[C], key=lambda r: r["params"])
        if len(rs) < 3:
            fits.append(dict(budget=C, n_runs=len(rs), points=[(r["params"], r["tokens"], r["final_bpc"]) for r in rs]))
            continue
        x = np.log([r["params"] for r in rs])
        y = np.array([r["final_bpc"] for r in rs])
        i = int(y.argmin())
        a, b, c = np.polyfit(x, y, 2)
        if a > 0:
            lx = float(np.clip(-b / (2 * a), x.min(), x.max()))
            interior = 0 < i < len(rs) - 1
        else:
            lx, interior = float(x[i]), False
        n_opt = math.exp(lx)
        fits.append(dict(budget=C, n_runs=len(rs), n_opt=n_opt,
                         d_opt=C / (6 * n_opt), bpc_opt=float(np.polyval([a, b, c], lx)),
                         interior_minimum=interior,
                         points=[(r["params"], r["tokens"], r["final_bpc"]) for r in rs]))
    good = [f for f in fits if "n_opt" in f]
    law = None
    if len(good) >= 2:
        lc = np.log([f["budget"] for f in good])
        a_n, _ = np.polyfit(lc, np.log([f["n_opt"] for f in good]), 1)
        a_d, _ = np.polyfit(lc, np.log([f["d_opt"] for f in good]), 1)
        law = dict(n_opt_exponent=float(a_n), d_opt_exponent=float(a_d))
    return dict(per_budget=fits, law=law)


def fig_scaling(runs, fig_dir, res_dir, train_tokens):
    fit = scaling_fit(runs)
    json.dump(fit, open(os.path.join(res_dir, "P6_scaling_fit.json"), "w"), indent=1)
    if not fit["per_budget"]:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    cmap = plt.get_cmap("viridis")
    nb = len(fit["per_budget"])
    for k, f in enumerate(fit["per_budget"]):
        col = cmap(k / max(1, nb - 1))
        P = np.array(f["points"])
        ax[0].plot(P[:, 0], P[:, 2], "o-", color=col, label=f"C = {f['budget']:.0e} FLOPs")
        for p in P:
            ax[0].annotate(f"{p[1]/train_tokens:.1f} ep", (p[0], p[2]), fontsize=6,
                           textcoords="offset points", xytext=(0, 5), ha="center")
        if "n_opt" in f:
            ax[0].plot(f["n_opt"], f["bpc_opt"], "*", ms=12, color=col,
                       markeredgecolor="k" if f["interior_minimum"] else "r")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("parameters")
    ax[0].set_ylabel("val bits / char")
    ax[0].set_title("IsoFLOP curves (★ fitted optimum; red edge = optimum at grid edge)")
    ax[0].legend(fontsize=7)
    good = [f for f in fit["per_budget"] if "n_opt" in f]
    if good:
        Cs = [f["budget"] for f in good]
        ax[1].loglog(Cs, [f["n_opt"] for f in good], "o-", label="optimal parameters N*")
        ax[1].loglog(Cs, [f["d_opt"] for f in good], "s-", label="optimal tokens D*")
        ax[1].axhline(train_tokens, ls=":", color="gray", label="one epoch of the corpus")
        ax[1].set_xlabel("compute budget C (FLOPs)")
        law = fit["law"]
        t = "compute-optimal allocation"
        if law:
            t += f"   N* ∝ C^{law['n_opt_exponent']:.2f}, D* ∝ C^{law['d_opt_exponent']:.2f}"
        ax[1].set_title(t, fontsize=8)
        ax[1].legend(fontsize=7)
    plt.tight_layout()
    p = os.path.join(fig_dir, "fig_scaling.png")
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def table(rows, cols):
    out = ["| " + " | ".join(c for c, _ in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(f(r) for _, f in cols) + " |")
    return "\n".join(out)


def summary(runs, res_dir, fig_dir, meta):
    L = ["# Shaaer v2 experiments: summary", ""]
    L.append(f"Data: {meta['train_tokens']:,} train tokens / {meta['val_tokens']:,} val tokens "
             f"(BPE-{meta['vocab_size']}), {meta['n_duplicates']:,} duplicate poems removed, "
             f"{100*meta['val_bait_overlap']:.2f}% of val lines also appear verbatim in train.")
    comp = meta.get("compressors_val", {})
    if comp:
        L.append("Compressors on the val text (bits/char): " +
                 ", ".join(f"{k} {v:.3f}" for k, v in comp.items()))
    L.append("")
    cols = [("run", lambda r: r["name"]),
            ("params", lambda r: f"{r['params']/1e6:.2f}M"),
            ("tokens", lambda r: f"{r['tokens']/1e6:.0f}M"),
            ("epochs", lambda r: f"{r['epochs']:.2f}"),
            ("lr", lambda r: f"{r['lr']:.1e}"),
            ("sched", lambda r: r["sched"]),
            ("val bpc", lambda r: f"{r['final_bpc']:.4f}" if ok(r) else "diverged"),
            ("tok/s", lambda r: f"{r['tok_per_s']/1e6:.2f}M"),
            ("min", lambda r: f"{r['seconds']/60:.1f}")]
    groups = [("Learning rate", "lr_"), ("Recipe ladder", "ladder"), ("Attention", "att_"),
              ("Schedules", "sched_"), ("Positions / MLP / dropout", ("pos_", "mlp_", "dropout")),
              ("Depth vs width", "dw_"), ("Batch size", "batch_"), ("Scaling", "sc_")]
    for title, pre in groups:
        rs = sorted([r for n, r in runs.items() if n.startswith(pre)], key=lambda r: r["name"])
        if rs:
            L += [f"## {title}", "", table(rs, cols), ""]
    f = os.path.join(res_dir, "P6_scaling_fit.json")
    if os.path.exists(f):
        fit = json.load(open(f))
        fence = chr(96) * 3
        L += ["## Scaling fit", "", fence, json.dumps(fit, indent=1)[:4000], fence, ""]
    for gname in ("P4_generation.json", "P10_generation_scaled.json"):
        g = os.path.join(res_dir, gname)
        if os.path.exists(g):
            G = json.load(open(g))
            keys = ["well_formed", "rhyme_consistency", "repeated_rhyme_word", "copied",
                    "invented_words", "invented_rhyme_words", "hemistich_chars"]
            rows = [(n, v["overall"]) for n, v in G.items()]
            L += [f"## Generation ({gname})", "",
                  "| model | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
            for n, o in rows:
                L.append(f"| {n} | " + " | ".join(
                    f"{o.get(k, float('nan')):.3f}" for k in keys) + " |")
            L.append("")
    figs = sorted(x for x in os.listdir(fig_dir) if x.endswith(".png"))
    L += ["## Figures", ""] + [f"![{x}](../figures/{x})" for x in figs]
    open(os.path.join(res_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(L))


def make_all(res_dir, fig_dir, meta, log=print):
    runs = load_runs(res_dir)
    fit = None
    f = os.path.join(res_dir, "P1_lr_fit.json")
    if os.path.exists(f):
        fit = json.load(open(f))
    bz = meta.get("compressors_val", {}).get("bzip2")
    fig_ladder(runs, fig_dir, bz)
    fig_lr(runs, fig_dir, fit)
    fig_heads(runs, fig_dir)
    fig_head_analysis(res_dir, fig_dir)
    fig_curves(runs, fig_dir, ["att_h6_s0", "sched_constant", "sched_cosine", "sched_wsd_sqrt"],
               ["WSD (linear decay, last 20%)", "constant", "cosine to 10%", "WSD (1-sqrt decay)"],
               "fig_schedules.png", "Learning-rate schedules (same model, same tokens)")
    fig_misc(runs, fig_dir)
    fig_scaling(runs, fig_dir, res_dir, meta["train_tokens"])
    summary(runs, res_dir, fig_dir, meta)

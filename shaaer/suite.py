"""Overnight experiment suite. Resumable: every finished run writes
results/<name>.json and is skipped on restart. A global deadline stops new
runs that would not finish in time.

Phases, in priority order:
  P0 throughput      micro-batch sizes on the GPU
  P1 lr              learning-rate sweep at widths 192/384/768 -> lr(width) fit
  P2 ladder          original recipe, then one fix at a time -> what was holding it back
  P3 attention       number of heads at fixed width; full-rank bilinear W_QK; wide QK
  P4 analysis        head ablation, rhyme attention, generation metrics
  P5 schedules       constant / cosine / WSD / WSD-sqrt
  P6 scaling         IsoFLOP sweep -> compute-optimal model size on this corpus
  P7 misc            positions, MLP, dropout, depth vs width, batch size
  P8 seeds           second seed for the attention study
  P9 scaling+        largest compute budget
  P10 generation     generation metrics for the best scaled models
"""
import json
import math
import os
import time
import traceback

import numpy as np
import torch

from . import analysis, data as D, plots
from .model import GPT, GPTConfig
from .train import Data, train_run


class Suite:
    def __init__(self, root, hours=9.5, quick=False, device=None, ckpt_local=None,
                 log_name="log.txt"):
        self.root = root
        self.quick = quick
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.res_dir = os.path.join(root, "results")
        self.fig_dir = os.path.join(root, "figures")
        self.ck_drive = os.path.join(root, "checkpoints")
        self.ck_local = ckpt_local or self.ck_drive
        for d in (self.res_dir, self.fig_dir, self.ck_drive, self.ck_local):
            os.makedirs(d, exist_ok=True)
        self.logf = open(os.path.join(root, log_name), "a", encoding="utf-8")
        self.t_start = time.time()
        self.deadline = self.t_start + hours * 3600
        self.flops_rate = None   # achieved FLOP/s, updated after each run
        self.micro = 256
        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------ utilities
    def log(self, *a):
        s = " ".join(str(x) for x in a)
        stamp = time.strftime("%H:%M:%S")
        print(f"{stamp} {s}", flush=True)
        self.logf.write(f"{stamp} {s}\n")
        self.logf.flush()

    def path(self, name):
        return os.path.join(self.res_dir, f"{name}.json")

    def done(self, name):
        return os.path.exists(self.path(name))

    def load_res(self, name):
        return json.load(open(self.path(name))) if self.done(name) else None

    def time_left(self):
        return self.deadline - time.time()

    def setup_data(self, poems_fn, vocab_size=512):
        ddir = os.path.join(self.root, "data")
        if not os.path.exists(os.path.join(ddir, "meta.json")):
            poems = poems_fn()
            D.build(ddir, poems, vocab_size=vocab_size, log=self.log)
        self.meta, self.tok, train, val = D.load(ddir)
        self.data = Data(train, val, self.tok.token_char_lengths(), self.device)
        self.train_text_path = os.path.join(ddir, "train_poems.txt")
        self.val_text_path = os.path.join(ddir, "val_poems.txt")
        self.log(f"data: {self.meta['train_tokens']:,} train tokens, "
                 f"{self.meta['val_tokens']:,} val tokens, vocab {self.meta['vocab_size']}, "
                 f"duplicates removed {self.meta['n_duplicates']:,}, "
                 f"val bait overlap {100*self.meta['val_bait_overlap']:.2f}%")

    def cfg(self, **kw):
        base = dict(vocab=self.meta["vocab_size"], block=self.block, n_layer=4,
                    n_embd=384, n_head=6, attn="mha", pos="rope", mlp="gelu",
                    init="scaled", bias=False, dropout=0.0)
        if self.quick:
            base.update(n_layer=2, n_embd=64, n_head=2)
        base.update(kw)
        return GPTConfig(**base)

    def run(self, name, cfg, tokens, lr, save=None, **kw):
        """Train unless already done. save in {None, 'drive', 'local'}."""
        if self.done(name):
            return self.load_res(name)
        fpt = GPT(cfg).flops_per_token() if self.flops_rate else None
        if fpt and self.flops_rate:
            est = fpt * tokens / self.flops_rate + 90
            if est > self.time_left():
                self.log(f"[{name}] skipped: needs ~{est/60:.0f} min, "
                         f"{self.time_left()/60:.0f} min left")
                return None
        micro = kw.pop("micro_batch", self.micro)
        if self.quick:
            kw.setdefault("batch_tokens", 2048)
        for attempt in range(4):
            try:
                out = self.ck_drive if save == "drive" else self.ck_local
                res, model = train_run(
                    name, cfg, self.data, tokens, lr, micro_batch=micro,
                    compile=not self.quick and self.device.startswith("cuda"),
                    out_dir=self.res_dir, save_ckpt=False, log=self.log,
                    eval_tokens=200_000 if self.quick else 1_000_000,
                    final_eval_tokens=400_000 if self.quick else 8_000_000, **kw)
                if save:
                    torch.save(dict(state_dict=model.state_dict(), cfg=res["model"]),
                               os.path.join(out, f"{name}.pt"))
                del model
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                if not res["diverged"] and res["seconds"] > 30:
                    rate = res["flops"] / res["seconds"]
                    self.flops_rate = rate if self.flops_rate is None else \
                        0.5 * self.flops_rate + 0.5 * rate
                return res
            except Exception as e:
                if "out of memory" not in str(e).lower():
                    self.log(f"[{name}] FAILED\n{traceback.format_exc()}")
                    break
                micro //= 2
                self.log(f"[{name}] OOM, retrying with micro-batch {micro}")
                import gc
                gc.collect()
                torch._dynamo.reset()
                torch.cuda.empty_cache()
                if micro < 8:
                    break
        return None

    def load_model(self, name):
        for d in (self.ck_drive, self.ck_local):
            p = os.path.join(d, f"{name}.pt")
            if os.path.exists(p):
                ck = torch.load(p, map_location=self.device, weights_only=False)
                m = GPT(GPTConfig(**ck["cfg"])).to(self.device)
                m.load_state_dict(ck["state_dict"])
                return m
        return None

    def save_json(self, name, obj):
        json.dump(obj, open(os.path.join(self.res_dir, name), "w"),
                  ensure_ascii=False, indent=1)

    def figures(self):
        try:
            plots.make_all(self.res_dir, self.fig_dir, self.meta, log=self.log)
        except Exception:
            self.log("plotting failed\n" + traceback.format_exc())

    # ------------------------------------------------------------ phases
    def p0_throughput(self):
        out = os.path.join(self.res_dir, "P0_throughput.json")
        if os.path.exists(out):
            self.micro = json.load(open(out))["best_micro"]
            return
        if not self.device.startswith("cuda"):
            self.micro = 16
            json.dump(dict(best_micro=16, rows=[]), open(out, "w"))
            return
        from .train import make_optimizer
        rows = []
        cfg = self.cfg()
        for micro in (64, 128, 256, 512):
            try:
                m = GPT(cfg).to(self.device)
                f = torch.compile(m)
                opt = make_optimizer(m, 1e-3, "new")
                g = torch.Generator(device=self.device)
                g.manual_seed(0)
                for i in range(25):
                    if i == 5:
                        torch.cuda.synchronize()
                        t0 = time.time()
                    x, y = self.data.batch(micro, self.block, g)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, loss = f(x, y)
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                tps = 20 * micro * self.block / (time.time() - t0)
                mem = torch.cuda.max_memory_allocated() / 2**30
                rows.append(dict(micro=micro, tok_per_s=tps, mem_gb=mem))
                self.log(f"[P0] micro {micro}: {tps/1e6:.2f}M tok/s, {mem:.1f} GB")
                del m, f, opt
            except torch.cuda.OutOfMemoryError:
                rows.append(dict(micro=micro, oom=True))
                self.log(f"[P0] micro {micro}: OOM")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        ok = [r for r in rows if "tok_per_s" in r]
        best = max(ok, key=lambda r: r["tok_per_s"])
        # prefer the smallest micro-batch within 3% of the best
        best = min((r for r in ok if r["tok_per_s"] > 0.97 * best["tok_per_s"]),
                   key=lambda r: r["micro"])
        self.micro = best["micro"]
        fpt = GPT(cfg).flops_per_token()
        self.flops_rate = 0.6 * best["tok_per_s"] * fpt  # conservative until real runs
        json.dump(dict(best_micro=self.micro, rows=rows,
                       gpu=torch.cuda.get_device_name(0)), open(out, "w"), indent=1)

    def p1_lr(self):
        """Sweep lr per width; extend the grid until the best lr is interior."""
        grids = {192: [1e-3, 2e-3, 4e-3], 384: [5e-4, 1e-3, 2e-3], 768: [2.5e-4, 5e-4, 1e-3]}
        if self.quick:
            grids = {64: [1e-3, 3e-3, 1e-2], 128: [1e-3, 3e-3, 1e-2]}
        best = {}
        for d, lrs in grids.items():
            lrs = sorted(lrs)
            pts = []
            for _ in range(6):
                for lr in lrs:
                    self.run(f"lr_d{d}_{lr:.2e}", self.cfg(n_embd=d, n_head=max(1, d // 64)),
                             self.T_lr, lr)
                pts = [(lr, self.load_res(f"lr_d{d}_{lr:.2e}")) for lr in lrs]
                pts = [(lr, r["final_bpc"]) for lr, r in pts if r and not r["diverged"]
                       and math.isfinite(r["final_bpc"])]
                if len(pts) < 2:
                    break
                i = int(np.argmin([p[1] for p in pts]))
                if i == 0 and pts[0][0] == lrs[0]:
                    lrs = [lrs[0] / 2] + lrs
                elif i == len(pts) - 1 and pts[-1][0] == lrs[-1]:
                    lrs = lrs + [lrs[-1] * 2]
                else:
                    break
            if len(pts) < 2:
                continue
            x = np.log([p[0] for p in pts])
            y = np.array([p[1] for p in pts])
            i = int(y.argmin())
            if 0 < i < len(x) - 1:
                a, b, _ = np.polyfit(x[i - 1:i + 2], y[i - 1:i + 2], 2)
                lx = -b / (2 * a) if a > 0 else x[i]
                lx = float(np.clip(lx, x[i - 1], x[i + 1]))
            else:
                lx = float(x[i])
                self.log(f"[P1] width {d}: optimum still at the grid edge")
            best[d] = math.exp(lx)
        if len(best) >= 2:
            ld = np.log(list(best.keys()))
            ll = np.log(list(best.values()))
            slope, icpt = np.polyfit(ld, ll, 1)
        else:
            slope, icpt = -1.0, math.log(1e-3 * 384)
        fit = dict(best=best, slope=float(slope), intercept=float(icpt))
        self.lr_fit = fit
        self.save_json("P1_lr_fit.json", fit)
        self.log(f"[P1] best lr per width {best}; lr(d) ~ d^{slope:.2f}")

    def lr_for(self, d):
        f = self.lr_fit
        return float(np.clip(math.exp(f["intercept"] + f["slope"] * math.log(d)), 1e-4, 2e-2))

    def p2_ladder(self):
        d = 64 if self.quick else 384
        lr = self.lr_for(d)
        steps = [
            ("ladder0_original", dict(pos="learned", mlp="relu", init="old", bias=True, dropout=0.1),
             dict(recipe="old", sched="constant"), 3e-4),
            ("ladder1_adamw_clip", dict(pos="learned", mlp="relu", init="old", bias=True, dropout=0.1),
             dict(recipe="new", sched="constant"), 3e-4),
            ("ladder2_tuned_lr_wsd", dict(pos="learned", mlp="relu", init="old", bias=True, dropout=0.1),
             dict(recipe="new", sched="wsd"), lr),
            ("ladder3_no_dropout", dict(pos="learned", mlp="relu", init="old", bias=True, dropout=0.0),
             dict(recipe="new", sched="wsd"), lr),
            ("ladder4_scaled_init_nobias", dict(pos="learned", mlp="relu"),
             dict(recipe="new", sched="wsd"), lr),
            ("ladder5_gelu", dict(pos="learned"), dict(recipe="new", sched="wsd"), lr),
        ]
        for name, mk, tk, lr_ in steps:
            self.run(name, self.cfg(**mk), self.T_base, lr_,
                     save="drive" if name == "ladder0_original" else None, **tk)
        self.base()   # ladder6 = base (+RoPE)

    def base(self, seed=0):
        d = 64 if self.quick else 384
        return self.run(f"att_h6_s{seed}" if not self.quick else f"att_h2_s{seed}",
                        self.cfg(), self.T_base, self.lr_for(d), seed=seed,
                        save="drive" if seed == 0 else None)

    def attention_variants(self):
        d = 64 if self.quick else 384
        heads = [1, 2] if self.quick else [1, 2, 3, 6, 12, 24]
        v = []
        for h in heads:
            v.append((f"att_h{h}", dict(n_head=h)))
        v.append(("att_bil_h1", dict(attn="bilinear", n_head=1)))
        v.append((f"att_bil_h{2 if self.quick else 6}",
                  dict(attn="bilinear", n_head=2 if self.quick else 6)))
        v.append((f"att_wideqk_h{2 if self.quick else 6}",
                  dict(n_head=2 if self.quick else 6, qk_dim=d)))
        return v

    def p3_attention(self, seed=0):
        d = 64 if self.quick else 384
        for name, mk in self.attention_variants():
            self.run(f"{name}_s{seed}", self.cfg(**mk), self.T_base, self.lr_for(d),
                     seed=seed, save="drive" if seed == 0 else None)

    def p3b_attention_lr(self):
        """Each attention variant at 0.5x and 2x the common lr (seed 0)."""
        d = 64 if self.quick else 384
        for name, mk in self.attention_variants():
            for mult in (0.5, 2.0):
                self.run(f"{name}_s0_lrx{mult}", self.cfg(**mk), self.T_base,
                         self.lr_for(d) * mult)

    def p4_analysis(self):
        out = os.path.join(self.res_dir, "P4_heads.json")
        heads = json.load(open(out)) if os.path.exists(out) else {}
        for name, _ in self.attention_variants():
            key = f"{name}_s0"
            if key in heads:
                continue
            m = self.load_model(key)
            if m is None:
                continue
            try:
                heads[key] = analysis.head_study(m, self.data, self.tok,
                                                 n_windows=16 if self.quick else 128)
                self.log(f"[P4] head study {key} done")
            except Exception:
                self.log(f"[P4] head study {key} failed\n{traceback.format_exc()}")
            del m
            json.dump(heads, open(out, "w"), indent=1)
        self.generation(["ladder0_original", "att_h6_s0" if not self.quick else "att_h2_s0",
                         "att_h1_s0", "att_bil_h1_s0"], "P4_generation.json")

    def _gen_setup(self):
        if getattr(self, "_train_sets", None) is None:
            tt = open(self.train_text_path, encoding="utf-8").read()
            self._train_sets = analysis.corpus_sets(tt)
            del tt
            self._val_text = open(self.val_text_path, encoding="utf-8").read()
            self._headers = [h for h, _ in self.meta["top_headers"][:4]]
        return self._train_sets

    def generation(self, names, fname):
        out = os.path.join(self.res_dir, fname)
        gen = json.load(open(out)) if os.path.exists(out) else {}
        sets = self._gen_setup()
        if "real" not in gen:
            gen["real"] = analysis.real_reference(self._val_text, self._headers, sets)
        for n in names:
            if n in gen:
                continue
            m = self.load_model(n)
            if m is None:
                continue
            try:
                gen[n] = analysis.generation_study(
                    m, self.tok, self._headers, sets,
                    n_per_header=8 if self.quick else 48,
                    max_new=120 if self.quick else 400)
                self.log(f"[gen] {n}: {json.dumps(gen[n]['overall'])}")
            except Exception:
                self.log(f"[gen] {n} failed\n{traceback.format_exc()}")
            del m
            json.dump(gen, open(out, "w"), ensure_ascii=False, indent=1)

    def p5_schedules(self):
        d = 64 if self.quick else 384
        lr = self.lr_for(d)
        for s in ("constant", "cosine", "wsd_sqrt"):
            self.run(f"sched_{s}", self.cfg(), self.T_base, lr, sched=s)

    def scaling_models(self):
        if self.quick:
            return [(1, 32), (2, 64), (2, 96), (3, 128)]
        return [(2, 128), (3, 192), (4, 256), (5, 320), (6, 384), (6, 512),
                (8, 576), (10, 640), (12, 768)]

    def p6_scaling(self, budgets):
        lo, hi = (2e5, 3e6) if self.quick else (64e6, 1.5e9)
        for C in budgets:
            for L, d in self.scaling_models():
                cfg = self.cfg(n_layer=L, n_embd=d, n_head=max(1, d // 64))
                fpt = GPT(cfg).flops_per_token()
                tokens = int(C / fpt)
                if not (lo <= tokens <= hi):
                    continue
                self.run(f"sc_{C:.0e}_L{L}_d{d}", cfg, tokens, self.lr_for(d),
                         save="local", extra=dict(budget=C))
            self.figures()

    def p7_misc(self):
        d = 64 if self.quick else 384
        lr = self.lr_for(d)
        runs = [
            ("pos_learned", dict(pos="learned"), {}),
            ("pos_none", dict(pos="none"), {}),
            ("mlp_relu", dict(mlp="relu"), {}),
            ("mlp_swiglu", dict(mlp="swiglu"), {}),
            ("dropout_0.1", dict(dropout=0.1), {}),
        ]
        if not self.quick:
            runs += [
                ("dw_L2_d512", dict(n_layer=2, n_embd=512, n_head=8), {}),
                ("dw_L8_d256", dict(n_layer=8, n_embd=256, n_head=4), {}),
                ("dw_L16_d192", dict(n_layer=16, n_embd=192, n_head=3), {}),
                ("batch_32k", {}, dict(batch_tokens=32768)),
                ("batch_65k", {}, dict(batch_tokens=65536)),
                ("batch_262k", {}, dict(batch_tokens=262144)),
            ]
        for name, mk, tk in runs:
            dd = mk.get("n_embd", d)
            self.run(name, self.cfg(**mk), self.T_base, self.lr_for(dd), **tk)

    def p10_best_scaled(self):
        rows = [json.load(open(os.path.join(self.res_dir, f)))
                for f in os.listdir(self.res_dir) if f.startswith("sc_") and f.endswith(".json")]
        best = {}
        for r in rows:
            C = r["extra"]["budget"]
            if C not in best or r["final_bpc"] < best[C]["final_bpc"]:
                best[C] = r
        names = [best[C]["name"] for C in sorted(best)]
        for n in names:   # keep the winners on Drive
            src = os.path.join(self.ck_local, f"{n}.pt")
            dst = os.path.join(self.ck_drive, f"{n}.pt")
            if os.path.exists(src) and not os.path.exists(dst):
                import shutil
                shutil.copy(src, dst)
        self.generation(names, "P10_generation_scaled.json")

    # ------------------------------------------------------------ main
    def main(self, poems_fn):
        self.block = 64 if self.quick else 512
        self.T_base = 300_000 if self.quick else 300_000_000
        self.T_lr = 150_000 if self.quick else 200_000_000
        self.setup_data(poems_fn)
        if self.quick:
            self.micro = 16
        phases = [
            ("P0", self.p0_throughput),
            ("P1", self.p1_lr),
            ("P2", self.p2_ladder),
            ("P3", lambda: self.p3_attention(0)),
            ("P3b", self.p3b_attention_lr),
            ("P4", self.p4_analysis),
            ("P5", self.p5_schedules),
            ("P6", lambda: self.p6_scaling([3e9, 1e10] if self.quick else [3e15, 1e16, 3e16, 1e17])),
            ("P7", self.p7_misc),
            ("P8", lambda: self.p3_attention(1)),
            ("P9", lambda: self.p6_scaling([3e10] if self.quick else [3e17])),
            ("P10", self.p10_best_scaled),
        ]
        for tag, fn in phases:
            if self.time_left() < 120:
                self.log(f"deadline reached before {tag}")
                break
            self.log(f"===== {tag}  ({self.time_left()/3600:.2f} h left)")
            try:
                fn()
            except Exception:
                self.log(f"{tag} FAILED\n{traceback.format_exc()}")
            self.figures()
        self.log(f"suite finished in {(time.time()-self.t_start)/3600:.2f} h")


def ashaar_poems():
    import datasets
    ds = datasets.load_dataset("arbml/ashaar", split="train")
    poems, stats = D.poems_from_ashaar(ds)
    print(stats)
    return poems

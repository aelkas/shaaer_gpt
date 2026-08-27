"""Convert a training checkpoint into a safe, publishable pair of files.

The training checkpoints store the tokenizer as live Python objects, so they can
only be read back with `torch.load(..., weights_only=False)` — which unpickles
arbitrary code. That is fine for your own files and a bad thing to ask strangers
on the internet to do. This splits each checkpoint into:

    <name>.safetensors   tensors only, no pickle, no code execution
    <name>.json          config + tokenizer vocab and merges

Usage:
    python export_checkpoint.py gpt_B_plain_bpe_ashaar.pt --out dist/
"""
import argparse, json, os
import torch
from safetensors.torch import save_file

META = ("cell", "kind", "tashkeel", "vocab_size", "n_embd", "n_head",
        "n_layer", "block_size", "bpc")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="dist")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.ckpt))[0]

    tensors = {k: v.contiguous() for k, v in ck["state_dict"].items()}
    save_file(tensors, os.path.join(args.out, stem + ".safetensors"))

    cfg = {k: ck[k] for k in META if k in ck}
    cfg["tok_vocab"] = list(ck["tok_vocab"])
    cfg["tok_merges"] = [list(m) for m in ck.get("tok_merges", [])]
    # Record what the checkpoints should have recorded all along (see README 5.2.1)
    for k in ("iters", "batch", "n_train_tokens", "n_chars", "chars_per_token"):
        if k in ck:
            cfg[k] = ck[k]
        else:
            cfg.setdefault("_unrecorded", []).append(k)

    with open(os.path.join(args.out, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)

    n = sum(v.numel() for v in tensors.values())
    print(f"{stem}: {n:,} params -> {args.out}/{stem}.safetensors + .json")
    if "_unrecorded" in cfg:
        print("  note: not recorded in this checkpoint:", ", ".join(cfg["_unrecorded"]))

if __name__ == "__main__":
    main()

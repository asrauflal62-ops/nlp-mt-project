import argparse
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Union

import torch
import sentencepiece as spm
import sacrebleu


# ---------------------------
# Dynamic import helpers
# ---------------------------
def load_module_from_path(py_path: str) -> ModuleType:
    import importlib.util

    py_path = os.path.abspath(py_path)
    name = "dyn_model_" + str(abs(hash(py_path)))
    spec = importlib.util.spec_from_file_location(name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def get_model_classes(model_py: str):
    mod = load_module_from_path(model_py)
    if not hasattr(mod, "TransformerConfig") or not hasattr(mod, "TransformerNMT"):
        raise RuntimeError(f"{model_py} must define TransformerConfig and TransformerNMT")
    return mod.TransformerConfig, mod.TransformerNMT


# ---------------------------
# Data helpers
# ---------------------------
def load_meta(data_dir: Path) -> Dict[str, int]:
    meta = json.load(open(data_dir / "meta.json", "r", encoding="utf-8"))
    return {
        "pad_id": int(meta["pad_id"]),
        "bos_id": int(meta["bos_id"]),
        "eos_id": int(meta["eos_id"]),
        "vocab_size": int(meta["vocab_size"]),
    }


def _to_1d_long_tensor(x: Any) -> torch.Tensor:
    """
    Convert list[int] / tuple[int] / torch.Tensor into 1D torch.LongTensor on CPU.
    """
    if isinstance(x, torch.Tensor):
        # allow shapes [T] or [1,T] etc; squeeze to [T]
        return x.detach().cpu().long().view(-1)
    if isinstance(x, (list, tuple)):
        return torch.tensor(list(x), dtype=torch.long)
    raise TypeError(f"Unsupported token container type: {type(x)}")


def load_pt(data_dir: Path, split: str) -> List[Dict[str, Union[List[int], torch.Tensor]]]:
    """
    Supports multiple .pt formats that may appear in different preprocess versions:
      1) dict: {"src": Tensor[N,S] or list, "tgt": Tensor[N,T] or list}
      2) list: each item is dict {"src": ..., "tgt": ...} or tuple/list (src, tgt)
    Returns a list of examples: [{"src": ..., "tgt": ...}, ...]
    NOTE: src/tgt may be list[int] or torch.Tensor; caller should convert to tensor.
    """
    pt = torch.load(data_dir / f"{split}.pt", map_location="cpu")

    # dict format
    if isinstance(pt, dict) and "src" in pt and "tgt" in pt:
        src = pt["src"]
        tgt = pt["tgt"]

        # Tensor case: shape [N, L]
        if isinstance(src, torch.Tensor) and src.dim() >= 2:
            n = src.size(0)
            return [{"src": src[i], "tgt": tgt[i]} for i in range(n)]

        # List-of-seqs case
        if isinstance(src, list):
            n = len(src)
            return [{"src": src[i], "tgt": tgt[i]} for i in range(n)]

        # Fallback: treat as iterable
        try:
            n = len(src)
            return [{"src": src[i], "tgt": tgt[i]} for i in range(n)]
        except Exception as e:
            raise ValueError(f"Unsupported dict pt format for src/tgt: {type(src)} / {type(tgt)}") from e

    # list format
    if isinstance(pt, list):
        out: List[Dict[str, Any]] = []
        for it in pt:
            if isinstance(it, dict):
                out.append({"src": it["src"], "tgt": it["tgt"]})
            else:
                # tuple/list: (src, tgt)
                out.append({"src": it[0], "tgt": it[1]})
        return out

    raise ValueError("Unsupported pt format")


def ids_to_text(
    sp: spm.SentencePieceProcessor,
    ids: Union[List[int], torch.Tensor],
    bos_id: int,
    eos_id: int,
    pad_id: int,
) -> str:
    if isinstance(ids, torch.Tensor):
        ids_list = ids.detach().cpu().tolist()
    else:
        ids_list = list(ids)

    clean = []
    for x in ids_list:
        x = int(x)
        if x in (bos_id, pad_id):
            continue
        if x == eos_id:
            break
        clean.append(x)
    return sp.decode(clean)


@torch.no_grad()
def beam_search(model, src: torch.Tensor, max_len: int, beam: int, alpha: float) -> List[int]:
    """
    Beam search for a single example (src shape: [1, S]).
    length penalty: lp = ((5+len)/6)^alpha
    """
    cfg = model.cfg
    device = src.device
    model.eval()

    src_pad = (src == cfg.pad_id)
    memory = model.encode(src, src_key_padding_mask=src_pad)

    beams = [([cfg.bos_id], 0.0, False)]

    def lp(L: int) -> float:
        return ((5.0 + L) / 6.0) ** alpha

    for _ in range(max_len):
        cand = []
        for tokens, score, ended in beams:
            if ended:
                cand.append((tokens, score, ended))
                continue

            ys = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1,T)
            tgt_pad = (ys == cfg.pad_id)
            dec = model.decode(
                tgt_in=ys,
                memory=memory,
                tgt_key_padding_mask=tgt_pad,
                src_key_padding_mask=src_pad,
            )
            logits = model.lm_head(dec[:, -1, :])  # (1,V)
            logp = torch.log_softmax(logits, dim=-1).squeeze(0)

            topk = torch.topk(logp, k=beam)
            for lpv, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                ntok = tokens + [idx]
                nscore = score + float(lpv)
                nend = (idx == cfg.eos_id)
                cand.append((ntok, nscore, nend))

        cand.sort(key=lambda x: x[1] / lp(len(x[0])), reverse=True)
        beams = cand[:beam]
        if all(e for _, _, e in beams):
            break

    best = max(beams, key=lambda x: x[1] / lp(len(x[0])))
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="best.pt")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--split", type=str, choices=["valid", "test"], required=True)

    ap.add_argument("--mode", type=str, choices=["greedy", "beam"], default="greedy")
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.6)

    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--num_show", type=int, default=10)

    # optional override; if not given, will use run_dir/config.json's model_py; otherwise fallback baseline
    ap.add_argument("--model_py", type=str, default=None)

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)
    meta = load_meta(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_json = {}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        cfg_json = json.load(open(cfg_path, "r", encoding="utf-8"))

    model_py = args.model_py or cfg_json.get("model_py", "model_transformer.py")
    TransformerConfig, TransformerNMT = get_model_classes(model_py)

    cfg = TransformerConfig(
        vocab_size=meta["vocab_size"],
        pad_id=meta["pad_id"],
        bos_id=meta["bos_id"],
        eos_id=meta["eos_id"],
        d_model=int(cfg_json.get("d_model", 512)),
        nhead=int(cfg_json.get("nhead", 8)),
        num_encoder_layers=int(cfg_json.get("enc_layers", 6)),
        num_decoder_layers=int(cfg_json.get("dec_layers", 6)),
        dim_feedforward=int(cfg_json.get("ffn", 2048)),
        dropout=float(cfg_json.get("dropout", 0.1)),
        max_len=int(cfg_json.get("max_len", 128)),
    )

    model = TransformerNMT(cfg).to(device)
    ckpt = torch.load(run_dir / args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.load(str(data_dir / "spm.model"))

    data = load_pt(data_dir, args.split)

    mode_tag = "greedy" if args.mode == "greedy" else f"beam{args.beam}_a{args.alpha}"
    out_dir = run_dir / f"eval_{args.split}_{mode_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    hyps, refs = [], []

    for ex in data:
        # ---- FIX: src/tgt may be list[int] OR torch.Tensor ----
        src_1d = _to_1d_long_tensor(ex["src"])
        tgt_1d = _to_1d_long_tensor(ex["tgt"])

        src = src_1d.unsqueeze(0).to(device)  # [1, S]
        tgt_ids = tgt_1d.tolist()

        if args.mode == "greedy":
            out_ids = model.greedy_decode(src, max_len=args.max_len)[0].detach().cpu().tolist()
        else:
            out_ids = beam_search(
                model, src, max_len=args.max_len, beam=args.beam, alpha=args.alpha
            )

        hyp = ids_to_text(sp, out_ids, meta["bos_id"], meta["eos_id"], meta["pad_id"])
        ref = ids_to_text(sp, tgt_ids, meta["bos_id"], meta["eos_id"], meta["pad_id"])

        hyps.append(hyp)
        refs.append(ref)

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score  # 0..100

    pred_path = out_dir / "pred.txt"
    ref_path = out_dir / "ref.txt"
    samp_path = out_dir / "samples.txt"

    pred_path.write_text("\n".join(hyps), encoding="utf-8")
    ref_path.write_text("\n".join(refs), encoding="utf-8")

    lines = []
    for i in range(min(args.num_show, len(hyps))):
        lines.append(f"[{i}] HYP: {hyps[i]}")
        lines.append(f"[{i}] REF: {refs[i]}")
        lines.append("")
    samp_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        f"{run_dir.name} | ckpt-{args.ckpt} | split={args.split} | mode={mode_tag} | BLEU = {bleu:.2f}"
    )
    print(f"saved: {pred_path}")
    print(f"saved: {ref_path}")
    print(f"saved: {samp_path}")


if __name__ == "__main__":
    main()

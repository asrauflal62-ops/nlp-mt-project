# decode_rnn.py
import os
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sentencepiece as spm
import sacrebleu

from dataset import Seq2SeqDataset, collate_fn
from model_rnn import EncoderGRU, DecoderGRU, DecoderAttnGRU, Seq2Seq


def load_config(run_dir: Path):
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    return json.load(open(cfg_path, "r", encoding="utf-8"))


def build_model_from_config(cfg: dict):
    V = int(cfg["vocab_size"])
    pad_id = int(cfg["pad_id"])
    emb = int(cfg["emb"])
    hid = int(cfg["hid"])
    layers = int(cfg["layers"])
    dropout = float(cfg["dropout"])
    attn = cfg.get("attn", "none")

    enc = EncoderGRU(V, emb, hid, layers, dropout, pad_id)

    # ✅ 关键：只要不是 none，就用 attention decoder，并把 attn_type 传进去
    if attn != "none":
        dec = DecoderAttnGRU(V, emb, hid, layers, dropout, pad_id, attn_type=attn)
    else:
        dec = DecoderGRU(V, emb, hid, layers, dropout, pad_id)

    model = Seq2Seq(enc, dec, pad_id)
    return model, attn


def ids_to_text(sp, ids):
    # ids: List[int]
    return sp.decode(ids)


@torch.no_grad()
def run_greedy_decode(
    model: Seq2Seq,
    loader: DataLoader,
    sp,
    bos_id: int,
    eos_id: int,
    max_len: int,
    device: torch.device,
):
    preds = []
    refs = []
    samples = []

    for batch in loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        # greedy decode: returns List[List[int]] (token ids without eos)
        out_ids = model.greedy_decode(src, bos_id=bos_id, eos_id=eos_id, max_len=max_len)

        # refs: tgt[:, 1:] up to eos
        tgt_ids = tgt[:, 1:].tolist()

        for i in range(len(out_ids)):
            hyp_ids = out_ids[i]

            ref_seq = tgt_ids[i]
            # cut at eos if exists
            if eos_id in ref_seq:
                ref_seq = ref_seq[:ref_seq.index(eos_id)]

            hyp_txt = ids_to_text(sp, hyp_ids)
            ref_txt = ids_to_text(sp, ref_seq)

            preds.append(hyp_txt)
            refs.append(ref_txt)
            samples.append((hyp_txt, ref_txt))

    return preds, refs, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="best.pt")
    ap.add_argument("--data_dir", type=str, default="Data/processed/spm16k")
    ap.add_argument("--split", type=str, choices=["valid", "test"], default="valid")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--num_show", type=int, default=10)

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)
    ckpt_path = run_dir / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    cfg = load_config(run_dir)

    sp_path = data_dir / "spm.model"
    if not sp_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {sp_path}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_path))

    bos_id = int(cfg["bos_id"])
    eos_id = int(cfg["eos_id"])
    pad_id = int(cfg["pad_id"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, attn = build_model_from_config(cfg)
    model.to(device)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    # dataset
    pt_name = "valid.pt" if args.split == "valid" else "test.pt"
    ds = Seq2SeqDataset(str(data_dir / pt_name))
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=2,
        pin_memory=True,
    )

    preds, refs, samples = run_greedy_decode(
        model=model,
        loader=loader,
        sp=sp,
        bos_id=bos_id,
        eos_id=eos_id,
        max_len=args.max_len,
        device=device,
    )

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score

    out_dir = run_dir / f"eval_{args.split}_greedy"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "pred.txt").write_text("\n".join(preds), encoding="utf-8")
    (out_dir / "ref.txt").write_text("\n".join(refs), encoding="utf-8")

    # save samples for qualitative analysis
    show_n = min(args.num_show, len(samples))
    lines = []
    for i in range(show_n):
        hyp, ref = samples[i]
        lines.append(f"[{i}] HYP: {hyp}")
        lines.append(f"[{i}] REF: {ref}")
        lines.append("")
    (out_dir / "samples.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"{run_dir.name} | ckpt-{args.ckpt} | attn={attn} | split={args.split} | BLEU = {bleu:.2f}")
    print(f"saved: {out_dir}/pred.txt")
    print(f"saved: {out_dir}/ref.txt")
    print(f"saved: {out_dir}/samples.txt")


if __name__ == "__main__":
    main()

# decode_rnn.py
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
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

    # 只要不是 none，就用 attention decoder，并把 attn_type 传进去
    if attn != "none":
        dec = DecoderAttnGRU(V, emb, hid, layers, dropout, pad_id, attn_type=attn)
    else:
        dec = DecoderGRU(V, emb, hid, layers, dropout, pad_id)

    model = Seq2Seq(enc, dec, pad_id)
    return model, attn


def ids_to_text(sp, ids):
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

        tgt_ids = tgt[:, 1:].tolist()
        for i in range(len(out_ids)):
            hyp_ids = out_ids[i]
            ref_seq = tgt_ids[i]
            if eos_id in ref_seq:
                ref_seq = ref_seq[:ref_seq.index(eos_id)]

            preds.append(ids_to_text(sp, hyp_ids))
            refs.append(ids_to_text(sp, ref_seq))
            samples.append((preds[-1], refs[-1]))

    return preds, refs, samples


@torch.no_grad()
def _beam_search_one(
    model: Seq2Seq,
    src_1: torch.Tensor,   # (S,)
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
    beam_size: int = 5,
    alpha: float = 0.6,    # length norm; 0 = off
):
    """
    单句 beam search（稳定优先；valid/test 很小，逐条解码足够快）
    返回：List[int]（不含 BOS；遇到 EOS 截断）
    """
    device = src_1.device
    src_1 = src_1.unsqueeze(0)  # (1,S)

    # encode
    enc_out, enc_h = model.encoder(src_1)
    dec_h0 = enc_h
    src_mask = (src_1 != pad_id)  # (1,S)

    def step(prev_token_id: int, dec_h):
        prev = torch.tensor([prev_token_id], device=device, dtype=torch.long)  # (1,)
        if isinstance(model.decoder, DecoderAttnGRU):
            logits, new_h, _attn = model.decoder.forward_step(prev, dec_h, enc_out, src_mask)
        else:
            logits, new_h = model.decoder.forward_step(prev, dec_h)
        logp = F.log_softmax(logits.squeeze(0), dim=-1)  # (V,)
        return logp, new_h

    # beam elements: (tokens[List[int]], sum_logp[float], hidden, ended[bool])
    beams = [([bos_id], 0.0, dec_h0, False)]

    for _ in range(max_len):
        candidates = []
        for tokens, score, dec_h, ended in beams:
            if ended:
                candidates.append((tokens, score, dec_h, True))
                continue

            logp, new_h = step(tokens[-1], dec_h)
            topk_logp, topk_ids = torch.topk(logp, k=beam_size)

            for lp, tid in zip(topk_logp.tolist(), topk_ids.tolist()):
                ntoks = tokens + [tid]
                nscore = score + lp
                nended = (tid == eos_id)
                candidates.append((ntoks, nscore, new_h, nended))

        def rank_key(b):
            toks, s, _h, _e = b
            L = max(1, len(toks) - 1)  # 不把 BOS 算进去
            if alpha <= 0:
                return s
            return s / (L ** alpha)

        candidates.sort(key=rank_key, reverse=True)
        beams = candidates[:beam_size]

        if all(b[3] for b in beams):
            break

    best_tokens = beams[0][0]  # 含 BOS
    out = []
    for t in best_tokens[1:]:
        if t == eos_id:
            break
        if t == pad_id:
            continue
        out.append(t)
    return out


@torch.no_grad()
def run_beam_decode(
    model: Seq2Seq,
    loader: DataLoader,
    sp,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
    device: torch.device,
    beam_size: int,
    alpha: float,
):
    preds = []
    refs = []
    samples = []

    for batch in loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        tgt_ids = tgt[:, 1:].tolist()

        # 逐条 beam（稳）
        for i in range(src.size(0)):
            hyp_ids = _beam_search_one(
                model=model,
                src_1=src[i],
                bos_id=bos_id,
                eos_id=eos_id,
                pad_id=pad_id,
                max_len=max_len,
                beam_size=beam_size,
                alpha=alpha,
            )

            ref_seq = tgt_ids[i]
            if eos_id in ref_seq:
                ref_seq = ref_seq[:ref_seq.index(eos_id)]

            preds.append(ids_to_text(sp, hyp_ids))
            refs.append(ids_to_text(sp, ref_seq))
            samples.append((preds[-1], refs[-1]))

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

    # NEW: decoding policy
    ap.add_argument("--mode", type=str, default="greedy", choices=["greedy", "beam"])
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.6, help="length norm alpha (0 disables)")

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

    if args.mode == "greedy":
        preds, refs, samples = run_greedy_decode(
            model=model, loader=loader, sp=sp,
            bos_id=bos_id, eos_id=eos_id, max_len=args.max_len, device=device
        )
        out_dir = run_dir / f"eval_{args.split}_greedy"
        tag = "greedy"
    else:
        preds, refs, samples = run_beam_decode(
            model=model, loader=loader, sp=sp,
            bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
            max_len=args.max_len, device=device,
            beam_size=args.beam, alpha=args.alpha
        )
        out_dir = run_dir / f"eval_{args.split}_beam{args.beam}_a{args.alpha}"
        tag = f"beam{args.beam}_a{args.alpha}"

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pred.txt").write_text("\n".join(preds), encoding="utf-8")
    (out_dir / "ref.txt").write_text("\n".join(refs), encoding="utf-8")

    show_n = min(args.num_show, len(samples))
    lines = []
    for i in range(show_n):
        hyp, ref = samples[i]
        lines.append(f"[{i}] HYP: {hyp}")
        lines.append(f"[{i}] REF: {ref}")
        lines.append("")
    (out_dir / "samples.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"{run_dir.name} | ckpt-{args.ckpt} | attn={attn} | split={args.split} | mode={tag} | BLEU = {bleu:.2f}")
    print(f"saved: {out_dir}/pred.txt")
    print(f"saved: {out_dir}/ref.txt")
    print(f"saved: {out_dir}/samples.txt")


if __name__ == "__main__":
    main()

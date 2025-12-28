# train_rnn.py
import os, json, argparse, math, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Seq2SeqDataset, collate_fn
from model_rnn import EncoderGRU, DecoderGRU, DecoderAttnGRU, Seq2Seq


def save_ckpt(path, model, optim, epoch, step, config):
    """
    epoch: 推荐存“下一次要开始的epoch”（epoch+1），便于 resume 逻辑一致
    step: global step
    """
    tmp = str(path) + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": config,
    }, tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, optim):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optim"])
    return ckpt["epoch"], ckpt["step"], ckpt.get("config", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="Data/processed/spm16k")
    ap.add_argument("--train_pt", type=str, default="train_10k.pt")
    ap.add_argument("--valid_pt", type=str, default="valid.pt")
    ap.add_argument("--run_dir", type=str, default="runs/rnn_gru_baseline")

    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--hid", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--tf_ratio", type=float, default=1.0)

    # ✅ 扩展：attention 类型（none / additive(Bahdanau) / dot / general）
    ap.add_argument("--attn", type=str, default="none",
                    choices=["none", "bahdanau", "dot", "general"])

    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--save_every", type=int, default=500)  # steps
    ap.add_argument("--resume", type=str, default="")

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = json.load(open(data_dir / "meta.json", "r", encoding="utf-8"))
    V = meta["vocab_size"]
    pad_id = meta["pad_id"]

    train_ds = Seq2SeqDataset(str(data_dir / args.train_pt))
    valid_ds = Seq2SeqDataset(str(data_dir / args.valid_pt))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=2, pin_memory=True
    )

    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=2, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = EncoderGRU(V, args.emb, args.hid, args.layers, args.dropout, pad_id)

    # ✅ 按开关选择 decoder
    if args.attn != "none":
        # 注意：attn_type 取 args.attn (bahdanau/dot/general)
        dec = DecoderAttnGRU(V, args.emb, args.hid, args.layers, args.dropout, pad_id, attn_type=args.attn)
    else:
        dec = DecoderGRU(V, args.emb, args.hid, args.layers, args.dropout, pad_id)

    model = Seq2Seq(enc, dec, pad_id).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    start_epoch, global_step = 0, 0
    if args.resume and Path(args.resume).exists():
        start_epoch, global_step, _ = load_ckpt(args.resume, model, optim)
        print(f"[RESUME] epoch={start_epoch} step={global_step} from {args.resume}")

    # ✅ config 里记录本次实验参数（包括 attn），decode 会用到
    config = (vars(args) | meta)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 读取历史 best（若存在）
    best_path = run_dir / "best.pt"
    best_loss_path = run_dir / "best_loss.txt"
    best = float("inf")
    if best_loss_path.exists():
        try:
            best = float(best_loss_path.read_text().strip())
        except Exception:
            best = float("inf")

    @torch.no_grad()
    def evaluate(model, loader):
        was_training = model.training
        model.eval()

        total_loss, total_tok = 0.0, 0
        for batch in loader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            # valid 固定 TF=1.0，保证可比
            logits = model(src, tgt, teacher_forcing_ratio=1.0)  # (B, T-1, V)
            labels = tgt[:, 1:].contiguous()

            loss = criterion(logits.view(-1, V), labels.view(-1))

            ntok = (labels != pad_id).sum().item()
            total_loss += loss.item() * ntok
            total_tok += ntok

        avg_loss = total_loss / max(1, total_tok)
        ppl = math.exp(min(20, avg_loss))

        if was_training:
            model.train()
        return avg_loss, ppl

    model.train()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss_tok_sum, train_tok_sum = 0.0, 0

        for batch in train_loader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            logits = model(src, tgt, teacher_forcing_ratio=args.tf_ratio)  # (B,T-1,V)
            labels = tgt[:, 1:].contiguous()

            loss = criterion(logits.view(-1, V), labels.view(-1))

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optim.step()

            global_step += 1

            ntok = (labels != pad_id).sum().item()
            train_loss_tok_sum += loss.item() * ntok
            train_tok_sum += ntok

            if global_step % 50 == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item():.4f}")

            if args.save_every > 0 and global_step % args.save_every == 0:
                save_ckpt(run_dir / "last.pt", model, optim, epoch + 1, global_step, config)

        save_ckpt(run_dir / "last.pt", model, optim, epoch + 1, global_step, config)

        vloss, vppl = evaluate(model, valid_loader)

        flag = ""
        if vloss < best:
            best = vloss
            best_loss_path.write_text(str(best))
            save_ckpt(best_path, model, optim, epoch + 1, global_step, config)
            flag = " (best)"

        avg_train_loss = train_loss_tok_sum / max(1, train_tok_sum)
        train_ppl = math.exp(min(20, avg_train_loss))

        print(
            f"[EPOCH DONE] {epoch} time={time.time()-t0:.1f}s "
            f"train_loss={avg_train_loss:.4f} train_ppl={train_ppl:.2f} "
            f"valid_loss={vloss:.4f} valid_ppl={vppl:.2f} saved last.pt{flag}"
        )


if __name__ == "__main__":
    main()

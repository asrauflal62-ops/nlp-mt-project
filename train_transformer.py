import argparse
import json
import math
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------
# Dynamic import helpers
# ---------------------------
def load_module_from_path(py_path: str) -> ModuleType:
    """Load a python module from a file path (no package needed)."""
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
    """
    Expect model_py to provide:
      - TransformerConfig
      - TransformerNMT
    """
    mod = load_module_from_path(model_py)
    if not hasattr(mod, "TransformerConfig") or not hasattr(mod, "TransformerNMT"):
        raise RuntimeError(
            f"{model_py} must define TransformerConfig and TransformerNMT"
        )
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


def load_pt(data_dir: Path, pt_name: str):
    return torch.load(data_dir / pt_name, map_location="cpu")


def collate_batch(batch, pad_id: int):
    # batch items can be dict or tuple/list
    srcs, tgts = [], []
    for it in batch:
        if isinstance(it, dict):
            src, tgt = it["src"], it["tgt"]
        else:
            src, tgt = it[0], it[1]

        #  fix: ensure tensors
        if not torch.is_tensor(src):
            src = torch.tensor(src, dtype=torch.long)
        else:
            src = src.long()

        if not torch.is_tensor(tgt):
            tgt = torch.tensor(tgt, dtype=torch.long)
        else:
            tgt = tgt.long()

        srcs.append(src)
        tgts.append(tgt)

    src_lens = [int(x.numel()) for x in srcs]
    tgt_lens = [int(x.numel()) for x in tgts]
    max_s = max(src_lens)
    max_t = max(tgt_lens)

    src_pad = torch.full((len(srcs), max_s), pad_id, dtype=torch.long)
    tgt_pad = torch.full((len(tgts), max_t), pad_id, dtype=torch.long)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, : s.numel()] = s
        tgt_pad[i, : t.numel()] = t

    return src_pad, tgt_pad


# ---------------------------
# Loss / Scheduler
# ---------------------------
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, smoothing: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B*T, V)
        target: (B*T,)
        """
        # mask padding
        mask = target.ne(self.pad_id)
        if mask.sum() == 0:
            return logits.sum() * 0.0

        logits = logits[mask]
        target = target[mask]

        log_probs = torch.log_softmax(logits, dim=-1)  # (N,V)

        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)  # (N,)
        smooth = -log_probs.mean(dim=-1)  # (N,)
        loss = self.confidence * nll + self.smoothing * smooth
        return loss.mean()


def make_paper_lr_lambda(d_model: int, warmup: int):
    # Paper: lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    def lr_lambda(step: int):
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))

    return lr_lambda


# ---------------------------
# Train / Eval
# ---------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device, pad_id: int) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        # teacher forcing: input is tgt[:, :-1], predict tgt[:, 1:]
        tgt_in = tgt[:, :-1]
        tgt_y = tgt[:, 1:]

        logits = model(src, tgt_in)  # (B, T, V) or (B,T,V)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), tgt_y.reshape(B * T))

        # count non-pad tokens
        ntok = tgt_y.ne(pad_id).sum().item()
        total_loss += loss.item() * ntok
        total_tokens += ntok

    model.train()
    return total_loss / max(total_tokens, 1)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--train_pt", type=str, required=True)
    ap.add_argument("--valid_pt", type=str, required=True)
    ap.add_argument("--run_dir", type=str, required=True)

    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)

    # paper-style LR schedule (recommended)
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--lr_factor", type=float, default=1.0)  # scale of paper lr
    ap.add_argument("--adam_beta1", type=float, default=0.9)
    ap.add_argument("--adam_beta2", type=float, default=0.98)
    ap.add_argument("--adam_eps", type=float, default=1e-9)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)

    ap.add_argument("--label_smoothing", type=float, default=0.1)

    # model selection (IMPORTANT for 2x2)
    ap.add_argument(
        "--model_py",
        type=str,
        default="model_transformer.py",
        help="Path to model python file that defines TransformerConfig/TransformerNMT",
    )

    # allow overriding model dims from CLI if your model supports it via config fields
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--enc_layers", type=int, default=6)
    ap.add_argument("--dec_layers", type=int, default=6)
    ap.add_argument("--ffn", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=128)

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_dir = Path(args.data_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(data_dir)
    pad_id = meta["pad_id"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_pt = load_pt(data_dir, args.train_pt)
    valid_pt = load_pt(data_dir, args.valid_pt)

    # train_pt may be dict("src","tgt") tensors or list
    if isinstance(train_pt, dict) and "src" in train_pt and "tgt" in train_pt:
        train_data = [{"src": train_pt["src"][i], "tgt": train_pt["tgt"][i]} for i in range(len(train_pt["src"]))]
    else:
        train_data = train_pt

    if isinstance(valid_pt, dict) and "src" in valid_pt and "tgt" in valid_pt:
        valid_data = [{"src": valid_pt["src"][i], "tgt": valid_pt["tgt"][i]} for i in range(len(valid_pt["src"]))]
    else:
        valid_data = valid_pt

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_id),
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_id),
        num_workers=0,
    )

    # build model from selected python
    TransformerConfig, TransformerNMT = get_model_classes(args.model_py)
    cfg = TransformerConfig(
        vocab_size=meta["vocab_size"],
        pad_id=meta["pad_id"],
        bos_id=meta["bos_id"],
        eos_id=meta["eos_id"],
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ffn,
        dropout=args.dropout,
        max_len=args.max_len,
    )
    model = TransformerNMT(cfg).to(device)

    criterion = LabelSmoothingLoss(meta["vocab_size"], pad_id, args.label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr_factor,  # IMPORTANT: base lr=lr_factor, schedule provides magnitude
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    lr_lambda = make_paper_lr_lambda(args.d_model, args.warmup)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # save config for decoding reproducibility
    config_json = {
        "model_py": os.path.abspath(args.model_py),
        "d_model": args.d_model,
        "nhead": args.nhead,
        "enc_layers": args.enc_layers,
        "dec_layers": args.dec_layers,
        "ffn": args.ffn,
        "dropout": args.dropout,
        "max_len": args.max_len,
        "warmup": args.warmup,
        "lr_factor": args.lr_factor,
        "label_smoothing": args.label_smoothing,
        "pad_id": meta["pad_id"],
        "bos_id": meta["bos_id"],
        "eos_id": meta["eos_id"],
        "vocab_size": meta["vocab_size"],
    }
    (run_dir / "config.json").write_text(json.dumps(config_json, indent=2), encoding="utf-8")

    best_valid = float("inf")
    global_step = 0

    for ep in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for src, tgt in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_in = tgt[:, :-1]
            tgt_y = tgt[:, 1:]

            optimizer.zero_grad(set_to_none=True)
            logits = model(src, tgt_in)  # (B,T,V)
            B, T, V = logits.shape

            loss = criterion(logits.reshape(B * T, V), tgt_y.reshape(B * T))
            loss.backward()

            if args.clip is not None and args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()
            global_step += 1
            scheduler.step()

            ntok = tgt_y.ne(pad_id).sum().item()
            total_loss += loss.item() * ntok
            total_tokens += ntok

            if global_step % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"epoch {ep} step {global_step} loss {loss.item():.4f} lr {lr:.6g}")

        train_loss = total_loss / max(total_tokens, 1)
        train_ppl = math.exp(min(train_loss, 20))

        valid_loss = evaluate(model, valid_loader, criterion, device, pad_id)
        valid_ppl = math.exp(min(valid_loss, 20))

        # checkpoint
        ckpt = {
            "model": model.state_dict(),
            "cfg": config_json,
            "epoch": ep,
            "global_step": global_step,
        }
        torch.save(ckpt, run_dir / "last.pt")

        tag = ""
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(ckpt, run_dir / "best.pt")
            tag = " (best)"

        print(
            f"[EPOCH DONE] {ep} train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} "
            f"valid_loss={valid_loss:.4f} valid_ppl={valid_ppl:.2f} saved last.pt{tag}"
        )


if __name__ == "__main__":
    main()

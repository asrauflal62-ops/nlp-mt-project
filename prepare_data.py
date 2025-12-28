# prepare_data.py
import os, json, argparse, random
from pathlib import Path

import torch

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            yield obj

def write_text_for_spm(paths, out_txt):
    # 将中英都写到一个文本里训练同一个 SentencePiece
    with open(out_txt, "w", encoding="utf-8") as w:
        for p in paths:
            for obj in read_jsonl(p):
                zh = obj["zh"].strip()
                en = obj["en"].strip()
                if zh:
                    w.write(zh.replace("\n", " ") + "\n")
                if en:
                    w.write(en.replace("\n", " ") + "\n")

def train_sentencepiece(corpus_txt, out_dir, vocab_size=16000, model_type="unigram"):
    import sentencepiece as spm
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / "spm")
    spm.SentencePieceTrainer.Train(
        input=str(corpus_txt),
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        bos_id=1, eos_id=2, pad_id=0, unk_id=3,
        user_defined_symbols=[]
    )
    return out_dir / "spm.model", out_dir / "spm.vocab"

def encode_split(jsonl_path, sp, max_src_len, max_tgt_len):
    src_list, tgt_list = [], []
    for obj in read_jsonl(jsonl_path):
        zh = obj["zh"].strip().replace("\n", " ")
        en = obj["en"].strip().replace("\n", " ")
        if not zh or not en:
            continue

        src_ids = sp.EncodeAsIds(zh)[:max_src_len]
        tgt_ids = sp.EncodeAsIds(en)[:max_tgt_len]

        # decoder 需要 BOS/EOS
        src_list.append(src_ids + [sp.eos_id()])  # 源句也加 eos，方便后续一致处理
        tgt_list.append([sp.bos_id()] + tgt_ids + [sp.eos_id()])

    return {"src": src_list, "tgt": tgt_list}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw")
    ap.add_argument("--out_dir", type=str, default="data/processed/spm16k")
    ap.add_argument("--vocab_size", type=int, default=16000)
    ap.add_argument("--max_src_len", type=int, default=100)
    ap.add_argument("--max_tgt_len", type=int, default=100)
    ap.add_argument("--train_spm_on", type=str, default="train_10k.jsonl")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_spm_file = raw_dir / args.train_spm_on
    assert train_spm_file.exists(), f"Missing: {train_spm_file}"

    corpus_txt = out_dir / "spm_corpus.txt"
    if not (out_dir / "spm.model").exists():
        print(f"[SPM] building corpus: {corpus_txt}")
        # 你也可以把 valid/test 加进去，但一般只用 train 就行
        write_text_for_spm([train_spm_file], corpus_txt)
        print("[SPM] training...")
        train_sentencepiece(corpus_txt, out_dir, vocab_size=args.vocab_size)
        print("[SPM] done.")
    else:
        print("[SPM] found existing spm.model, skip training.")

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(str(out_dir / "spm.model"))

    for split in ["train_10k.jsonl", "train_100k.jsonl", "valid.jsonl", "test.jsonl"]:
        src = raw_dir / split
        if not src.exists():
            print(f"[WARN] skip missing: {src}")
            continue

        print(f"[ENCODE] {split}")
        data = encode_split(src, sp, args.max_src_len, args.max_tgt_len)
        out_pt = out_dir / split.replace(".jsonl", ".pt")
        torch.save(data, out_pt)
        print(f"[SAVE] {out_pt} | n={len(data['src'])}")

    meta = {
        "vocab_size": sp.GetPieceSize(),
        "pad_id": sp.pad_id(),
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
        "unk_id": sp.unk_id(),
        "max_src_len": args.max_src_len,
        "max_tgt_len": args.max_tgt_len,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[META] saved meta.json")

if __name__ == "__main__":
    main()

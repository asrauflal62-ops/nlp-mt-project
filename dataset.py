# dataset.py
import torch
from torch.utils.data import Dataset

class Seq2SeqDataset(Dataset):
    def __init__(self, pt_path: str):
        obj = torch.load(pt_path)
        self.src = obj["src"]
        self.tgt = obj["tgt"]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

def collate_fn(batch, pad_id: int):
    src_seqs, tgt_seqs = zip(*batch)
    bs = len(src_seqs)
    src_max = max(len(x) for x in src_seqs)
    tgt_max = max(len(x) for x in tgt_seqs)

    src = torch.full((bs, src_max), pad_id, dtype=torch.long)
    tgt = torch.full((bs, tgt_max), pad_id, dtype=torch.long)
    src_len = torch.tensor([len(x) for x in src_seqs], dtype=torch.long)
    tgt_len = torch.tensor([len(x) for x in tgt_seqs], dtype=torch.long)

    for i,(s,t) in enumerate(zip(src_seqs, tgt_seqs)):
        src[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        tgt[i, :len(t)] = torch.tensor(t, dtype=torch.long)

    return {"src": src, "tgt": tgt, "src_len": src_len, "tgt_len": tgt_len}

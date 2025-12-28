# model_transformer_rel_rms.py
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int

    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_len: int = 512


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


def _build_rope_cache(seq_len: int, head_dim: int, device: torch.device, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, H, T, Dh = x.shape
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)


class MultiheadSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        cos, sin = _build_rope_cache(T, self.head_dim, x.device)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiheadSelfAttentionRoPE(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, src_key_padding_mask):
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, attn_mask=None, key_padding_mask=src_key_padding_mask))
        h = self.norm2(x)
        x = x + self.drop(self.linear2(self.drop(self.act(self.linear1(h)))))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiheadSelfAttentionRoPE(d_model, nhead, dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)

    def forward(self, y, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        h = self.norm1(y)
        y = y + self.drop(self.self_attn(h, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask))

        h = self.norm2(y)
        cross, _ = self.multihead_attn(h, memory, memory, key_padding_mask=memory_key_padding_mask, need_weights=False)
        y = y + self.drop(cross)

        h = self.norm3(y)
        y = y + self.drop(self.linear2(self.drop(self.act(self.linear1(h)))))
        return y


class Encoder(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout)
            for _ in range(cfg.num_encoder_layers)
        ])

    def forward(self, x, src_key_padding_mask):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout)
            for _ in range(cfg.num_decoder_layers)
        ])

    def forward(self, y, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        for layer in self.layers:
            y = layer(y, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return y


class TransformerNMT(nn.Module):
    """
    Relative position (RoPE) + RMSNorm.
    """
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.src_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.tgt_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.drop = nn.Dropout(cfg.dropout)
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        nn.init.normal_(self.src_emb.weight, mean=0.0, std=self.cfg.d_model ** -0.5)
        nn.init.normal_(self.tgt_emb.weight, mean=0.0, std=self.cfg.d_model ** -0.5)
        nn.init.xavier_uniform_(self.lm_head.weight)

    def encode(self, src, src_key_padding_mask):
        x = self.drop(self.src_emb(src) * math.sqrt(self.cfg.d_model))
        return self.encoder(x, src_key_padding_mask)

    def decode(self, tgt_in, memory, tgt_key_padding_mask, src_key_padding_mask):
        y = self.drop(self.tgt_emb(tgt_in) * math.sqrt(self.cfg.d_model))
        T = tgt_in.size(1)
        tgt_mask = torch.triu(torch.ones(T, T, device=tgt_in.device), diagonal=1).bool()
        return self.decoder(y, memory, tgt_mask, tgt_key_padding_mask, src_key_padding_mask)

    def forward(self, src, tgt_in):
        src_pad = (src == self.cfg.pad_id)
        tgt_pad = (tgt_in == self.cfg.pad_id)
        mem = self.encode(src, src_pad)
        dec = self.decode(tgt_in, mem, tgt_pad, src_pad)
        return self.lm_head(dec)

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, max_len: int = 128):
        self.eval()
        device = src.device
        src_pad = (src == self.cfg.pad_id)
        mem = self.encode(src, src_pad)

        ys = torch.full((src.size(0), 1), self.cfg.bos_id, dtype=torch.long, device=device)
        ended = torch.zeros(src.size(0), dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_pad = (ys == self.cfg.pad_id)
            dec = self.decode(ys, mem, tgt_pad, src_pad)
            logits = self.lm_head(dec[:, -1, :])
            next_tok = torch.argmax(logits, dim=-1)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            ended |= (next_tok == self.cfg.eos_id)
            if bool(ended.all()):
                break
        return ys

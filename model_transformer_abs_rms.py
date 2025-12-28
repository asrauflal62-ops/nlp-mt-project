# model_transformer_abs_rms.py
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # x: (..., D)
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1), :]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, src_key_padding_mask: Optional[torch.Tensor]):
        # pre-norm (common when using RMSNorm)
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, key_padding_mask=src_key_padding_mask, need_weights=False)
        x = x + self.drop(attn_out)

        h = self.norm2(x)
        ff = self.linear2(self.drop(self.act(self.linear1(h))))
        x = x + self.drop(ff)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)

    def forward(
        self,
        y,
        memory,
        tgt_mask: Optional[torch.Tensor],
        tgt_key_padding_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
    ):
        h = self.norm1(y)
        self_out, _ = self.self_attn(
            h, h, h,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False
        )
        y = y + self.drop(self_out)

        h = self.norm2(y)
        cross, _ = self.multihead_attn(
            h, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False
        )
        y = y + self.drop(cross)

        h = self.norm3(y)
        ff = self.linear2(self.drop(self.act(self.linear1(h))))
        y = y + self.drop(ff)
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
    Absolute positional encoding + RMSNorm.
    """
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.src_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.tgt_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len)
        self.drop = nn.Dropout(cfg.dropout)

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.src_emb.weight, mean=0.0, std=self.cfg.d_model ** -0.5)
        nn.init.normal_(self.tgt_emb.weight, mean=0.0, std=self.cfg.d_model ** -0.5)
        nn.init.xavier_uniform_(self.lm_head.weight)

    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor):
        x = self.src_emb(src) * math.sqrt(self.cfg.d_model)
        x = x + self.pos_enc(x)
        x = self.drop(x)
        return self.encoder(x, src_key_padding_mask)

    def decode(self, tgt_in, memory, tgt_key_padding_mask, src_key_padding_mask):
        y = self.tgt_emb(tgt_in) * math.sqrt(self.cfg.d_model)
        y = y + self.pos_enc(y)
        y = self.drop(y)

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

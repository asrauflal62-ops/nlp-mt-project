# model_transformer_abs_ln.py
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


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return self.pe[:, : x.size(1), :]


class TransformerNMT(nn.Module):
    """
    Absolute positional encoding + LayerNorm (baseline-style).
    Keys layout matches your current baseline style: encoder.layers.* / decoder.layers.*
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.src_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.tgt_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len)
        self.drop = nn.Dropout(cfg.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=False,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_encoder_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.num_decoder_layers)

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
        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
    ):
        y = self.tgt_emb(tgt_in) * math.sqrt(self.cfg.d_model)
        y = y + self.pos_enc(y)
        y = self.drop(y)

        T = tgt_in.size(1)
        tgt_mask = torch.triu(torch.ones(T, T, device=tgt_in.device), diagonal=1).bool()

        out = self.decoder(
            y,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return out

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor):
        src_pad = (src == self.cfg.pad_id)
        tgt_pad = (tgt_in == self.cfg.pad_id)
        mem = self.encode(src, src_key_padding_mask=src_pad)
        dec = self.decode(tgt_in, mem, tgt_key_padding_mask=tgt_pad, src_key_padding_mask=src_pad)
        logits = self.lm_head(dec)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, max_len: int = 128):
        self.eval()
        device = src.device
        src_pad = (src == self.cfg.pad_id)
        mem = self.encode(src, src_key_padding_mask=src_pad)

        ys = torch.full((src.size(0), 1), self.cfg.bos_id, dtype=torch.long, device=device)
        ended = torch.zeros(src.size(0), dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_pad = (ys == self.cfg.pad_id)
            dec = self.decode(ys, mem, tgt_key_padding_mask=tgt_pad, src_key_padding_mask=src_pad)
            logits = self.lm_head(dec[:, -1, :])
            next_tok = torch.argmax(logits, dim=-1)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            ended |= (next_tok == self.cfg.eos_id)
            if bool(ended.all()):
                break
        return ys

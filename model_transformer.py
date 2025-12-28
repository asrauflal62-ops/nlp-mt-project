"""
Transformer-based NMT model (from scratch) for Chinese-English translation.

This file is designed to work with the provided:
  - train_transformer.py
  - decode_transformer.py

Key conventions (matching task1 style):
  * Dataset provides tensors:
      src: LongTensor (B, S)  with <pad> padding
      tgt: LongTensor (B, T)  with <bos> ... <eos> and <pad> padding
  * forward(src, tgt) returns:
      logits: (B, T-1, V)   for predicting tgt[:, 1:]
      labels: (B, T-1)      equals tgt[:, 1:]
  * encode/decode are exposed for decoding scripts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import math
import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int

    # Transformer-Base defaults
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # absolute positional encoding length
    max_len: int = 512

    # weight tying (common for seq2seq with shared vocab)
    tie_embeddings: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard absolute sinusoidal positional encoding.
    Returns a (1, L, D) tensor added to token embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        L = x.size(1)
        return self.pe[:, :L, :]


def causal_mask_bool(L: int, device: torch.device) -> torch.Tensor:
    """
    Bool causal mask for self-attention.

    True = masked (disallowed)
    False = allowed

    Shape: (L, L)
    """
    return torch.triu(torch.ones((L, L), dtype=torch.bool, device=device), diagonal=1)


class TransformerNMT(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len)
        self.drop = nn.Dropout(cfg.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=False,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=False,
            activation="relu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.num_decoder_layers)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # Share token embedding & output projection weights
            self.lm_head.weight = self.tok_emb.weight

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Match common Transformer initialization style
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) Long
        emb = self.tok_emb(x) * math.sqrt(self.cfg.d_model)  # scale
        emb = emb + self.pos_enc(emb)
        return self.drop(emb)

    @torch.no_grad()
    def make_src_key_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        # True means "ignore" for key padding mask in PyTorch
        return src.eq(self.cfg.pad_id)

    @torch.no_grad()
    def make_tgt_key_padding_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        return tgt.eq(self.cfg.pad_id)

    def encode(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        src: (B, S) Long
        src_key_padding_mask: (B, S) bool, True=pad
        returns memory: (B, S, D)
        """
        if src_key_padding_mask is None:
            src_key_padding_mask = self.make_src_key_padding_mask(src)
        x = self._embed(src)
        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        tgt_in: (B, T) Long, typically starts with <bos>
        memory: (B, S, D) encoder outputs
        returns dec_out: (B, T, D)
        """
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self.make_tgt_key_padding_mask(tgt_in)

        # IMPORTANT: make attn_mask and key_padding_mask same "type family" (both bool),
        # to avoid PyTorch warning about mismatched mask types.
        T = tgt_in.size(1)
        tgt_mask = causal_mask_bool(T, device=tgt_in.device)  # (T, T) bool

        y = self._embed(tgt_in)
        dec_out = self.decoder(
            y,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return dec_out

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        Teacher-forcing training:
          input  = tgt[:, :-1]
          labels = tgt[:, 1:]
        """
        assert tgt.dim() == 2 and src.dim() == 2, "Expect src/tgt shape (B, L)"

        src_pad = self.make_src_key_padding_mask(src)
        memory = self.encode(src, src_key_padding_mask=src_pad)

        tgt_in = tgt[:, :-1].contiguous()
        labels = tgt[:, 1:].contiguous()

        tgt_pad = self.make_tgt_key_padding_mask(tgt_in)
        dec_out = self.decode(
            tgt_in=tgt_in,
            memory=memory,
            tgt_key_padding_mask=tgt_pad,
            src_key_padding_mask=src_pad,
        )
        logits = self.lm_head(dec_out)  # (B, T-1, V)
        return logits, labels

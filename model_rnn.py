# model_rnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=2, dropout=0.2, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, src, src_len=None):
        x = self.emb(src)        # (B,S,E)
        out, h = self.rnn(x)     # out: (B,S,H), h: (L,B,H)
        return out, h


class DecoderGRU(nn.Module):
    """Baseline decoder: no attention."""
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=2, dropout=0.2, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward_step(self, input_token, hidden):
        x = self.emb(input_token).unsqueeze(1)  # (B,1,E)
        out, hidden = self.rnn(x, hidden)       # out: (B,1,H)
        logits = self.fc(out.squeeze(1))        # (B,V)
        return logits, hidden


class AlignmentAttention(nn.Module):
    """
    Alignment functions:
      - bahdanau (additive): v^T tanh(W_h q + W_s k)
      - dot:                q^T k
      - general:            q^T W k   (Luong general / multiplicative)
    where q is decoder hidden (top layer), k are encoder outputs.
    """
    def __init__(self, hid_dim, attn_type="bahdanau"):
        super().__init__()
        self.hid_dim = hid_dim
        self.attn_type = attn_type

        if attn_type == "bahdanau":
            self.W_h = nn.Linear(hid_dim, hid_dim, bias=False)
            self.W_s = nn.Linear(hid_dim, hid_dim, bias=False)
            self.v = nn.Linear(hid_dim, 1, bias=False)
        elif attn_type == "general":
            self.W_s = nn.Linear(hid_dim, hid_dim, bias=False)
        elif attn_type == "dot":
            pass
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

    def forward(self, dec_h_top, enc_out, src_mask):
        """
        dec_h_top: (B,H)
        enc_out:   (B,S,H)
        src_mask:  (B,S) True for non-pad
        returns:
          ctx:  (B,H)
          attn: (B,S)
        """
        if self.attn_type == "bahdanau":
            e = torch.tanh(self.W_h(dec_h_top).unsqueeze(1) + self.W_s(enc_out))  # (B,S,H)
            scores = self.v(e).squeeze(-1)                                        # (B,S)
        elif self.attn_type == "dot":
            scores = torch.bmm(enc_out, dec_h_top.unsqueeze(2)).squeeze(2)        # (B,S)
        else:  # general
            proj = self.W_s(enc_out)                                              # (B,S,H)
            scores = torch.bmm(proj, dec_h_top.unsqueeze(2)).squeeze(2)           # (B,S)

        scores = scores.masked_fill(~src_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)                                          # (B,S)
        ctx = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)                    # (B,H)
        return ctx, attn


class DecoderAttnGRU(nn.Module):
    """GRU decoder with attention (bahdanau/dot/general)."""
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=2, dropout=0.2, pad_id=0, attn_type="bahdanau"):
        super().__init__()
        self.pad_id = pad_id
        self.attn_type = attn_type

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.attn = AlignmentAttention(hid_dim, attn_type=attn_type)

        # input: [emb ; ctx]  -> keep SAME for all attention types (fair comparison)
        self.rnn = nn.GRU(
            input_size=emb_dim + hid_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hid_dim + hid_dim, vocab_size)

    def forward_step(self, input_token, hidden, enc_out, src_mask):
        emb = self.emb(input_token)           # (B,E)
        dec_h_top = hidden[-1]                # (B,H)
        ctx, attn = self.attn(dec_h_top, enc_out, src_mask)  # (B,H), (B,S)

        rnn_in = torch.cat([emb, ctx], dim=-1).unsqueeze(1)  # (B,1,E+H)
        out, hidden = self.rnn(rnn_in, hidden)               # out:(B,1,H)
        out = out.squeeze(1)                                 # (B,H)

        logits = self.fc(torch.cat([out, ctx], dim=-1))       # (B,V)
        return logits, hidden, attn


class Seq2Seq(nn.Module):
    def __init__(self, encoder: EncoderGRU, decoder: nn.Module, pad_id=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id

    def forward(self, src, tgt, teacher_forcing_ratio=1.0):
        B, T = tgt.size()
        enc_out, enc_h = self.encoder(src)
        dec_h = enc_h

        src_mask = (src != self.pad_id)  # (B,S)

        inputs = tgt[:, 0]  # BOS
        logits_all = []

        for t in range(1, T):
            if isinstance(self.decoder, DecoderAttnGRU):
                logits, dec_h, _ = self.decoder.forward_step(inputs, dec_h, enc_out, src_mask)
            else:
                logits, dec_h = self.decoder.forward_step(inputs, dec_h)

            logits_all.append(logits.unsqueeze(1))  # (B,1,V)

            use_teacher = (torch.rand(B, device=tgt.device) < teacher_forcing_ratio)
            pred = logits.argmax(dim=-1)
            gold = tgt[:, t]
            inputs = torch.where(use_teacher, gold, pred)

        return torch.cat(logits_all, dim=1)  # (B, T-1, V)

    @torch.no_grad()
    def greedy_decode(self, src, bos_id, eos_id, max_len=120):
        enc_out, enc_h = self.encoder(src)
        dec_h = enc_h
        src_mask = (src != self.pad_id)

        B = src.size(0)
        inputs = torch.full((B,), bos_id, dtype=torch.long, device=src.device)
        out_ids = []

        for _ in range(max_len):
            if isinstance(self.decoder, DecoderAttnGRU):
                logits, dec_h, _ = self.decoder.forward_step(inputs, dec_h, enc_out, src_mask)
            else:
                logits, dec_h = self.decoder.forward_step(inputs, dec_h)

            pred = logits.argmax(dim=-1)
            out_ids.append(pred.unsqueeze(1))
            inputs = pred

        out = torch.cat(out_ids, dim=1)  # (B,L)

        results = []
        for i in range(B):
            seq = out[i].tolist()
            if eos_id in seq:
                seq = seq[:seq.index(eos_id)]
            results.append(seq)
        return results

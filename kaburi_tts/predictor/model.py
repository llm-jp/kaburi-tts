"""KaburiTimingPredictor = phone+SIL joint duration predictor。

入力: per-channel に並べた [PRE_SIL, PHONE, PHONE, ..., PRE_SIL, PHONE, ...] token 列を
A の token 列 + B の token 列 として concat したもの。

各 token は: token_type (= PHONE / PRE_SIL / FIRST_SIL / PAD), phone_id, channel_id, speaker_id,
utt_index, pos_in_channel を持つ。

shared Transformer encoder で 全 token を処理し、
PHONE 用 dur head、SIL 用 dur head、turn-taking any-gap head を 別々に持つ。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from kaburi_tts.predictor import (
    N_PHONE_CLASSES, N_SIL_CLASSES, N_GAP_CLASSES, N_TOKEN_TYPES,
    TOKEN_TYPE_PHONE, TOKEN_TYPE_PRE_SIL, TOKEN_TYPE_FIRST_SIL,
)


class KaburiTimingPredictor(nn.Module):
    """joint duration predictor (= PHONE+SIL joint duration + gap + frame-state)。

    Args:
      phone_vocab_size: phone embedding の vocab (= 80, padding_idx=0)
      n_speakers_max: speaker embedding の max
      hidden: Transformer hidden dim (= 192)
      layers: Transformer encoder layer 数 (= 2)
      heads: attention heads (= 4)
      ff_dim: feedforward dim (= 768)
      dropout: 0.1
      n_phone_classes: 30
      n_sil_classes: 24
      n_gap_classes: cross-channel any-gap classes
      max_pos: pos emb max (= 1024、 channel 内 position 用)
    """

    def __init__(
        self,
        *,
        phone_vocab_size: int = 80,
        n_speakers_max: int = 100,
        hidden: int = 192,
        layers: int = 2,
        heads: int = 4,
        ff_dim: int = 768,
        dropout: float = 0.1,
        n_phone_classes: int = N_PHONE_CLASSES,
        n_sil_classes: int = N_SIL_CLASSES,
        n_gap_classes: int = N_GAP_CLASSES,
        max_pos: int = 1024,
        use_text: bool = False,
        text_vocab_size: int = 100000,
        text_emb_dim: int = 64,
        use_utt_context: bool = False,
        use_dialog_order_context: bool = False,
        utt_context_layers: int = 1,
        max_dialog_pos: int = 256,
    ):
        super().__init__()
        self.n_phone_classes = n_phone_classes
        self.n_sil_classes = n_sil_classes
        self.n_gap_classes = n_gap_classes
        self.use_text = bool(use_text)
        self.use_utt_context = bool(use_utt_context)
        self.use_dialog_order_context = bool(use_dialog_order_context)

        # input embeddings
        self.phone_emb = nn.Embedding(phone_vocab_size, 128, padding_idx=0)
        self.token_type_emb = nn.Embedding(N_TOKEN_TYPES, 16)
        self.channel_emb = nn.Embedding(2, 16)
        self.speaker_emb = nn.Embedding(n_speakers_max, 64, padding_idx=0)
        self.pos_emb = nn.Embedding(max_pos, hidden)

        in_dim = 128 + 16 + 16 + 64
        self.input_proj = nn.Linear(in_dim, hidden)
        if self.use_text:
            self.text_emb = nn.Embedding(text_vocab_size, text_emb_dim, padding_idx=0)
            self.text_proj = nn.Sequential(
                nn.LayerNorm(text_emb_dim),
                nn.Linear(text_emb_dim, hidden),
            )
            if self.use_utt_context:
                utt_layer = nn.TransformerEncoderLayer(
                    d_model=hidden, nhead=heads, dim_feedforward=ff_dim,
                    dropout=dropout, batch_first=True, norm_first=True,
                )
                self.utt_context_encoder = nn.TransformerEncoder(
                    utt_layer, num_layers=int(utt_context_layers)
                )
                self.utt_context_norm = nn.LayerNorm(hidden)
                self.utt_context_gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
                if self.use_dialog_order_context:
                    self.utt_context_channel_emb = nn.Embedding(2, hidden)
                    self.utt_context_speaker_proj = nn.Linear(64, hidden)
                    self.utt_context_transition_emb = nn.Embedding(6, hidden, padding_idx=0)
                    self.utt_context_length_emb = nn.Embedding(4, hidden, padding_idx=0)
                    self.utt_context_pos_emb = nn.Embedding(max_dialog_pos + 1, hidden, padding_idx=0)
                else:
                    self.utt_context_channel_emb = None
                    self.utt_context_speaker_proj = None
                    self.utt_context_transition_emb = None
                    self.utt_context_length_emb = None
                    self.utt_context_pos_emb = None
            else:
                self.utt_context_encoder = None
                self.utt_context_norm = None
                self.utt_context_gate = None
                self.utt_context_channel_emb = None
                self.utt_context_speaker_proj = None
                self.utt_context_transition_emb = None
                self.utt_context_length_emb = None
                self.utt_context_pos_emb = None
        else:
            self.text_emb = None
            self.text_proj = None
            self.utt_context_encoder = None
            self.utt_context_norm = None
            self.utt_context_gate = None
            self.utt_context_channel_emb = None
            self.utt_context_speaker_proj = None
            self.utt_context_transition_emb = None
            self.utt_context_length_emb = None
            self.utt_context_pos_emb = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out_norm = nn.LayerNorm(hidden)

        # heads
        self.phone_dur_head = nn.Linear(hidden, n_phone_classes)
        self.sil_dur_head = nn.Linear(hidden, n_sil_classes)
        self.any_gap_head = nn.Linear(hidden, n_gap_classes)

    def forward(
        self,
        token_type: torch.Tensor,        # [B, N] long
        phone_id: torch.Tensor,           # [B, N] long
        channel_id: torch.Tensor,         # [B, N] long
        speaker_id: torch.Tensor,         # [B, N] long
        pos_in_channel: torch.Tensor,     # [B, N] long
        attn_mask: torch.Tensor,          # [B, N] bool (True = valid)
        utt_text_ids: torch.Tensor | None = None,    # [B, N, L] long
        utt_text_mask: torch.Tensor | None = None,   # [B, N, L] bool
        utt_index: torch.Tensor | None = None,        # [B, N] long, channel-local utt id
        is_utt_first_token: torch.Tensor | None = None,  # [B, N] long/bool
        dialog_rank: torch.Tensor | None = None,      # [B, N] long, dialog-order utt rank
        transition_type_id: torch.Tensor | None = None,  # [B, N] long
        length_bin_id: torch.Tensor | None = None,    # [B, N] long
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns phone, sil, any-gap logits, all token-aligned."""
        B, N = token_type.shape

        # PHONE token 以外 は phone_id=0 で padding embedding を引く
        e_phone = self.phone_emb(phone_id)
        e_type = self.token_type_emb(token_type)
        e_ch = self.channel_emb(channel_id)
        e_spk = self.speaker_emb(speaker_id)

        x = torch.cat([e_phone, e_type, e_ch, e_spk], dim=-1)
        x = self.input_proj(x)

        # position emb (= channel 内 position)
        pos = pos_in_channel.clamp(max=self.pos_emb.num_embeddings - 1)
        x = x + self.pos_emb(pos)

        text_cond = None
        if self.use_text and utt_text_ids is not None and self.text_emb is not None:
            ids = utt_text_ids.clamp(min=0, max=self.text_emb.num_embeddings - 1)
            te = self.text_emb(ids)  # [B, N, L, D]
            if utt_text_mask is None:
                tm = (ids != 0)
            else:
                tm = utt_text_mask.bool()
            denom = tm.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
            pooled = (te * tm.unsqueeze(-1).float()).sum(dim=-2) / denom
            text_cond = self.text_proj(pooled)
            x = x + text_cond

            if (
                self.use_utt_context
                and self.utt_context_encoder is not None
                and utt_index is not None
                and is_utt_first_token is not None
            ):
                first_mask = attn_mask & is_utt_first_token.bool()
                token_ctx = torch.zeros_like(x)
                if self.use_dialog_order_context and dialog_rank is not None:
                    if transition_type_id is None:
                        transition_type_id = torch.zeros_like(token_type)
                    if length_bin_id is None:
                        length_bin_id = torch.zeros_like(token_type)
                    rank_pos = (dialog_rank + 1).clamp(
                        min=0, max=self.utt_context_pos_emb.num_embeddings - 1
                    )
                    context_base = (
                        text_cond
                        + self.utt_context_channel_emb(channel_id.clamp(min=0, max=1))
                        + self.utt_context_speaker_proj(e_spk)
                        + self.utt_context_transition_emb(transition_type_id.clamp(min=0, max=5))
                        + self.utt_context_length_emb(length_bin_id.clamp(min=0, max=3))
                        + self.utt_context_pos_emb(rank_pos)
                    )
                    # Gather utterance first tokens in true dialog order, not A-then-B
                    # concat order, then scatter each contextualized vector back.
                    for b in range(B):
                        first_pos = first_mask[b].nonzero(as_tuple=False).squeeze(-1)
                        if first_pos.numel() == 0:
                            continue
                        order = torch.argsort(dialog_rank[b, first_pos])
                        first_pos = first_pos[order]
                        ctx_seq = self.utt_context_encoder(context_base[b, first_pos].unsqueeze(0))
                        ctx_seq = self.utt_context_norm(ctx_seq).squeeze(0)
                        for j, fp_t in enumerate(first_pos):
                            fp = int(fp_t.item())
                            same_utt = (
                                attn_mask[b]
                                & (channel_id[b] == channel_id[b, fp])
                                & (utt_index[b] == utt_index[b, fp])
                            )
                            token_ctx[b, same_utt] = ctx_seq[j]
                else:
                    # Legacy textctx: contextualize at original token positions. This
                    # keeps older configs/checkpoints compatible for baseline runs.
                    utt_ctx = self.utt_context_encoder(
                        text_cond,
                        src_key_padding_mask=~first_mask,
                    )
                    utt_ctx = self.utt_context_norm(utt_ctx)
                    for b in range(B):
                        first_pos = first_mask[b].nonzero(as_tuple=False).squeeze(-1)
                        for fp in first_pos.tolist():
                            same_utt = (
                                attn_mask[b]
                                & (channel_id[b] == channel_id[b, fp])
                                & (utt_index[b] == utt_index[b, fp])
                            )
                            token_ctx[b, same_utt] = utt_ctx[b, fp]
                x = x + self.utt_context_gate.to(x.dtype) * token_ctx

        # encoder
        x = self.encoder(x, src_key_padding_mask=~attn_mask)
        x = self.out_norm(x)

        phone_logits = self.phone_dur_head(x)   # [B, N, n_phone_classes]
        sil_logits = self.sil_dur_head(x)        # [B, N, n_sil_classes]
        any_gap_logits = self.any_gap_head(x)    # [B, N, n_gap_classes]
        return phone_logits, sil_logits, any_gap_logits


__all__ = ["KaburiTimingPredictor"]

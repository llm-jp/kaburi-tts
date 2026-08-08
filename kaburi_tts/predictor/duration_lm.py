"""自己回帰 Duration モデル (DurationLM)。

MFA 実測の音素継続長分布を模倣する生成モデル。音素列全体を双方向 encoder で
読み、継続長を左から順に「直前までの継続長」に条件付けてサンプルする。

動機 (2026-07-18 dur_analysis): timing predictor の expected デコードは条件付き
平均のため分散が 1/4 に潰れ nPVI が半減する。一方、独立サンプリングは隣接音素間の
継続長相関を持たないため無相関ジッタになり不安定に聞こえる。AR 構造により
「実測に似た、首尾一貫した揺らぎ」を学習ベースで生成する。ポーズ直前伸長・
発話末伸長などの系統構造もデータから学習される (ハンドスケール不要)。

継続長ビン: 音素は 1..30 frame、発話内 <sil> は 1..30 + 拡張ビン (32..60)。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

PAD = 0
SIL_ID = 1
MAX_LEN = 128
PHONE_MAX_BIN = 30                       # 音素は bin 0..29 (= 1..30 frame)
EXT_VALUES = [32, 35, 38, 42, 46, 50, 55, 60]  # sil 専用の拡張ビン
BIN_VALUES = list(range(1, PHONE_MAX_BIN + 1)) + EXT_VALUES
N_BINS = len(BIN_VALUES)
BOS_BIN = N_BINS  # AR の开始トークン (embedding 専用)


def dur_to_bin(d: float, is_sil: bool) -> int:
    d = max(1, int(round(d)))
    if is_sil:
        best, bi = None, 0
        for i, v in enumerate(BIN_VALUES):
            e = abs(v - min(d, BIN_VALUES[-1]))
            if best is None or e < best:
                best, bi = e, i
        return bi
    return min(d, PHONE_MAX_BIN) - 1


class DurationLM(nn.Module):
    def __init__(self, vocab_size: int = 79, dim: int = 128, enc_layers: int = 4,
                 dec_layers: int = 2, n_heads: int = 4, ff: int = 256, max_len: int = MAX_LEN):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, dim)
        enc = nn.TransformerEncoderLayer(dim, n_heads, ff, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, enc_layers)
        self.dur_emb = nn.Embedding(N_BINS + 1, dim)  # +1 = BOS
        dec = nn.TransformerEncoderLayer(dim, n_heads, ff, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.dec = nn.TransformerEncoder(dec, dec_layers)
        self.head = nn.Linear(dim, N_BINS)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, prev_bins: torch.Tensor) -> torch.Tensor:
        """x: (B,L) phone ids / prev_bins: (B,L) 直前 bin (先頭は BOS_BIN)。"""
        L = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(L, device=x.device))[None]
        h = self.enc(h, src_key_padding_mask=~mask)
        h = h + self.dur_emb(prev_bins)
        causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        h = self.dec(h, mask=causal, src_key_padding_mask=~mask)
        return self.head(h)  # (B, L, N_BINS)


def load_duration_lm(ckpt_path: str | Path, device: str = "cpu") -> DurationLM:
    d = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DurationLM(**d.get("config", {}))
    model.load_state_dict(d["model"])
    model.to(device).eval()
    return model


@torch.no_grad()
def sample_durations(model: DurationLM, phone_ids: list[int], *, device: str = "cpu",
                     temperature: float = 1.0, generator=None) -> list[float]:
    """発話 1 本の継続長を AR サンプル。返り値は frame 数 (float) のリスト。"""
    ids = phone_ids[:MAX_LEN]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    m = torch.ones_like(x, dtype=torch.bool)
    L = len(ids)
    h_enc = model.emb(x) + model.pos(torch.arange(L, device=x.device))[None]
    h_enc = model.enc(h_enc, src_key_padding_mask=~m)
    bins = []
    prev = torch.full((1, L), BOS_BIN, dtype=torch.long, device=device)
    for i in range(L):
        h = h_enc + model.dur_emb(prev)
        causal = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), 1)
        hd = model.dec(h, mask=causal, src_key_padding_mask=~m)
        logits = model.head(hd)[0, i]
        if ids[i] != SIL_ID:
            logits[PHONE_MAX_BIN:] = float("-inf")  # 音素に拡張ビンは使わない
        p = torch.softmax(logits / max(temperature, 1e-3), dim=-1)
        b = int(torch.multinomial(p, 1, generator=generator))
        bins.append(b)
        if i + 1 < L:
            prev[0, i + 1] = b
    out = [float(BIN_VALUES[b]) for b in bins]
    out += [2.0] * (len(phone_ids) - len(out))  # MAX_LEN 超過分の保険
    return out

"""発話内ポーズ挿入モデル (PauseTagger) の定義とロード。

学習は scripts/train_pause_model.py。音素列の各位置について「直後にポーズ
(<sil>) が入るか」の logit を返す 0.56M の Transformer タガー。
チェックポイントは {"model": state_dict, "config": {...}} 形式。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

PAD = 0
MAX_LEN = 128


class PauseTagger(nn.Module):
    def __init__(self, vocab_size: int = 79, dim: int = 128, n_layers: int = 4,
                 n_heads: int = 4, ff: int = 256, max_len: int = MAX_LEN,
                 use_end_pos: bool = False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, dim)
        # 末尾からの距離 embedding。文末直前のポーズはコーパスで極めて稀
        # (す|よ 間 0.04%) なのに絶対位置だけでは学習されなかったため追加 (v2)。
        self.use_end_pos = bool(use_end_pos)
        if self.use_end_pos:
            self.pos_end = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(dim, n_heads, ff, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        idx = torch.arange(L, device=x.device)
        h = self.emb(x) + self.pos(idx)[None]
        if self.use_end_pos:
            lens = mask.long().sum(1)                        # (B,)
            dist = (lens[:, None] - 1 - idx[None]).clamp(0, self.pos_end.num_embeddings - 1)
            h = h + self.pos_end(dist)
        h = self.enc(h, src_key_padding_mask=~mask)
        return self.head(h)[..., 0]  # (B, L) logits


def load_tagger(ckpt_path: str | Path, device: str = "cpu") -> PauseTagger:
    d = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PauseTagger(**d.get("config", {}))
    model.load_state_dict(d["model"])
    model.to(device).eval()
    return model


@torch.no_grad()
def pause_probs(model: PauseTagger, phone_ids: list[int], device: str = "cpu") -> list[float]:
    """各位置の「直後ポーズ」確率。MAX_LEN 超は末尾を打ち切り (0 扱い)。"""
    ids = phone_ids[:MAX_LEN]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    m = torch.ones_like(x, dtype=torch.bool)
    p = torch.sigmoid(model(x, m))[0].tolist()
    return p + [0.0] * (len(phone_ids) - len(ids))

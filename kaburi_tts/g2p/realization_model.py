"""実現形変換モデル (RealizationTagger)。

G2P 規範形音素列を MFA 実現形へ写す monotonic edit-tagger:
  各位置に op (KEEP / DEL / SUB→上位K クラス) + insert_sil (直後に <sil>) を予測。
PauseTagger・無声化規則・縮約/削除/長音化を単一モデルに統合する (E16 診断の
第 1・第 2 因子への対策、DIAGNOSIS_E2E_GAP_20260719.md)。

チェックポイント形式: {"model": state_dict, "config": {...}, "sub_targets": [r_id...],
                       "sil_threshold": float}
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

PAD = 0
MAX_LEN = 176
SIL_ID = 1


class RealizationTagger(nn.Module):
    def __init__(self, vocab_size: int = 79, n_sub: int = 48, dim: int = 160,
                 n_layers: int = 4, n_heads: int = 4, ff: int = 320,
                 max_len: int = MAX_LEN):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, dim)
        self.pos_end = nn.Embedding(max_len, dim)   # 末尾距離 (文末文脈)
        self.tokf = nn.Embedding(2, dim)            # トークン最終音素フラグ
        layer = nn.TransformerEncoderLayer(dim, n_heads, ff, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.op_head = nn.Linear(dim, 2 + n_sub)    # 0=KEEP 1=DEL 2..=SUB_k
        self.sil_head = nn.Linear(dim, 1)
        self.n_sub = n_sub

    def forward(self, x, tf, mask):
        L = x.shape[1]
        idx = torch.arange(L, device=x.device)
        lens = mask.long().sum(1)
        dist = (lens[:, None] - 1 - idx[None]).clamp(0, self.pos_end.num_embeddings - 1)
        h = self.emb(x) + self.pos(idx)[None] + self.pos_end(dist) + self.tokf(tf)
        h = self.enc(h, src_key_padding_mask=~mask)
        return self.op_head(h), self.sil_head(h)[..., 0]


def load_realizer(ckpt_path: str | Path, device: str = "cpu"):
    d = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = RealizationTagger(**d.get("config", {}))
    model.load_state_dict(d["model"])
    model.to(device).eval()
    return (model, d["sub_targets"], float(d.get("sil_threshold", 0.5)),
            float(d.get("op_threshold", 0.0)))


@torch.no_grad()
def realize(model, sub_targets: list[int], sil_threshold: float,
            phone_ids: list[int], tok_final: list[int], device: str = "cpu",
            op_threshold: float = 0.0, vocab: dict | None = None,
            edit_stats: dict | None = None, majority_rate: float = 0.5,
            min_context_n: int = 20, sil_gate_rate: float = 0.0274) -> list[int]:
    """規範形 → 実現形音素列 (内部 <sil> 込み)。決定的 (argmax + 較正済み閾値)。

    vocab (symbol->id) を渡すと、同一母音連続の削除を補償延長 (a a → aː) に
    変換する (実音声の融合は削除でなく長母音化。位置独立デコードの整合性補正)。"""
    ids = phone_ids[:MAX_LEN]
    tf = tok_final[:MAX_LEN]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    t = torch.tensor([tf], dtype=torch.long, device=device)
    m = torch.ones_like(x, dtype=torch.bool)
    op_logits, sil_logits = model(x, t, m)
    op_probs = torch.softmax(op_logits[0], -1)
    ops = op_probs.argmax(-1).tolist()
    # DEL/SUB は確信度が閾値未満なら KEEP に落とす (過剰削除の抑制、較正済み)
    for i in range(len(ops)):
        if ops[i] != 0 and float(op_probs[i, ops[i]]) < op_threshold:
            ops[i] = 0
    # 多数派編集ゲート: 編集はその trigram 文脈でコーパス自身が多数派として
    # 行っている場合のみ許可 (統計表は教師データから機械生成。手書き規則なし)
    if edit_stats is not None:
        for i in range(len(ops)):
            if ops[i] == 0:
                continue
            prev = ids[i-1] if i > 0 else 0
            nxt = ids[i+1] if i+1 < len(ids) else 0
            t = tf[i] if i < len(tf) else 1
            ctr = edit_stats.get(f"{prev},{ids[i]},{nxt},{t}")
            if ctr is None:
                ops[i] = 0
                continue
            tot = sum(ctr.values())
            op_key = "D" if ops[i] == 1 else f"S{sub_targets[ops[i]-2]}"
            if tot < min_context_n or ctr.get(op_key, 0) / tot < majority_rate:
                ops[i] = 0
    sils = torch.sigmoid(sil_logits[0]).tolist()
    # sil ゲート: コーパスの当該文脈ポーズ率が基準率 (全体 2.74%) 以上の位置のみ許可
    # (語中など、実会話でポーズが起きない文脈への挿入を統計で排除)
    if edit_stats is not None:
        for i in range(len(sils)):
            prev = ids[i-1] if i > 0 else 0
            nxt = ids[i+1] if i+1 < len(ids) else 0
            t = tf[i] if i < len(tf) else 1
            ctr = edit_stats.get(f"{prev},{ids[i]},{nxt},{t}")
            tot = sum(v for k, v in ctr.items() if k != "SIL") if ctr else 0
            if tot < min_context_n or ctr.get("SIL", 0) / tot < sil_gate_rate:
                sils[i] = 0.0
    inv = {v: k for k, v in vocab.items()} if vocab else {}
    VOWELS = ("a", "i", "e", "o", "ɯ", "ɨ")
    # 整合性制約: 挿入 sil をまたぐ融合・同一母音削除は無効 (ポーズ越しに母音は
    # 融合できない)。sil 決定を先に確定し、境界をまたぐ編集を KEEP に戻す。
    sil_after = [sils[i] >= sil_threshold and i < len(ids) - 1 for i in range(len(ids))]
    for i in range(len(ops)):
        if ops[i] != 1:
            continue
        same_prev = i > 0 and ids[i-1] == ids[i] and not sil_after[i-1]
        same_next = i + 1 < len(ids) and ids[i+1] == ids[i] and not sil_after[i]
        cross_prev = i > 0 and ids[i-1] == ids[i] and sil_after[i-1]
        cross_next = i + 1 < len(ids) and ids[i+1] == ids[i] and sil_after[i]
        if (cross_prev or cross_next) and not (same_prev or same_next):
            ops[i] = 0  # 融合相手が sil の向こう側にしかない削除は無効
    # 標準長音規則の融合対 (o+ɯ→oː 等)。SUB→長母音の相方消費 (融合デコード) に使用
    FUSE = {("o", "ɯ"): "oː", ("e", "i"): "eː", ("a", "a"): "aː", ("i", "i"): "iː",
            ("ɯ", "ɯ"): "ɯː", ("e", "e"): "eː", ("o", "o"): "oː", ("ɨ", "ɨ"): "ɨː"}
    out = []
    skip = False
    for i, pid in enumerate(ids):
        if skip:
            skip = False
            continue
        op = ops[i]
        if op == 0:
            out.append(pid)
        elif op == 1:
            # 同一母音連続の削除は時間保存デコードで長母音へ。コーパス記号上の
            # 多数派は「単一の短い a への融合」だが、融合母音は時間側で長く実現
            # されており (MFA が長区間を 1 記号に割当)、記号列 interface では
            # 長母音記号が時間情報の唯一の運搬手段のため (aː も 1.2k 件実証)。
            cur = inv.get(pid, "") if vocab else ""
            long_id = vocab.get(cur + "ː") if (vocab and cur in VOWELS) else None
            if long_id is not None and out and inv.get(out[-1], "") == cur:
                out[-1] = long_id          # 前方融合: ...a + DEL(a) → ...aː
            elif long_id is not None and i + 1 < len(ids) and ids[i + 1] == pid                     and ops[i + 1] == 0:
                out.append(long_id)        # 後方融合: DEL(a) + a → aː
                skip = True
        else:
            sub_id = sub_targets[op - 2]
            out.append(sub_id)
            # 融合デコード: SUB→長母音 で次音素が標準融合対の相方なら相方を消費
            if vocab and i + 1 < len(ids):
                sub_sym = inv.get(sub_id, "")
                cur_sym = inv.get(pid, "")
                nxt_sym = inv.get(ids[i + 1], "")
                if sub_sym.endswith("ː") and FUSE.get((cur_sym, nxt_sym)) == sub_sym:
                    skip = True
        if sils[i] >= sil_threshold and i < len(ids) - 1 and out and out[-1] != SIL_ID:
            out.append(SIL_ID)
    out += phone_ids[MAX_LEN:]  # 超過分はそのまま (176 音素超は実質ない)
    return out

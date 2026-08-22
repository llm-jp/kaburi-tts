"""規範形 (canonical) 音素列の構築と対話文脈: 音素ラスタ生成が使う共通 G2P 部品。

- canon(): text -> 規範形音素列 + トークン整列 (Sudachi 分割 + 辞書 + overrides +
  語末促音 gemination)。dict_g2p の sudachi モードと同一の読み解決。
- build_dialog_ctx(): 発話列の i 番目の対話文脈 (話者遷移・前後話者・chunk 内位置)。
- FUSE / VOWELS / SIL_ID / MAX_LEN: 実現形デコードで使う音韻定数。
"""
from __future__ import annotations

from kaburi_tts.g2p.kana2phone import kana_to_phones
from kaburi_tts.g2p.dict_g2p import (segment_sudachi, remerge_dict_compounds,
                                     resolve_gemination, _dict_is_reduction)

MAX_LEN = 176
SIL_ID = 1
VOWELS = ("a", "i", "e", "o", "ɯ", "ɨ")
FUSE = {("o", "ɯ"): "oː", ("e", "i"): "eː", ("a", "a"): "aː", ("i", "i"): "iː",
        ("ɯ", "ɯ"): "ɯː", ("e", "e"): "eː", ("o", "o"): "oː", ("ɨ", "ɨ"): "ɨː"}


def canon(text, d, ov, vocab, _cache=None):
    """text -> (c, tf, tokens_meta, ph_tok, ph_mora) or None。規範形音素列。"""
    if _cache is not None and text in _cache:
        return _cache[text]
    try:
        pairs = remerge_dict_compounds(segment_sudachi(text), {**d, **ov}, curated=set(ov))
    except Exception:
        if _cache is not None:
            _cache[text] = None
        return None
    tphones, srcs, metas, rds = [], [], [], []
    for surf, reading, pos in pairs:
        ph = ov.get(surf)
        if ph is not None:
            pass  # 手キュレーション override は最優先
        elif (ph := d.get(surf)) is not None:
            # 接尾辞の誤タグで Sudachi 読みが辞書名詞読みを上書きするのを防ぐ
            if reading and pos != "接尾辞":
                _k = kana_to_phones(reading)
                if not _dict_is_reduction(ph, _k):
                    ph = _k
        elif reading:
            ph = kana_to_phones(reading)
        else:
            ph = None
        if not ph:
            if _cache is not None:
                _cache[text] = None
            return None
        tphones.append(list(ph)); srcs.append("d"); metas.append((surf, pos)); rds.append(reading)
    resolve_gemination(tphones, srcs, rds)   # 語末促音 (だった→d a tː a)
    c, tf, ph_tok, ph_mora, tokens_meta = [], [], [], [], []
    for ph, (surf, pos) in zip(tphones, metas):
        ids = [vocab[p] for p in ph if p in vocab]
        if not ids:
            continue
        for j, pid in enumerate(ids):
            c.append(pid); ph_tok.append(len(tokens_meta)); ph_mora.append(j)
        tf += [0] * (len(ids) - 1) + [1]
        tokens_meta.append([surf, pos, len(ids)])
    res = (c, tf, tokens_meta, ph_tok, ph_mora)
    if _cache is not None:
        _cache[text] = res
    return res


def build_dialog_ctx(utterances, idx):
    """順序付き utterances[(speaker,text),...] の idx 番目の対話文脈。
    transition_type=chunk_first or f"{prev}_to_{cur}"、pos_in_chunk=(i+0.5)/n。"""
    n = len(utterances)
    cur = utterances[idx][0]
    prev = utterances[idx - 1][0] if idx > 0 else None
    nxt = utterances[idx + 1][0] if idx + 1 < n else None
    trans = "chunk_first" if prev is None else f"{prev}_to_{cur}"
    return {"prev_spk": prev if prev is not None else "<bos>",
            "next_spk": nxt if nxt is not None else "<eos>",
            "trans": trans, "pos_in_chunk": (idx + 0.5) / max(n, 1)}

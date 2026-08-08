"""G2P 音素列への発話内ポーズ (<sil>) の学習ベース挿入。

背景: 学習コーパス (自発対話の MFA アライメント) では発話内ポーズが発話長と
ともにほぼ必須化する (21-30 音素で 55%、51+ 音素で 91% の発話が内部 <sil> を持つ)
のに対し、辞書 G2P の規範形音素列はポーズを含まない。この分布ずれが長い発話の
棒読み・不安定さの主因であることを対照実験で確認済み (2026-07-18 E1)。

挿入位置は PauseTagger (kaburi_tts/g2p/pause_model.py、学習コーパスの実現形
音素列で学習した系列ラベラ) の予測確率に従う。規則ベースの位置ヒューリスティック
は用いない。挿入はトークン境界に限定する — 教師である MFA の sil も単語境界に
しか現れないため、これは教師の構造と同じ制約である。

ポーズの継続長は与えない — timing predictor が <sil> トークンの長さを文脈から
予測する (学習時に実データの <sil> で学習済み)。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

PAUSE_TOKEN = "<pause>"
SIL_PHONE = "<sil>"
MIN_GAP_PHONES = 3

# 事故削除フィルタ: 「規則でポーズを入れる」のではなく「明らかな事故だけ消す」安全策。
# PauseTagger は音素列しか見ないため、表層で自明な誤挿入 (語中・重畳語内部・補助動詞
# 連鎖の間) を検出できない。挿入後に表層条件で削除する (外部レビュー提案③、2026-07-18)。
_AUX_AFTER_TE = {"いる", "いく", "くる", "おく", "しまう", "みる", "ある", "た",
                 "いた", "いて", "いい", "る", "く"}
_KANA_SINGLE = set("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほ"
                   "まみむめもやゆよらりるれろわをんっゃゅょぁぃぅぇぉ")


def drop_accident_pauses(tokens: list[str], token_phones: list[list[str]]
                         ) -> tuple[list[str], list[list[str]], list[str]]:
    """挿入済み <pause> のうち表層的に明らかな事故を削除。返り値 3 つ目は削除理由 log。"""
    dropped = []
    out_t, out_p = [], []
    for i, (tok, ph) in enumerate(zip(tokens, token_phones)):
        if tok != PAUSE_TOKEN:
            out_t.append(tok)
            out_p.append(ph)
            continue
        prev_t = tokens[i - 1] if i > 0 else ""
        next_t = tokens[i + 1] if i + 1 < len(tokens) else ""
        reason = None
        if prev_t and prev_t == next_t:
            reason = f"重畳語内部 ({prev_t}|{next_t})"
        elif prev_t.endswith(("て", "で")) and next_t in _AUX_AFTER_TE:
            reason = f"補助動詞連鎖 ({prev_t}|{next_t})"
        elif len(next_t) == 1 and next_t in _KANA_SINGLE:
            reason = f"断片トークン前 ({prev_t}|{next_t})"
        if reason:
            dropped.append(reason)
        else:
            out_t.append(tok)
            out_p.append(ph)
    return out_t, out_p, dropped


def load_pause_stats(path: str | Path) -> dict:
    obj = json.loads(Path(path).read_text())
    obj["_bins"] = [tuple(b) for b in obj["bins"]]
    return obj


def _sample_k(n_phones: int, stats: dict, rng: random.Random) -> int:
    """発話長ビンの経験分布からポーズ個数 k をサンプル。"""
    for (a, b) in stats["_bins"]:
        if a <= n_phones <= b:
            hist = stats["k_hist"][f"{a}-{b}"]
            r = rng.random() * sum(hist)
            acc = 0
            for k, c in enumerate(hist):
                acc += c
                if r <= acc:
                    return k
    return 0


def insert_pauses_model(
    tokens: list[str],
    token_phones: list[list[str]],
    phone_ids: list[int],
    *,
    model,
    device: str = "cpu",
    mode: str = "count",
    stats: dict | None = None,
    rng: random.Random | None = None,
    threshold: float = 0.5,
) -> tuple[list[str], list[list[str]], list[int]]:
    """PauseTagger の予測でトークン境界に <pause> を挿入する。

    mode="count": 発話長ビンの経験分布から個数 k を先に決め、モデル確率の上位 k 境界に
      挿入 (最小間隔 3 音素)。境界独立の閾値判定より長さ別の挿入率が正確で、
      短発話への過剰挿入 (カタコト化) を抑える。stats + rng 必須。
    mode="threshold": 較正済み閾値 (calibrate_pause_threshold.py) で境界独立判定。
    返り値: (tokens, token_phones, 挿入した境界 index 一覧)。
    """
    from kaburi_tts.g2p.pause_model import pause_probs

    if len(tokens) < 2 or len(phone_ids) < 6:
        return tokens, token_phones, []
    probs = pause_probs(model, phone_ids, device=device)

    # トークン境界 = 各トークン最終音素の位置 (最後のトークンは境界なし)
    bounds = []  # (境界 index, 音素 offset, モデル確率)
    off = 0
    for i, ph in enumerate(token_phones[:-1]):
        off += len(ph)
        bounds.append((i + 1, off, probs[off - 1]))

    if mode == "count":
        assert stats is not None and rng is not None
        k = _sample_k(len(phone_ids), stats, rng)
        chosen, chosen_off = [], []
        for b_idx, b_off, p in sorted(bounds, key=lambda t: -t[2]):
            if len(chosen) >= k:
                break
            if any(abs(b_off - o) < MIN_GAP_PHONES for o in chosen_off):
                continue
            chosen.append(b_idx)
            chosen_off.append(b_off)
    else:
        chosen = [b_idx for b_idx, _, p in bounds if p >= threshold]
    if not chosen:
        return tokens, token_phones, []

    out_t, out_p = [], []
    ins = set(chosen)
    for i, (tok, ph) in enumerate(zip(tokens, token_phones)):
        if i in ins:
            out_t.append(PAUSE_TOKEN)
            out_p.append([SIL_PHONE])
        out_t.append(tok)
        out_p.append(list(ph))
    return out_t, out_p, chosen

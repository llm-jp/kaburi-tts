"""カタカナ読み → MFA japanese_mfa 音素列。

Sudachi 分割で OOV (辞書に無い活用語幹など、全体の ~1.5%) になったトークンを、
Sudachi の読み (reading_form, カタカナ) から音素へ変換するフォールバック。
規約は japanese_mfa 辞書の実エントリに合わせる (き=c i、す=s ɨ):
  - 長音 (ー / 連母音 オウ・エイ 等) は直前母音を **長音 1 音素** (Vː) 化。
  - 促音 (ッ) は次子音の **geminate** (Cː)。語末ッは越境するので sentinel `GEM` を返し、
    dict_g2p 側で次トークン頭子音を Cː 化する (声門閉鎖 ʔ は使わない)。
  - 撥音 (ン) は後続音で異音 (m/n/ŋ/ɰ̃/ɴ)。
辞書引き経路 (98.5%) を主とし、本表は残り 1.5% の近似に使う。辞書エントリとの音素一致で
検証する (scripts/validate_kana2phone.py 相当)。
"""
from __future__ import annotations

from functools import lru_cache

# 語末促音の越境 gemination を dict_g2p に伝える sentinel (vocab 外)。
GEM = "\x00GEM"

# 基本モーラ (拗音・長音・促音・撥音は下で処理)
_MORA = {
    "ア": ["a"], "イ": ["i"], "ウ": ["ɯ"], "エ": ["e"], "オ": ["o"],
    "カ": ["k", "a"], "キ": ["c", "i"], "ク": ["k", "ɯ"], "ケ": ["k", "e"], "コ": ["k", "o"],
    "ガ": ["ɡ", "a"], "ギ": ["ɟ", "i"], "グ": ["ɡ", "ɯ"], "ゲ": ["ɡ", "e"], "ゴ": ["ɡ", "o"],
    "サ": ["s", "a"], "シ": ["ɕ", "i"], "ス": ["s", "ɨ"], "セ": ["s", "e"], "ソ": ["s", "o"],
    "ザ": ["z", "a"], "ジ": ["dʑ", "i"], "ズ": ["z", "ɨ"], "ゼ": ["z", "e"], "ゾ": ["z", "o"],
    "タ": ["t", "a"], "チ": ["tɕ", "i"], "ツ": ["ts", "ɨ"], "テ": ["t", "e"], "ト": ["t", "o"],
    "ダ": ["d", "a"], "ヂ": ["dʑ", "i"], "ヅ": ["z", "ɨ"], "デ": ["d", "e"], "ド": ["d", "o"],
    "ナ": ["n", "a"], "ニ": ["ɲ", "i"], "ヌ": ["n", "ɯ"], "ネ": ["n", "e"], "ノ": ["n", "o"],
    "ハ": ["h", "a"], "ヒ": ["ç", "i"], "フ": ["ɸ", "ɯ"], "ヘ": ["h", "e"], "ホ": ["h", "o"],
    "バ": ["b", "a"], "ビ": ["bʲ", "i"], "ブ": ["b", "ɯ"], "ベ": ["b", "e"], "ボ": ["b", "o"],
    "パ": ["p", "a"], "ピ": ["pʲ", "i"], "プ": ["p", "ɯ"], "ペ": ["p", "e"], "ポ": ["p", "o"],
    "マ": ["m", "a"], "ミ": ["mʲ", "i"], "ム": ["m", "ɯ"], "メ": ["m", "e"], "モ": ["m", "o"],
    "ヤ": ["j", "a"], "ユ": ["j", "ɯ"], "ヨ": ["j", "o"],
    "ラ": ["ɾ", "a"], "リ": ["ɾʲ", "i"], "ル": ["ɾ", "ɯ"], "レ": ["ɾ", "e"], "ロ": ["ɾ", "o"],
    "ワ": ["w", "a"], "ヰ": ["i"], "ヱ": ["e"], "ヲ": ["o"],
    "ヴ": ["v", "ɯ"], "ファ": ["ɸ", "a"], "フィ": ["ɸʲ", "i"], "フェ": ["ɸ", "e"], "フォ": ["ɸ", "o"],
    "ティ": ["tʲ", "i"], "ディ": ["dʲ", "i"], "トゥ": ["t", "ɯ"], "ドゥ": ["d", "ɯ"],
    "ウィ": ["w", "i"], "ウェ": ["w", "e"], "ウォ": ["w", "o"], "ヴァ": ["v", "a"], "ヴィ": ["vʲ", "i"],
    "チェ": ["tɕ", "e"], "ジェ": ["dʑ", "e"], "シェ": ["ɕ", "e"], "ツァ": ["ts", "a"], "ツォ": ["ts", "o"],
}
# 拗音 (子音 + j + 母音)。i 段の子音を流用。ュ (u 段拗音) は高central母音 ɨ。
_YOON = {
    "キャ": ["c", "a"], "キュ": ["c", "ɨ"], "キョ": ["c", "o"],
    "ギャ": ["ɟ", "a"], "ギュ": ["ɟ", "ɨ"], "ギョ": ["ɟ", "o"],
    "シャ": ["ɕ", "a"], "シュ": ["ɕ", "ɨ"], "ショ": ["ɕ", "o"],
    "ジャ": ["dʑ", "a"], "ジュ": ["dʑ", "ɨ"], "ジョ": ["dʑ", "o"],
    "チャ": ["tɕ", "a"], "チュ": ["tɕ", "ɨ"], "チョ": ["tɕ", "o"],
    "ニャ": ["ɲ", "a"], "ニュ": ["ɲ", "ɨ"], "ニョ": ["ɲ", "o"],
    "ヒャ": ["ç", "a"], "ヒュ": ["ç", "ɨ"], "ヒョ": ["ç", "o"],
    "ビャ": ["bʲ", "a"], "ビュ": ["bʲ", "ɨ"], "ビョ": ["bʲ", "o"],
    "ピャ": ["pʲ", "a"], "ピュ": ["pʲ", "ɨ"], "ピョ": ["pʲ", "o"],
    "ミャ": ["mʲ", "a"], "ミュ": ["mʲ", "ɨ"], "ミョ": ["mʲ", "o"],
    "リャ": ["ɾʲ", "a"], "リュ": ["ɾʲ", "ɨ"], "リョ": ["ɾʲ", "o"],
}
_VOWEL = {"a", "i", "ɯ", "e", "o", "ɨ"}

# 撥音 (ン) の異音: 後続音素の調音位置で決まる (japanese_mfa 規約)。
_N_LABIAL = {"p", "b", "m"}
_N_LABIAL_PAL = {"pʲ", "bʲ", "mʲ"}
_N_VELAR = {"k", "ɡ"}
_N_PALATAL = {"c", "ɟ", "tɕ", "dʑ"}
_N_CORONAL = {"t", "d", "ts", "dz", "n", "ɲ", "ɾ", "ɾʲ", "tʲ", "dʲ"}


def _moraic_n(next_phone: str | None) -> str:
    """撥音 ン → 後続音素による異音。None (語末/次不明) は ɴ。
    s/z/ɕ/ʑ・母音・半母音・h/ç/ɸ 前は ɰ̃ (鼻母音的接近音)。"""
    if next_phone is None:
        return "ɴ"
    if next_phone in _N_LABIAL:
        return "m"
    if next_phone in _N_LABIAL_PAL:
        return "mʲ"
    if next_phone in _N_VELAR:
        return "ŋ"
    if next_phone in _N_PALATAL:
        return "ɲ"
    if next_phone in _N_CORONAL:
        return "n"
    return "ɰ̃"


# 連母音の長音化ペア (前母音, 後母音) → 前母音を長音化 (Vː)。
_COALESCE = {(v, v) for v in _VOWEL} | {("o", "ɯ"), ("e", "i"), ("ɨ", "ɯ")}


def _mora_units(reading: str) -> list[str]:
    """読みをモーラ単位 (拗音 2 字 / モーラ 1 字 / ー / ッ / ン) に分割。"""
    units: list[str] = []
    i = 0
    n = len(reading)
    while i < n:
        two = reading[i:i + 2]
        if two in _YOON or two in _MORA:
            units.append(two)
            i += 2
        else:
            units.append(reading[i])
            i += 1
    return units


def kana_to_phones(reading: str) -> list[str]:
    """カタカナ読み → 音素列 (japanese_mfa 規約)。未知文字は無視。
    語末促音は sentinel GEM を末尾に付す (dict_g2p が越境 gemination で解決)。
    読みは大量に重複するためキャッシュ (呼び出し側の破壊的変更に備え毎回新規 list を返す)。"""
    return list(_kana_to_phones_cached(reading))


@lru_cache(maxsize=200000)
def _kana_to_phones_cached(reading: str) -> tuple[str, ...]:
    units = _mora_units(reading)
    out: list[str] = []
    pending_gem = False  # 直前が ッ (次子音を geminate)
    n = len(units)
    for i, u in enumerate(units):
        if u == "ッ":
            pending_gem = True
            continue
        if u == "ー":
            if out and out[-1] in _VOWEL and not out[-1].endswith("ː"):
                out[-1] = out[-1] + "ː"
            continue
        if u == "ン":
            nxt = None
            if i + 1 < n:
                nu = units[i + 1]
                if nu not in ("ー", "ッ", "ン"):
                    ph = _YOON.get(nu) or _MORA.get(nu)
                    if ph:
                        nxt = ph[0]
            out.append(_moraic_n(nxt))
            continue
        ph = _YOON.get(u) or _MORA.get(u)
        if ph is None:
            continue
        ph = list(ph)
        if pending_gem:
            if ph[0] not in _VOWEL:
                ph[0] = ph[0] + "ː"  # 次子音を geminate
            pending_gem = False
        # 連母音の長音化 (オウ→oː, エイ→eː, 同母音連続→Vː)
        if (out and out[-1] in _VOWEL and not out[-1].endswith("ː")
                and ph[0] in _VOWEL and (out[-1], ph[0]) in _COALESCE):
            out[-1] = out[-1] + "ː"
            ph = ph[1:]
        out.extend(ph)
    if pending_gem:
        out.append(GEM)  # 語末促音: dict_g2p で次トークン頭子音を Cː 化
    return tuple(out)

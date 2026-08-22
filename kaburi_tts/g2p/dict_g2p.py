"""Eval/test dialog text -> phone ids using MFA dictionary longest match.

This helper is intentionally separate from g2p_dialog_mfa.py.  It does not
change the training-time path.  Use it only for hand-authored eval/test dialogs
where pronunciation quality matters more than reproducing the Sudachi token
boundary behavior used in training.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from functools import lru_cache as _lru_cache
from pathlib import Path

# スクリプト実行時 (python kaburi_tts/g2p/dict_g2p.py) でも kaburi_tts を解決可能に
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from kaburi_tts.g2p.kana2phone import kana_to_phones
from pathlib import Path


# 母音無声化の文脈修正: 無声化母音 (ɨ̥ 等) は無声子音間か発話末 (ポーズ前) でのみ生じる。
# 辞書は文脈非依存に最頻形を返すため、「ですよ」の す のように有声音が続く位置にも
# 無声化形が割り当てられる。学習データでは発話中間の無声化はポーズの強い前触れ
# (s ɨ̥ 直後 59% がポーズ) なので、誤った無声化は PauseTagger の誤挿入と音の不明瞭化を
# 併発する (2026-07-18)。次音が有声なら平母音へ戻す。
DEVOICED2PLAIN = {"ɨ̥": "ɨ", "i̥": "i", "ɯ̥": "ɯ"}
_VOICELESS_ONSET = {"p", "t", "k", "c", "ts", "tɕ", "s", "ɕ", "ç", "ɸ", "h"}


def _is_voiceless_onset(phone: str) -> bool:
    base = phone.replace("ː", "").replace("ʲ", "")
    return base in _VOICELESS_ONSET or (base[:1] in {"p", "t", "k", "c", "s", "h"} and base[:2] != "d")




PUNCT_CHARS =set("、。！？・,.!?…「」『』（）()［］[] \t\n\r")


def is_punct(text: str) -> bool:
    return bool(text) and all(ch in PUNCT_CHARS for ch in text)


def has_punct(text: str) -> bool:
    return any(ch in PUNCT_CHARS for ch in text)


def load_dict_entries(dict_path: str) -> dict[str, list[tuple[float, list[str]]]]:
    """語 → [(確率, 音素列), ...] (確率降順)。確率は MFA がコーパス整列から推定した
    実現形バリアントの分布 (無声化母音・縮約形など)。"""
    word2entries: dict[str, list[tuple[float, list[str]]]] = defaultdict(list)
    with open(dict_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            word = parts[0]
            phones_str = parts[-1]
            if "spn" in phones_str:
                continue
            prob = 1.0
            if len(parts) >= 6:
                try:
                    prob = float(parts[1])
                except ValueError:
                    prob = 1.0
            word2entries[word].append((prob, phones_str.split()))
    for entries in word2entries.values():
        entries.sort(key=lambda x: -x[0])
    return dict(word2entries)


def load_existing_dict(dict_path: str) -> dict[str, list[str]]:
    return {w: es[0][1] for w, es in load_dict_entries(dict_path).items()}


def sample_entry(entries: list[tuple[float, list[str]]], rng) -> list[str]:
    """実現形バリアントを確率に比例してサンプル。"""
    tot = sum(p for p, _ in entries)
    r = rng.random() * tot
    acc = 0.0
    for p, ph in entries:
        acc += p
        if r <= acc:
            return ph
    return entries[0][1]


def load_phone_overrides(path: str | None) -> dict[str, list[str]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    raw = json.load(open(p, encoding="utf-8"))
    raw_overrides = raw.get("overrides", raw)
    overrides = {}
    for word, phones in raw_overrides.items():
        if isinstance(phones, str):
            phones = phones.split()
        overrides[str(word)] = [str(x) for x in phones]
    return overrides


def load_preserve_phrases(path: str | None) -> list[str]:
    if not path:
        return []
    raw = json.load(open(path, encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("phrases", [])
    phrases = [str(x) for x in raw if str(x)]
    return sorted(set(phrases), key=len, reverse=True)


def segment_longest(
    text: str,
    *,
    dict_words: set[str],
    preserve_phrases: list[str],
    max_dict_word_len: int,
    particle_wa: bool = False,
    wa_ok_words: frozenset[str] = frozenset(),
) -> list[str]:
    tokens = []
    i = 0
    while i < len(text):
        if is_punct(text[i]):
            i += 1
            continue

        matched = None
        for phrase in preserve_phrases:
            if text.startswith(phrase, i):
                matched = phrase
                break
        if matched is not None:
            tokens.append(matched)
            i += len(matched)
            continue

        if particle_wa and text[i] == "は" and i > 0:
            tokens.append("は")
            i += 1
            continue

        next_phrase_start = None
        for phrase in preserve_phrases:
            pos = text.find(phrase, i + 1)
            if pos >= 0 and (next_phrase_start is None or pos < next_phrase_start):
                next_phrase_start = pos

        max_len = min(max_dict_word_len, len(text) - i)
        if next_phrase_start is not None:
            max_len = min(max_len, next_phrase_start - i)
        for n in range(max_len, 0, -1):
            cand = text[i:i + n]
            if has_punct(cand):
                continue
            # particle-wa（eval専用）時は、 末尾が「は」の多字辞書語（それは 等）を採らない。
            # → 「は」を後段の particle 分割に回し topic 助詞として wa 読みさせる
            #   （これをしないと辞書が それは を1語で巻き込み は が ha 読みになる）。
            # 注: 「w a 終端の辞書語は許可」の一般則は 今日は→こんにちは の別語義を
            #   拾う事故があり撤回 (2026-07-18)。実は 等の例外は発音 override で登録する。
            if particle_wa and n > 1 and cand.endswith("は") and cand not in wa_ok_words:
                continue
            if cand in dict_words:
                matched = cand
                break
        if matched is not None:
            tokens.append(matched)
            i += len(matched)
        else:
            tokens.append(text[i])
            i += 1
    return tokens


import math

_UNI_TOTAL_CACHE = {}


def _lp(word, unigram) -> float:
    tot = _UNI_TOTAL_CACHE.get(id(unigram))
    if tot is None:
        tot = sum(unigram.values()) + 0.5 * (len(unigram) + 1)
        _UNI_TOTAL_CACHE[id(unigram)] = tot
    c = unigram.get(word, 0) if word is not None else 0
    return math.log((c + 0.5) / tot)


_SUDACHI = None


def _get_sudachi():
    """SudachiPy tokenizer (mode C) を遅延初期化。学習時 MFA JapaneseTokenizer と同系統。"""
    global _SUDACHI
    if _SUDACHI is None:
        from sudachipy import Dictionary, SplitMode
        _SUDACHI = (Dictionary(dict="core").create(), SplitMode.C)
    return _SUDACHI


def segment_sudachi(text: str) -> list[tuple[str, str, str]]:
    """SudachiPy 形態素分割。(surface, 読みカタカナ, 品詞大分類) のリスト。句読点は除く。
    自前分割 (greedy/dp) の過少分割バグ (暇な時|間 → 時間 が とき+あいだ) を、
    本物の形態素解析器で根本解消する。OOV は読みから kana2phone で埋める。"""
    tok, mode = _get_sudachi()
    out = []
    for m in tok.tokenize(text, mode):
        s = m.surface().strip()
        if not s or is_punct(s):
            continue
        out.append((s, m.reading_form(), m.part_of_speech()[0]))
    return out


# 複合語再結合時に「またいで結合してはいけない」品詞 (暇|な|時 の過剰結合を防ぐ)。
_REMERGE_BLOCK_POS = {"助詞", "助動詞", "補助記号", "空白", "記号"}


def _is_subseq(a: list, b: list) -> bool:
    i = 0
    for x in b:
        if i < len(a) and a[i] == x:
            i += 1
    return i == len(a)


def remerge_dict_compounds(
    triples: list[tuple[str, str, str]], phones_map: dict,
    max_window: int = 4, curated: set | None = None,
) -> list[tuple[str, str, str]]:
    """Sudachi が細分割した複合語を、連接表層が辞書見出しにあり かつ 各構成トークンの
    Sudachi 読みが併合後の辞書読みに (母音骨格で) 保存される場合だけ再結合する。
    何|年 → 何年 (なんねん、両成分保存) は回収し、同じ|よう → 同じよう (辞書 どうじよう は
    同じ=おなじ を壊す) は拒否。助詞/助動詞をまたぐ結合も禁止 (暇な時 過剰結合防止)。
    phones_map: surface→音素列 (辞書+override)。curated: 手キュレーション override の表層集合。
    curated に完全一致する表層は Sudachi の誤読(食い|気|味→けあじ 等)に依らず優先再結合する。
    返り値: (surface, 読み, 品詞) のリスト。"""
    out: list[tuple[str, str, str]] = []
    curated = curated or set()
    i, n = 0, len(triples)
    while i < n:
        best_j = None
        for j in range(min(n, i + max_window), i + 1, -1):
            comp = triples[i:j]
            if any(c[2] in _REMERGE_BLOCK_POS for c in comp):
                continue
            surf = "".join(c[0] for c in comp)
            ph = phones_map.get(surf)
            if ph is None:
                continue
            # 手キュレーション override は Sudachi 分割/読みより優先 (母音骨格チェック免除)。
            if surf in curated:
                best_j = j
                break
            merged_v = _vowel_skeleton(ph)
            # 各成分の読み母音が併合後辞書読みに保存されるか (別語化を防ぐ)
            if all(_is_subseq(_vowel_skeleton(kana_to_phones(c[1])), merged_v)
                   for c in comp):
                best_j = j
                break
        if best_j is not None:
            comp = triples[i:best_j]
            out.append(("".join(c[0] for c in comp),
                        "".join(c[1] for c in comp), "複合"))
            i = best_j
        else:
            out.append(triples[i])
            i += 1
    return out


# 辞書の見出しは表層引きで文脈非依存のため、別語の読みを指すことがある
# (何→なに(実は なん)、炊い→たかい(実は たい)、同じよう→どうじよう(実は おなじよう))。
# 全トークンで、辞書読みが Sudachi の文脈読みと「母音骨格」で一致すれば辞書を使い
# (無声化/長音/連濁/鼻音異音は母音を変えないので保持)、母音が食い違えば辞書は別語
# → Sudachi 読みを採用する。これで読み修正を 1 つの規則に統一する。
_VOW_BASE = {"a", "i", "ɯ", "e", "o", "ɨ"}


def _vowel_skeleton(phones: list[str]) -> list[str]:
    out = []
    for p in phones:
        b = p.replace("ː", "").replace("̥", "")
        if b in _VOW_BASE:
            out.append("ɯ" if b == "ɨ" else b)  # ɨ~ɯ は同一視
    return out


@_lru_cache(maxsize=500000)
def _dict_is_reduction_cached(dict_ph: tuple, read_ph: tuple) -> bool:
    dv = _vowel_skeleton(dict_ph)
    rv = _vowel_skeleton(read_ph)
    i = 0
    for x in rv:
        if i < len(dv) and dv[i] == x:
            i += 1
    return i == len(dv)


def _dict_is_reduction(dict_ph: list[str], read_ph: list[str]) -> bool:
    """辞書音素の母音列が Sudachi 読みの母音列の部分列 (= 同語。無声化脱落・長音簡約を許容)
    なら True。False なら母音が食い違う別語読み → Sudachi 読みを優先すべき。
    連濁・無声化・鼻音異音 (ʑ/dʑ, ん→n/ɲ/ŋ 等) は母音を変えないので誤判定しない。
    (surface,reading) 対は大量に重複するためタプル化してメモ化。"""
    return _dict_is_reduction_cached(tuple(dict_ph), tuple(read_ph))


def resolve_gemination(token_phones_all: list[list[str]], sources: list[str],
                       readings: list[str] | None = None) -> None:
    """kana2phone が付けた語末促音 sentinel GEM を、次トークン頭子音の geminate (Cː) で解決。
    辞書引き等で GEM が付かない語末促音 (だっ/行っ 型: 読みが っ で終わる) も同様に越境 gemination する。
    さらに kana2phone 由来トークン末尾の撥音 ɴ を、次トークン頭子音で文脈異音化する。in-place。"""
    from kaburi_tts.g2p.kana2phone import GEM, _moraic_n, _VOWEL

    def next_onset(idx: int):
        for k in range(idx + 1, len(token_phones_all)):
            if token_phones_all[k]:
                return token_phones_all[k], 0
        return None, None

    def geminate_next(idx: int) -> None:
        nxt, pos = next_onset(idx)
        if nxt and nxt[pos] not in _VOWEL and not nxt[pos].endswith("ː"):
            nxt[pos] = nxt[pos] + "ː"  # 次子音を geminate

    for ti, tp in enumerate(token_phones_all):
        if not tp:
            continue
        if tp[-1] == GEM:
            tp.pop()
            geminate_next(ti)
        elif (readings is not None and ti < len(readings) and readings[ti]
              and readings[ti][-1] in ("っ", "ッ")):
            # 辞書/override/reading_ctx 由来で GEM sentinel が無い語末促音 (だっ+た→d a tː a、
            # 行っ+た→i tː a)。 GEM 分岐で処理済の kana2phone は tp[-1]==GEM で先に捕捉されるため二重化しない。
            geminate_next(ti)
        elif sources[ti] == "kana2phone" and tp and tp[-1] == "ɴ":
            nxt, pos = next_onset(ti)
            if nxt:
                tp[-1] = _moraic_n(nxt[pos])


def segment_dp(
    text: str,
    *,
    dict_words: set[str],
    preserve_phrases: list[str],
    max_dict_word_len: int,
    particle_wa: bool = False,
    wa_ok_words: frozenset[str] = frozenset(),
    unigram: dict | None = None,
) -> list[str]:
    """DP 分割 (最小コスト Viterbi)。greedy 最長一致の「ねど|うし|ても」型の
    誤分割 (辞書語 どうしても に到達できない) を解消する一様なアルゴリズム改良。
    コスト: unigram があれば -log P(語) (コーパス頻度、加算 0.5 平滑化) —
    好き|な vs 好|きな のようなトークン数・均衡が同値のタイも語彙頻度で一意に解ける。
    unigram なしのフォールバックは 辞書語=1 / 未知単字=5 + 均衡タイブレーク。
    particle_wa の規則 (は 分割・wa_ok 例外) は greedy 版と同一。"""
    n = len(text)
    INF = float("inf")
    best = [(INF, 0.0)] * (n + 1)   # (cost, -sum_len2)
    back = [None] * (n + 1)
    best[0] = (0.0, 0.0)
    for i in range(n):
        if best[i][0] == INF:
            continue
        if is_punct(text[i]):
            if best[i] < best[i + 1]:
                best[i + 1] = best[i]
                back[i + 1] = (i, None)
            continue
        if particle_wa and text[i] == "は" and i > 0:
            wa_cost = -_lp("は", unigram) if unigram else 1.0
            cand_cost = (best[i][0] + wa_cost, best[i][1] + 1)
            if cand_cost < best[i + 1]:
                best[i + 1] = cand_cost
                back[i + 1] = (i, "は")
            continue
        for L in range(1, min(max_dict_word_len, n - i) + 1):
            w = text[i:i + L]
            if has_punct(w) and L > 1:
                break
            if particle_wa and L > 1 and w.endswith("は") and w not in wa_ok_words:
                continue
            known = (w in dict_words) or (w in preserve_phrases)
            if L > 1 and not known:
                continue
            if unigram is not None:
                cost = -_lp(w, unigram) if known else -_lp(None, unigram) * 1.2
            else:
                cost = 1.0 if known else 5.0
            cand = (best[i][0] + cost, best[i][1] + L * L)
            if cand < best[i + L]:
                best[i + L] = cand
                back[i + L] = (i, w)
    if best[n][0] == INF:
        return segment_longest(text, dict_words=dict_words, preserve_phrases=preserve_phrases,
                               max_dict_word_len=max_dict_word_len, particle_wa=particle_wa,
                               wa_ok_words=wa_ok_words)
    toks = []
    j = n
    while j > 0:
        i, w = back[j]
        if w is not None:
            toks.append(w)
        j = i
    return toks[::-1]


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from kaburi_tts import ASSETS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phone-vocab", default=str(ASSETS_DIR / "phone_vocab.json"))
    ap.add_argument("--dictionary", default=str(ASSETS_DIR / "g2p" / "japanese_mfa.dict"))
    ap.add_argument("--pronunciation-overrides", default=None)
    ap.add_argument("--preserve-phrases", default=None)
    ap.add_argument("--max-dict-word-len", type=int, default=12)
    ap.add_argument("--particle-wa", action="store_true",
                    help="Eval-only heuristic: split non-initial は as the topic particle so it can read as wa.")
    ap.add_argument("--strict", action="store_true", help="Fail on missing words or phones outside vocab.")
    ap.add_argument("--segment-mode", choices=["greedy", "dp", "sudachi"], default="greedy",
                    help="dp = 最小コスト Viterbi 分割 (greedy 誤分割を一部解消)。"
                         "sudachi = SudachiPy 形態素解析 (学習時 MFA tokenizer と同系統、"
                         "過少分割バグ 時間→とき+あいだ を根本解消、OOV は読み→音素)。"
                         "既定 greedy = 従来互換")
    ap.add_argument("--sample-pronunciations", action="store_true",
                    help="発音を最高確率エントリ固定でなく MFA 辞書の変異確率からサンプルする"
                         " (無声化母音・縮約形などの実現形の揺らぎを復元)")
    ap.add_argument("--pron-seed", type=int, default=0)
    args = ap.parse_args()

    spec = json.load(open(args.spec, encoding="utf-8"))
    vocab = json.load(open(args.phone_vocab, encoding="utf-8"))["phone_vocab"]
    dict_entries = load_existing_dict(args.dictionary)
    overrides = load_phone_overrides(args.pronunciation_overrides)
    preserve_phrases = load_preserve_phrases(args.preserve_phrases)
    preserve_phrases = sorted(set(preserve_phrases) | set(overrides.keys()), key=len, reverse=True)
    # は-終わり語の 1 語マッチは、キュレーション済み override に載っている語のみ許可。
    # (辞書からの自動判定は 今日は→こんにちは の別語義を拾う事故があり不可)
    wa_ok_words = frozenset(w for w in overrides if len(w) > 1 and w.endswith("は"))

    seg_unigram = None
    if args.segment_mode == "dp":
        up = ASSETS_DIR / "g2p" / "token_unigram_counts.json"
        seg_unigram = json.loads(up.read_text()) if up.exists() else None

    dict_variants = None
    if args.sample_pronunciations:
        import random
        dict_variants = load_dict_entries(args.dictionary)

    out = {"dialogs": [], "speaker_combos": spec["speaker_combos"]}
    issues = []
    for dialog in spec["dialogs"]:
        nd = {"id": dialog["id"], "title": dialog.get("title", dialog["id"]), "utts": []}
        for u_idx, utt in enumerate(dialog["utts"]):
            readings = None
            poss = None
            if args.segment_mode == "sudachi":
                triples = segment_sudachi(utt["text"])
                pairs = remerge_dict_compounds(
                    triples, {**dict_entries, **overrides}, curated=set(overrides))
                tokens = [s for s, _, _ in pairs]
                readings = [r for _, r, _ in pairs]
                poss = [p for _, _, p in pairs]
            elif args.segment_mode == "dp":
                tokens = segment_dp(
                    utt["text"],
                    dict_words=set(dict_entries.keys()) | set(overrides.keys()),
                    preserve_phrases=preserve_phrases,
                    max_dict_word_len=args.max_dict_word_len,
                    particle_wa=args.particle_wa,
                    wa_ok_words=wa_ok_words,
                    unigram=seg_unigram,
                )
            else:
                tokens = segment_longest(
                    utt["text"],
                dict_words=set(dict_entries.keys()) | set(overrides.keys()),
                preserve_phrases=preserve_phrases,
                max_dict_word_len=args.max_dict_word_len,
                particle_wa=args.particle_wa,
                wa_ok_words=wa_ok_words,
            )
            rng_utt = None
            if dict_variants is not None:
                rng_utt = random.Random(f"{args.pron_seed}:{dialog['id']}:{u_idx}:{utt['text']}")
            phones = []
            token_phones_all = []
            sources = []
            for ti, token in enumerate(tokens):
                if token in overrides:
                    token_phones = overrides[token]
                    source = "override"
                elif token in dict_entries:
                    token_phones = dict_entries[token]
                    source = "dictionary"
                    # 辞書読みが Sudachi 文脈読みと母音骨格で食い違う=別語読み なら読みを採用
                    # (何→なん, 炊い→たい, 同じよう→おなじよう を 1 規則で。連濁等は保持)
                    if poss is not None and readings[ti]:
                        _k2p = kana_to_phones(readings[ti])
                        if not _dict_is_reduction(token_phones, _k2p):
                            token_phones = _k2p
                            source = "reading_ctx"
                    if source == "dictionary" and rng_utt is not None and len(dict_variants.get(token, [])) > 1:
                        cand = sample_entry(dict_variants[token], rng_utt)
                        if all(p in vocab for p in cand):  # vocab 外変異は最頻形に留める
                            token_phones = cand
                            if cand != dict_entries[token]:
                                source = "dictionary_variant"
                elif readings is not None and readings[ti]:
                    # sudachi モード OOV: 読み (カタカナ) から音素を生成
                    token_phones = kana_to_phones(readings[ti])
                    source = "kana2phone"
                    if not token_phones:
                        issues.append(f"{dialog['id']}[{u_idx}] kana2phone empty: {token}({readings[ti]})")
                else:
                    token_phones = []
                    source = "missing"
                    issues.append(f"{dialog['id']}[{u_idx}] missing token: {token}")
                phones.extend(token_phones)
                token_phones_all.append(list(token_phones))
                sources.append(source)

            if args.segment_mode == "sudachi":
                # 語末促音 GEM の越境 gemination・OOV 末尾撥音の文脈異音を解決
                resolve_gemination(token_phones_all, sources, readings)
                phones = [p for tp in token_phones_all for p in tp]

            phone_ids = []
            for phone in phones:
                if phone in vocab:
                    phone_ids.append(int(vocab[phone]))
                else:
                    issues.append(f"{dialog['id']}[{u_idx}] phone outside vocab: {phone}")

            nd["utts"].append({
                "speaker": utt["speaker"],
                "text": utt["text"],
                "tokens": tokens,
                "token_phone_sources": sources,
                "token_phones": token_phones_all,
                "phones": phones,
                "phone_ids": phone_ids,
                "n_phones": len(phone_ids),
            })
        print(f"[dialog] {nd['id']} utts={len(nd['utts'])} phones={sum(u['n_phones'] for u in nd['utts'])}", flush=True)
        out["dialogs"].append(nd)

    if issues:
        print(f"[warn] issues={len(issues)}", flush=True)
        for issue in issues[:50]:
            print(f"  {issue}", flush=True)
        if args.strict:
            raise SystemExit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[save] {out_path}", flush=True)


if __name__ == "__main__":
    main()

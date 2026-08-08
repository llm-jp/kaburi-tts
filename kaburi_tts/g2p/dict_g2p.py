"""Eval/test dialog text -> phone ids using MFA dictionary longest match.

This helper is intentionally separate from g2p_dialog_mfa.py.  It does not
change the training-time path.  Use it only for hand-authored eval/test dialogs
where pronunciation quality matters more than reproducing the Sudachi token
boundary behavior used in training.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def fix_devoicing_context(token_phones: list[list[str]]) -> set[int]:
    """トークン末尾の無声化母音を文脈で修正 (第 1 パス、ポーズ未挿入前提)。
    次トークンの頭が有声なら平母音へ戻す。最終トークン (発話末) は保持。
    返り値 = 置換したトークン index 集合 (ポーズ挿入後の復元判定に使う)。"""
    fixed = set()
    for i, ph in enumerate(token_phones):
        if not ph or ph[-1] not in DEVOICED2PLAIN:
            continue
        if i + 1 >= len(token_phones):
            continue  # 発話末 → 無声化保持
        nxt = token_phones[i + 1]
        if nxt and not _is_voiceless_onset(nxt[0]):
            ph[-1] = DEVOICED2PLAIN[ph[-1]]
            fixed.add(i)
    return fixed


PLAIN2DEVOICED = {v: k for k, v in DEVOICED2PLAIN.items()}


def restore_devoicing_before_pause(tokens: list[str], token_phones: list[list[str]],
                                   fixed_orig_idx: set[int]) -> int:
    """ポーズ挿入後の第 2 パス: 挿入 <pause> の直前トークンが第 1 パスで平母音化した
    ものなら無声化形へ復元 (ポーズ前は実音声でも無声化するため)。"""
    n = 0
    orig = -1
    for i, tok in enumerate(tokens):
        if tok == PAUSE_TOKEN_NAME:
            continue
        orig += 1
        if orig in fixed_orig_idx and i + 1 < len(tokens) and tokens[i + 1] == PAUSE_TOKEN_NAME:
            ph = token_phones[i]
            if ph and ph[-1] in PLAIN2DEVOICED:
                ph[-1] = PLAIN2DEVOICED[ph[-1]]
                n += 1
    return n


PAUSE_TOKEN_NAME = "<pause>"


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
    ap.add_argument("--insert-pauses", action="store_true",
                    help="PauseTagger (学習ベース) で発話内ポーズ (<sil>) をトークン境界に挿入する")
    ap.add_argument("--pause-model", default=str(ASSETS_DIR / "g2p" / "pause_tagger.pt"))
    ap.add_argument("--pause-mode", choices=["threshold", "count"], default="threshold",
                    help="threshold = 較正済み閾値の境界独立判定 (採用構成) / count = 個数制御 (実験用。"
                         "集計レートは正確だが確信の低い境界にも挿入するため聴感劣化、2026-07-18 撤回)")
    ap.add_argument("--pause-stats", default=str(ASSETS_DIR / "g2p" / "pause_stats.json"))
    ap.add_argument("--pause-seed", type=int, default=0)
    ap.add_argument("--pause-threshold", type=float, default=0.874,
                    help="threshold モードの挿入判定閾値 (calibrate_pause_threshold.py の較正値)")
    ap.add_argument("--fix-devoicing", action="store_true",
                    help="無声化母音を文脈修正 (有声音の前は平母音へ、ポーズ/発話末前は保持)")
    ap.add_argument("--segment-mode", choices=["greedy", "dp"], default="greedy",
                    help="dp = 最小コスト Viterbi 分割 (ねど|うし|ても 型の greedy 誤分割を解消)。"
                         "既定 greedy = 従来互換")
    ap.add_argument("--realize-model", default=None,
                    help="RealizationTagger ckpt。指定時は規範形→実現形変換 (縮約/削除/無声化/"
                         "sil 挿入を統合) を適用し、--insert-pauses/--fix-devoicing を代替する")
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

    realizer = None
    if args.realize_model:
        from kaburi_tts.g2p.realization_model import load_realizer, realize
        realizer = load_realizer(args.realize_model)
        edit_stats_path = ASSETS_DIR / "g2p" / "realization_edit_stats.json"
        realize_edit_stats = json.loads(edit_stats_path.read_text())             if edit_stats_path.exists() else None
        if args.insert_pauses or args.fix_devoicing:
            print("[realize] --insert-pauses/--fix-devoicing は realize に統合されるため無視します",
                  flush=True)
            args.insert_pauses = False
            args.fix_devoicing = False

    seg_unigram = None
    if args.segment_mode == "dp":
        up = ASSETS_DIR / "g2p" / "token_unigram_counts.json"
        seg_unigram = json.loads(up.read_text()) if up.exists() else None

    pause_model = pause_stats = None
    if args.insert_pauses:
        import random
        from kaburi_tts.g2p.pause_insert import (
            drop_accident_pauses, insert_pauses_model, load_pause_stats,
        )
        from kaburi_tts.g2p.pause_model import load_tagger
        pause_model = load_tagger(args.pause_model)
        if args.pause_mode == "count":
            pause_stats = load_pause_stats(args.pause_stats)

    dict_variants = None
    if args.sample_pronunciations:
        import random
        dict_variants = load_dict_entries(args.dictionary)

    out = {"dialogs": [], "speaker_combos": spec["speaker_combos"]}
    issues = []
    n_pauses = 0
    n_devoice_fix = 0
    for dialog in spec["dialogs"]:
        nd = {"id": dialog["id"], "title": dialog.get("title", dialog["id"]), "utts": []}
        for u_idx, utt in enumerate(dialog["utts"]):
            if args.segment_mode == "dp":
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
            for token in tokens:
                if token in overrides:
                    token_phones = overrides[token]
                    source = "override"
                elif token in dict_entries:
                    token_phones = dict_entries[token]
                    source = "dictionary"
                    if rng_utt is not None and len(dict_variants.get(token, [])) > 1:
                        cand = sample_entry(dict_variants[token], rng_utt)
                        if all(p in vocab for p in cand):  # vocab 外変異は最頻形に留める
                            token_phones = cand
                            if cand != dict_entries[token]:
                                source = "dictionary_variant"
                else:
                    token_phones = []
                    source = "missing"
                    issues.append(f"{dialog['id']}[{u_idx}] missing token: {token}")
                phones.extend(token_phones)
                token_phones_all.append(list(token_phones))
                sources.append(source)

            if realizer is not None:
                model_r, sub_targets, tau_r, tau_op = realizer
                pre_ids = [int(vocab[p]) for p in phones if p in vocab]
                tok_final = []
                for tp in token_phones_all:
                    n_in = sum(1 for p in tp if p in vocab)
                    if n_in:
                        tok_final += [0] * (n_in - 1) + [1]
                rids = realize(model_r, sub_targets, tau_r, pre_ids, tok_final,
                               op_threshold=tau_op, vocab=vocab,
                               edit_stats=realize_edit_stats)
                inv_v = {v: k for k, v in vocab.items()}
                phones = [inv_v.get(i, "<sil>") for i in rids]
                tokens = ["<realized>"]
                token_phones_all = [list(phones)]
                sources = ["realized"]
                n_pauses += sum(1 for k, i in enumerate(rids)
                                if i == 1 and 0 < k < len(rids) - 1)

            fixed_idx = set()
            if args.fix_devoicing:
                fixed_idx = fix_devoicing_context(token_phones_all)
                if fixed_idx:
                    n_devoice_fix += len(fixed_idx)
                    phones = [p for tp in token_phones_all for p in tp]
            if pause_model is not None:
                pre_ids = [int(vocab[p]) for p in phones if p in vocab]
                rng = random.Random(f"{args.pause_seed}:{dialog['id']}:{u_idx}:{utt['text']}")
                tokens, token_phones_all, ins = insert_pauses_model(
                    tokens, token_phones_all, pre_ids,
                    model=pause_model, mode=args.pause_mode, stats=pause_stats,
                    rng=rng, threshold=args.pause_threshold)
                if ins:
                    tokens, token_phones_all, dropped = drop_accident_pauses(tokens, token_phones_all)
                    for r in dropped:
                        print(f"[pause-drop] {dialog['id']}[{u_idx}] {r}", flush=True)
                    # sources を tokens に沿って再構築 (<pause> 位置に "pause" を差し込む)
                    base_sources = iter(sources)
                    sources = ["pause" if t == "<pause>" else next(base_sources) for t in tokens]
                    n_pauses += tokens.count("<pause>")
                    if args.fix_devoicing:
                        restore_devoicing_before_pause(tokens, token_phones_all, fixed_idx)
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
    if pause_model is not None:
        print(f"[pause] inserted {n_pauses} intra-utterance <sil>", flush=True)
    if args.fix_devoicing:
        print(f"[devoice] context-fixed {n_devoice_fix} devoiced vowels", flush=True)

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

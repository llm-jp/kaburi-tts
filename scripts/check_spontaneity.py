"""対話テキストの「話し言葉らしさ」チェッカ。

KABURI-TTS は自発対話音声で学習されているため、入力テキストが実際の話し言葉の
分布（短いターン・相槌・フィラー・砕けた終止形）に近いほど自然な合成になる。
本スクリプトは、学習コーパスの実書き起こしと書き言葉的な人手対話テキストで
較正した判別器で、入力対話の話し言葉らしさを 0〜1 でスコアリングする
（スクリプトに埋め込まれているのは集計済みの較正係数のみ）。

usage:
  python scripts/check_spontaneity.py dialog.txt            # スコア + 改善ヒント
  python scripts/check_spontaneity.py dialog.txt --json     # 機械可読出力

LLM で複数案を書き直して本スコアで最良案を選ぶ、という使い方を想定
(assets/spontaneous_rewrite_prompt.txt 参照)。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

# 較正済み判別器 (実書き起こし 1,499 対話 vs 書き言葉的対話 60 本、精度 0.97)
KEYS = ["median_len", "short_ratio", "filler_ratio", "polite_ratio"]
W = [0.7855, 0.9809, 5.0472, -2.4207]
B = 8.3187
MU = [10.8284, 0.2286, 0.1396, 0.094]
SD = [3.9258, 0.1291, 0.1195, 0.147]

FILLERS = ("あー", "えー", "うーん", "えっと", "あの", "なんか", "まあ", "んー",
           "えーと", "うん", "へー", "ほー", "ふーん")
POLITE_RE = re.compile(r"(です|ます|ですか|ますか|でした|ません)ね?よ?か?$")


def parse_dialog(path: Path) -> list[str]:
    utts = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        m = re.match(r"^[ＡＢAB]\s*[:：]\s*(.+)$", ln)
        if m:
            utts.append(m.group(1).strip())
    return utts


def features(utts: list[str]) -> dict:
    n = len(utts)
    lens = [len(u) for u in utts]
    return {
        "median_len": statistics.median(lens),
        "short_ratio": sum(1 for u in utts if len(u) <= 4) / n,
        "filler_ratio": sum(1 for u in utts if any(u.startswith(f) for f in FILLERS)) / n,
        "polite_ratio": sum(1 for u in utts if POLITE_RE.search(u)) / n,
    }


def score(utts: list[str]) -> tuple[float, dict]:
    f = features(utts)
    z = B + sum(w * (f[k] - m) / s for w, k, m, s in zip(W, KEYS, MU, SD))
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z)))), f


def advice(f: dict) -> list[str]:
    tips = []
    if f["filler_ratio"] < 0.08:
        tips.append("フィラー・相槌で始まる発話がほぼありません（例:「あー」「うん」「なんか」「えっと」）。実対話では 15% 程度の発話がこれらで始まります。")
    if f["polite_ratio"] > 0.3:
        tips.append("「です・ます」で終わる発話が多すぎます（実対話では 1 割未満）。言い差しや砕けた終止形を混ぜてください。")
    if f["short_ratio"] < 0.15:
        tips.append("短い相槌ターン（4 文字以下、「うん」「そうそう」等）が少なめです。")
    return tips


def main() -> None:
    ap = argparse.ArgumentParser(description="dialogue text spontaneity checker")
    ap.add_argument("input", help="対話テキスト（1 行 1 発話、'A: ...' / 'B: ...'）")
    ap.add_argument("--json", action="store_true", help="機械可読出力")
    args = ap.parse_args()
    utts = parse_dialog(Path(args.input))
    if len(utts) < 3:
        sys.exit("発話が少なすぎます (3 発話以上必要)")
    s, f = score(utts)
    if args.json:
        print(json.dumps({"score": s, "features": f}, ensure_ascii=False))
        return
    print(f"話し言葉らしさスコア: {s:.2f}  (1.0=実対話並み / 0.0=書き言葉的)")
    for k, v in f.items():
        print(f"  {k}: {v:.2f}")
    if s < 0.5:
        print("\n[改善ヒント]")
        for t in advice(f):
            print(f"  - {t}")
        print("  - assets/spontaneous_rewrite_prompt.txt を LLM に渡して書き直し、"
              "複数案を本スクリプトで比較するのが手軽です。")


if __name__ == "__main__":
    main()

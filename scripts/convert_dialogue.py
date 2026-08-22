#!/usr/bin/env python
"""対話テキストを「話し言葉」に変換する (KABURI-TTS の前処理)。

usage:
  python scripts/convert_dialogue.py in.txt -o out.txt
  python scripts/convert_dialogue.py in.txt -o out.txt --converter-ckpt <local_dir>

入力/出力とも 1 行 1 発話「A: 〜」「B: 〜」。変換後は短いターン・相槌・
フィラー入りの話し言葉になり、そのまま kaburi_cli.py に渡せる。
kaburi_cli.py --convert でも同じ変換を合成と一括で行える。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from kaburi_tts.textconv import KaburiTextConverter, _parse_lines  # noqa: E402
from check_spontaneity import score as spont_score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="dialogue text -> spoken-style text")
    ap.add_argument("input", help="対話テキスト（1 行 1 発話 'A: ...' / 'B: ...'）")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--converter-ckpt", default=None,
                    help="local ckpt dir; omit to download from HF")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temperature", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=None,
                    help="optional torch RNG seed for reproducible sampled conversion")
    args = ap.parse_args()

    parsed, n_bad = _parse_lines(Path(args.input).read_text(encoding="utf-8"))
    if not parsed:
        sys.exit("入力に 'A: ...' / 'B: ...' 形式の行がありません")
    if n_bad:
        print(f"[warn] 形式外の行 {n_bad} 行を無視しました", file=sys.stderr)
    utts = [{"speaker": s, "text": t} for s, t in parsed]
    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    if args.converter_ckpt:
        conv = KaburiTextConverter(args.converter_ckpt, device=args.device,
                                   temperature=args.temperature)
    else:
        conv = KaburiTextConverter.from_hf(device=args.device,
                                           temperature=args.temperature)
    out_utts = conv.convert_utts(utts)
    Path(args.output).write_text(
        "\n".join(f"{u['speaker']}: {u['text']}" for u in out_utts) + "\n",
        encoding="utf-8")

    if len(utts) >= 3 and len(out_utts) >= 3:
        s_in, _ = spont_score([u["text"] for u in utts])
        s_out, _ = spont_score([u["text"] for u in out_utts])
        print(f"話し言葉らしさ: {s_in:.2f} -> {s_out:.2f}")
    print(f"{len(utts)} 発話 -> {len(out_utts)} 発話, wrote {args.output}")


if __name__ == "__main__":
    main()

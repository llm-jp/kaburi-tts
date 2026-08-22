#!/usr/bin/env python
"""KABURI-TTS 簡易 CLI: 対話テキストファイル -> 2ch ステレオ wav。

使い方:
  # 1) 参照音声 2 本から ref pack を作る（初回のみ）
  python scripts/make_ref_pack.py --ref-a voiceA.wav --ref-b voiceB.wav --out-dir refpack/

  # 2) 対話テキストを合成
  python scripts/kaburi_cli.py dialog.txt --ref-pack refpack/
  python scripts/kaburi_cli.py dialog.txt --ref-pack refpack/ --convert    # 書き言葉を話し言葉に自動変換
  python scripts/kaburi_cli.py dialog.txt --ref-pack refpack/ --mode stat   # 統計配置 timing
  python scripts/kaburi_cli.py dialog.txt --ref-pack refpack/ --paper-mode  # 論文版
  python scripts/kaburi_cli.py dialog.txt --ref-pack refpack/ -o out.wav --device cuda:0

入力テキスト（1 行 1 発話。 話者は A / B）:
  A: 最近ゲームにはまってて
  B: へえ
  A: 気づいたら朝になってるんですよね
  B: それはやりすぎですね

- 左チャネル=話者A、 右チャネル=話者B。
- --mode pred は学習済み realizer と gap model、 --mode stat は統計配置で音素ラスタを作る。
- 記号・数字・英字・固有名詞は読み上げが不安定になりやすいので避ける。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
G2P = REPO / "kaburi_tts/g2p/dict_g2p.py"
SYNTH = REPO / "scripts/synth_from_text.py"
SYNTH_RASTER = REPO / "scripts/synth_raster.py"
DIALOG_ID = "sample"


def parse_dialog(path: Path) -> list[dict]:
    utts = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        # "A: text" / "A： text" / "Ａ:text" を許容
        sep = None
        for c in (":", "："):
            if c in ln:
                sep = c
                break
        if sep is None:
            print(f"[warn] skip (話者ラベルなし): {ln}", file=sys.stderr)
            continue
        who, text = ln.split(sep, 1)
        who = who.strip().upper().replace("Ａ", "A").replace("Ｂ", "B")
        text = text.strip()
        if who not in ("A", "B") or not text:
            print(f"[warn] skip (A/B でない or 空): {ln}", file=sys.stderr)
            continue
        utts.append({"speaker": who, "text": text})
    return utts


def main() -> None:
    ap = argparse.ArgumentParser(description="KABURI-TTS: dialogue text -> stereo wav")
    ap.add_argument("input", help="対話テキストファイル（1 行 1 発話, 'A: ...' / 'B: ...'）")
    ap.add_argument("--mode", choices=["pred", "stat"], default="pred",
                    help="pred=学習ベース timing / stat=統計配置（既定 pred）")
    ap.add_argument("-o", "--out", default=None, help="出力 wav（既定=入力名.wav）")
    ap.add_argument("--device", default="cuda:0")
    # 経路 1: ref pack (外部ユーザ向け、 make_ref_pack.py で生成)
    ap.add_argument("--ref-pack", default=None,
                    help="make_ref_pack.py の出力ディレクトリ（manifest / refs / utt_timing を含む）")
    # 経路 2: 内部データ保有者向け（コーパス manifest + template chunk）
    ap.add_argument("--manifest", default=None, help="chunk manifest jsonl")
    ap.add_argument("--speakerA", default=None)
    ap.add_argument("--speakerB", default=None)
    ap.add_argument("--template_chunk", default=None,
                    help="話者参照に使う chunk（--manifest 使用時に指定）")
    ap.add_argument("--utt-timing-jsonl", default=None,
                    help="per-utt timing record jsonl（--manifest 使用時、 pred mode で指定）")
    ap.add_argument("--acoustic-ckpt", default=None, help="local ckpt; omit to download from HF")
    ap.add_argument("--realizer-ckpt", default=None, help="realizer local ckpt; omit to download from HF")
    ap.add_argument("--gap-ckpt", default=None, help="gap model local ckpt; omit to download from HF")
    ap.add_argument("--raster-release", default=None,
                    help="使用する raster release (既定 = 最新)")
    ap.add_argument("--no-activity-gate", action="store_true",
                    help="無発話区間の activity gate を無効化 (debug 用)")
    ap.add_argument("--gap-bias-frames", type=float, default=0.0,
                    help="論文版 predictor の gap 補正 (frame, 25fps)。既定 0")
    ap.add_argument("--paper-mode", action="store_true",
                    help="論文版で合成する（規範形 G2P + timing predictor）")
    ap.add_argument("--predictor-ckpt", default=None,
                    help="論文版 timing predictor の local ckpt; omit to download from HF")
    ap.add_argument("--cfg-scale", type=float, default=2.5,
                    help="音響 CFG スケール（release-eval / 論文と同一の 2.5）")
    ap.add_argument("--convert", action="store_true",
                    help="合成前にテキストコンバータで話し言葉に自動変換する "
                         "(書き言葉的な入力の品質低下を防ぐ)")
    ap.add_argument("--converter-ckpt", default=None,
                    help="コンバータの local ckpt dir; omit to download from HF")
    ap.add_argument("--converter-temperature", type=float, default=0.55,
                    help="text converter sampling temperature")
    ap.add_argument("--converter-seed", type=int, default=None,
                    help="optional torch RNG seed for reproducible text conversion")
    ap.add_argument("--keep-temp", action="store_true", help="中間ファイルを消さない（デバッグ用）")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"入力が見つかりません: {in_path}")
    out_path = Path(args.out) if args.out else in_path.with_suffix(".wav")

    # 参照経路の解決
    utt_timing_jsonl = None
    if args.ref_pack:
        pack = Path(args.ref_pack)
        manifest = pack / "manifest.jsonl"
        if not manifest.exists():
            sys.exit(f"ref pack に manifest.jsonl がありません: {pack}（make_ref_pack.py で生成してください）")
        row = json.loads(manifest.read_text().splitlines()[0])
        spk_a, spk_b = row["speaker_A"], row["speaker_B"]
        template_chunk = row["chunk_id"]
        utt_timing_jsonl = pack / "utt_timing.jsonl"
    elif args.manifest:
        manifest = Path(args.manifest)
        spk_a, spk_b = args.speakerA, args.speakerB
        template_chunk = args.template_chunk
        if not (spk_a and spk_b and template_chunk):
            sys.exit("--manifest 使用時は --speakerA/--speakerB/--template_chunk を指定してください")
        if args.utt_timing_jsonl:
            utt_timing_jsonl = Path(args.utt_timing_jsonl)
    else:
        sys.exit("--ref-pack か --manifest のどちらかを指定してください")

    utts = parse_dialog(in_path)
    if not utts:
        sys.exit("有効な発話がありません（'A: ...' / 'B: ...' 形式で記述してください）")

    if args.convert:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        import torch
        from kaburi_tts.textconv import KaburiTextConverter
        if args.converter_seed is not None:
            torch.manual_seed(int(args.converter_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(args.converter_seed))
        conv = (KaburiTextConverter(args.converter_ckpt, device=args.device)
                if args.converter_ckpt else KaburiTextConverter.from_hf(
                    device=args.device, temperature=args.converter_temperature))
        if args.converter_ckpt:
            conv.temperature = float(args.converter_temperature)
        n_before = len(utts)
        utts = conv.convert_utts(utts)
        print(f"[convert] 話し言葉に変換: {n_before} 発話 -> {len(utts)} 発話")

    print(f"[kaburi] {len(utts)} 発話, mode={args.mode}, {spk_a}x{spk_b} -> {out_path}")

    # 入力テキストの話し言葉らしさチェック (= 低いと合成品質が落ちやすい)
    if len(utts) >= 3:
        try:
            from check_spontaneity import advice as spont_advice, score as spont_score
            s, f = spont_score([u["text"] for u in utts])
            if s < 0.5:
                print(f"[warn] 入力テキストが書き言葉的です (話し言葉らしさ {s:.2f})。"
                      "合成品質が落ちる可能性があります。", file=sys.stderr)
                for t in spont_advice(f):
                    print(f"  - {t}", file=sys.stderr)
                print("  - --convert を付けると同梱のテキストコンバータで話し言葉に自動変換します。"
                      "詳細: python scripts/check_spontaneity.py", file=sys.stderr)
        except Exception:
            pass

    work = Path(tempfile.mkdtemp(prefix="kaburi_"))
    try:
        # 1) authored spec
        spec = {"dialogs": [{"id": DIALOG_ID, "title": DIALOG_ID, "utts": utts}], "speaker_combos": []}
        spec_p = work / "spec.json"
        spec_p.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")

        # pred (既定): 実現形化もタイミングも kaburi_tts.raster が一括で行うため
        # G2P 実現形ステップと synth_from_text は使わない。
        use_raster = (args.mode == "pred" and not args.paper_mode)
        if use_raster:
            speakers = {DIALOG_ID: {"A": spk_a, "B": spk_b, "template_chunk": template_chunk}}
            spk_p = work / "speakers.json"
            spk_p.write_text(json.dumps(speakers, ensure_ascii=False), encoding="utf-8")
            out_root = work / "out"
            cmd = [PY, str(SYNTH_RASTER), "--spec", str(spec_p),
                   "--manifest", str(manifest), "--speakers_json", str(spk_p),
                   "--out_root", str(out_root), "--device", args.device,
                   "--cfg_scale", str(args.cfg_scale)]
            if args.acoustic_ckpt:
                cmd += ["--acoustic_ckpt", args.acoustic_ckpt]
            if args.realizer_ckpt:
                cmd += ["--realizer_ckpt", args.realizer_ckpt]
            if args.gap_ckpt:
                cmd += ["--gap_ckpt", args.gap_ckpt]
            if args.raster_release:
                cmd += ["--raster-release", args.raster_release]
            if args.no_activity_gate:
                cmd += ["--no-activity-gate"]
            if subprocess.run(cmd).returncode != 0:
                sys.exit("合成に失敗しました (ラスタ生成)")
            produced = out_root / "raster" / f"{DIALOG_ID}.wav"
            if not produced.exists():
                sys.exit(f"出力 wav が生成されませんでした: {produced}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(produced), str(out_path))
            print(f"[kaburi] 完了: {out_path}")
            return

        # 2) g2p -> phones (stat / --paper-mode)
        phones_p = work / "phones.json"
        cmd = [PY, str(G2P), "--spec", str(spec_p), "--out", str(phones_p), "--particle-wa"]
        # 論文版: 規範形 G2P と timing predictor を使う。
        if subprocess.run(cmd).returncode != 0:
            sys.exit("g2p に失敗しました")

        # 3) speakers
        speakers = {DIALOG_ID: {"A": spk_a, "B": spk_b, "template_chunk": template_chunk}}
        spk_p = work / "speakers.json"
        spk_p.write_text(json.dumps(speakers, ensure_ascii=False), encoding="utf-8")

        # 4) synth
        mode = {"pred": "kaburi_pred", "stat": "kaburi_stat"}[args.mode]
        out_root = work / "out"
        cmd = [PY, str(SYNTH), "--mode", mode, "--device", args.device,
               "--manifest", str(manifest), "--phones_json", str(phones_p),
               "--speakers_json", str(spk_p), "--out_root", str(out_root)]
        if utt_timing_jsonl is not None:
            cmd += ["--utt_timing_jsonls", str(utt_timing_jsonl)]
        if args.acoustic_ckpt:
            cmd += ["--acoustic_ckpt", args.acoustic_ckpt]
        if args.predictor_ckpt:
            cmd += ["--predictor_ckpt", args.predictor_ckpt]
        cmd += ["--gap_bias_frames", str(args.gap_bias_frames)]
        cmd += ["--cfg_scale", str(args.cfg_scale)]
        if subprocess.run(cmd).returncode != 0:
            sys.exit("合成に失敗しました")

        produced = out_root / mode / f"{DIALOG_ID}.wav"
        if not produced.exists():
            sys.exit(f"出力 wav が生成されませんでした: {produced}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(produced), str(out_path))
        print(f"[kaburi] 完了: {out_path}")
    finally:
        if not args.keep_temp:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()

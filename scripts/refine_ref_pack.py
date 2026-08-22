"""参照 pack の対話ドメイン精錬 (bootstrap refinement)。

朗読調・スタジオ品質などドメインの異なる参照は話者条件付けが不安定になる。
本スクリプトは、既存の参照 pack で一度 KABURI-TTS の対話合成を行い、その
出力（= 学習ドメインと同じ話し言葉スタイルの音声）から各チャネルの発話
密度が高い 10 秒を切り出して、参照を作り直す。

  元 pack --(対話合成)--> 2ch 出力 --(高密度切り出し)--> 精錬 pack

usage:
  python scripts/refine_ref_pack.py --ref-pack refpack/ --out-dir refpack_refined/ \
      [--dialog assets/bootstrap_dialog.txt] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torchaudio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from make_ref_pack import densify  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Refine a ref pack via one KABURI synthesis pass")
    ap.add_argument("--ref-pack", required=True, help="元の参照 pack (make_ref_pack.py 出力)")
    ap.add_argument("--out-dir", required=True, help="精錬後 pack の出力先")
    ap.add_argument("--dialog", default=str(REPO / "assets/bootstrap_dialog.txt"),
                    help="bootstrap 用対話テキスト (両話者が十分に喋るもの)")
    ap.add_argument("--mode", choices=["pred", "stat"], default="pred")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--acoustic-ckpt", default=None)
    ap.add_argument("--predictor-ckpt", default=None)
    ap.add_argument("--keep-audio", default=None,
                    help="切り出した参照 wav を残すディレクトリ (ページ掲載用サンプル等)")
    args = ap.parse_args()

    src = Path(args.ref_pack)
    row = json.loads((src / "manifest.jsonl").read_text().splitlines()[0])
    spk_a, spk_b = row["speaker_A"], row["speaker_B"]

    work = Path(tempfile.mkdtemp(prefix="kaburi_refine_"))
    boot_wav = work / "bootstrap.wav"

    # 1) 元 pack で bootstrap 対話を合成
    cmd = [sys.executable, str(REPO / "scripts/kaburi_cli.py"), args.dialog,
           "--ref-pack", str(src), "--mode", args.mode,
           "--device", args.device, "-o", str(boot_wav)]
    if args.acoustic_ckpt:
        cmd += ["--acoustic-ckpt", args.acoustic_ckpt]
    if args.predictor_ckpt:
        cmd += ["--predictor-ckpt", args.predictor_ckpt]
    if subprocess.run(cmd).returncode != 0:
        sys.exit("bootstrap 合成に失敗しました")

    # 2) 各チャネルから高密度 10 秒を切り出し
    wav, sr = torchaudio.load(str(boot_wav))
    assert wav.shape[0] == 2, "2ch 出力を想定"
    refs = {}
    for ch, spk in [(0, spk_a), (1, spk_b)]:
        seg = densify(wav[ch], sr)
        p = work / f"refined_{spk}.wav"
        torchaudio.save(str(p), seg.unsqueeze(0), sr)
        refs[ch] = p
        print(f"[refine] ch{ch} ({spk}): {seg.shape[-1]/sr:.1f}s")

    # 3) 精錬 pack を構築
    r = subprocess.run([sys.executable, str(REPO / "scripts/make_ref_pack.py"),
                        "--ref-a", str(refs[0]), "--ref-b", str(refs[1]),
                        "--out-dir", args.out_dir,
                        "--speaker-a", spk_a, "--speaker-b", spk_b, "--device", "cpu"])
    if r.returncode != 0:
        sys.exit("pack 構築に失敗しました")

    if args.keep_audio:
        keep = Path(args.keep_audio)
        keep.mkdir(parents=True, exist_ok=True)
        for ch, spk in [(0, spk_a), (1, spk_b)]:
            (keep / f"{spk}.wav").write_bytes(refs[ch].read_bytes())
        print(f"[refine] ref audio -> {keep}")
    print(f"[refine] done: {args.out_dir}")


if __name__ == "__main__":
    main()

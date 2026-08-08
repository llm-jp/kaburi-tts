"""Build a self-contained speaker reference pack from two reference wavs.

KABURI-TTS の E2E 合成 (scripts/kaburi_cli.py / synth_from_text.py) は、 話者参照
latent とパイプライン骨格を「template chunk」から取る。 学習コーパスは非公開のため、
外部ユーザは本スクリプトで自分の参照音声 (話者 A/B 各 1 wav、 3〜10 秒推奨) から
同じ構造の pack を生成して使う。

出力 (--out-dir 配下):
  refs/<SPK>_ref_000.pt   参照 latent (= DACVAE encode、 fp32)
  chunks/template.pt      構造のみの空 chunk (= zeros latent + SIL raster)
  manifest.jsonl          1 行 manifest (= 上記 chunk を指す)
  utt_timing.jsonl        1 record (= dummy 1 utt、 predictor dataset の骨格用)

usage:
  python scripts/make_ref_pack.py --ref-a myvoiceA.wav --ref-b myvoiceB.wav \
      --out-dir refpack/ [--speaker-a SPK_A --speaker-b SPK_B] [--device cpu]

生成後:
  python scripts/kaburi_cli.py dialog.txt --ref-pack refpack/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

T_FRAMES = 750
LATENT_DIM = 32
SIL_PHONE_ID = 1
REF_TARGET_SEC = 10.0  # 学習時の参照長 (= 250 frames @ 25fps)


def densify(wav: torch.Tensor, sr: int, *, target_sec: float = REF_TARGET_SEC,
            frame_sec: float = 0.05, thr_rel: float = 0.08, pad_sec: float = 0.08,
            fade_sec: float = 0.01) -> torch.Tensor:
    """参照を学習時の標準 (= 10 秒・発話純度 ~0.94) に揃える。

    無音を除いて有声セグメントを連結し、足りなければループ充填して target_sec に
    切り揃える。学習時の話者参照は 10 秒・ほぼ連続発話で選ばれているため、短い/
    無音の多い参照をそのまま渡すと話者条件付けが弱り、音質が劣化する。
    """
    win = int(frame_sec * sr)
    if wav.shape[0] < 2 * win:
        return wav
    rms = wav.unfold(0, win, win).pow(2).mean(-1).sqrt()
    thr = thr_rel * float(rms.max())
    voiced = (rms > thr).tolist()
    segs = []
    i = 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            s = max(0, int(i * win - pad_sec * sr))
            e = min(wav.shape[0], int(j * win + pad_sec * sr))
            segs.append((s, e))
            i = j
        else:
            i += 1
    fade = int(fade_sec * sr)
    ramp = torch.linspace(0, 1, fade) if fade > 0 else None

    def _faded(seg: torch.Tensor) -> torch.Tensor:
        seg = seg.clone()
        if ramp is not None and seg.shape[0] > 2 * fade:
            seg[:fade] *= ramp
            seg[-fade:] *= ramp.flip(0)
        return seg

    dense = torch.cat([_faded(wav[s:e]) for s, e in segs]) if segs else wav
    target = int(target_sec * sr)
    if dense.shape[0] < target:  # ループ充填 (= 同一話者の繰り返しは speaker 条件付けに無害)
        reps = target // dense.shape[0] + 1
        dense = torch.cat([dense] * reps)
    return dense[:target]


def encode_ref(codec, wav_path: Path, *, max_sec: float, no_densify: bool = False) -> torch.Tensor:
    wav, sr = torchaudio.load(str(wav_path))
    wav = wav.mean(0) if wav.dim() == 2 else wav
    if no_densify:
        wav = wav[: int(max_sec * sr)]
    else:
        wav = densify(wav, sr, target_sec=min(max_sec, REF_TARGET_SEC))
    # DACVAE encode は fp32 で行う (参照 latent の dtype 事故を避ける)
    latent = codec.encode_waveform(wav.float().unsqueeze(0), sr)  # (1, T_latent, D)
    return latent[0].float().cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a speaker reference pack from two wavs")
    ap.add_argument("--ref-a", required=True, help="reference wav for channel A (3-10s recommended)")
    ap.add_argument("--ref-b", required=True, help="reference wav for channel B (3-10s recommended)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--speaker-a", default="SPK_A", help="speaker label for channel A")
    ap.add_argument("--speaker-b", default="SPK_B", help="speaker label for channel B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-ref-sec", type=float, default=10.0)
    ap.add_argument("--no-densify", action="store_true",
                    help="無音除去 + 10 秒ループ充填を行わず、先頭 max_ref_sec をそのまま使う")
    args = ap.parse_args()

    from irodori_tts.codec import DACVAECodec

    out = Path(args.out_dir)
    (out / "refs").mkdir(parents=True, exist_ok=True)
    (out / "chunks").mkdir(parents=True, exist_ok=True)

    codec = DACVAECodec.load(device=args.device, dtype=torch.float32)

    ref_paths = {}
    for ch, spk, wav_path in [("A", args.speaker_a, args.ref_a), ("B", args.speaker_b, args.ref_b)]:
        latent = encode_ref(codec, Path(wav_path), max_sec=args.max_ref_sec, no_densify=args.no_densify)
        ref_p = out / "refs" / f"{spk}_ref_000.pt"
        torch.save({
            "ref_id": f"{spk}_ref_000",
            "speaker": spk,
            "channel": ch,
            "n_frames": int(latent.shape[0]),
            "latent": latent,
        }, ref_p)
        ref_paths[ch] = ref_p.resolve()
        print(f"[ref] {ch}={spk}: {wav_path} -> {ref_p} ({latent.shape[0]} frames)", flush=True)

    chunk_id = "template_chunk_000"
    chunk_p = (out / "chunks" / "template.pt").resolve()
    torch.save({
        "chunk_id": chunk_id,
        "dialogue_id": "template",
        "chunk_idx": 0,
        "start_sec": 0.0,
        "end_sec": T_FRAMES / 25.0,
        "speaker_A": args.speaker_a,
        "speaker_B": args.speaker_b,
        "latent_A": torch.zeros(T_FRAMES, LATENT_DIM),
        "latent_B": torch.zeros(T_FRAMES, LATENT_DIM),
        "phone_A": torch.full((T_FRAMES,), SIL_PHONE_ID, dtype=torch.long),
        "phone_B": torch.full((T_FRAMES,), SIL_PHONE_ID, dtype=torch.long),
        "activity_A": torch.zeros(T_FRAMES),
        "activity_B": torch.zeros(T_FRAMES),
        "usable_chunk": True,
    }, chunk_p)

    # manifest 内は pack 相対パス (= pack ごと移動・配布しても壊れない)
    manifest_p = out / "manifest.jsonl"
    manifest_p.write_text(json.dumps({
        "chunk_id": chunk_id,
        "dialogue_id": "template",
        "chunk_idx": 0,
        "speaker_A": args.speaker_a,
        "speaker_B": args.speaker_b,
        "chunk_path": "chunks/template.pt",
        "ref_A_path": f"refs/{args.speaker_a}_ref_000.pt",
        "ref_B_path": f"refs/{args.speaker_b}_ref_000.pt",
        "usable_chunk": True,
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    # predictor dataset の骨格用 dummy record (= 実際の発話内容は text 経路で完全置換される)
    utt_timing_p = out / "utt_timing.jsonl"
    utt_timing_p.write_text(json.dumps({
        "chunk_id": chunk_id,
        "speaker_A": args.speaker_a,
        "speaker_B": args.speaker_b,
        "n_utts": 1,
        "utterances": [{
            "utt_id": "template_u000",
            "speaker": "A",
            "text": "",
            "phone_ids": [2, 3, 4],
            "n_phones": 3,
            "gt_start_frame": 10,
            "gt_end_frame": 20,
            "gt_region_len": 10,
            "gt_phone_durations": [3, 4, 3],
            "gt_inter_utt_gap": 0,
        }],
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[pack] done: {out}", flush=True)
    print(f"  manifest: {manifest_p}", flush=True)
    print(f"  next: python scripts/kaburi_cli.py <dialog.txt> --ref-pack {out}", flush=True)


if __name__ == "__main__":
    main()

"""Generate raw Irodori utterance-level sequential/statistical placement baselines."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml
from huggingface_hub import hf_hub_download


def _resolve_ckpt(cfg: dict) -> str:
    local = str(cfg["paths"].get("base_checkpoint_local", "")).strip()
    if local:
        p = Path(local).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"base_checkpoint_local not found: {p}")
        return str(p)
    repo = str(cfg["paths"]["base_checkpoint_hf"]).strip()
    return hf_hub_download(repo_id=repo, filename="model.safetensors")


def _load_dialog(phones_json: Path, dialog_id: str, combo_id: str) -> tuple[dict, dict]:
    raw = json.loads(phones_json.read_text(encoding="utf-8"))
    dialog = next((d for d in raw["dialogs"] if str(d.get("id")) == dialog_id), None)
    combo = next((c for c in raw.get("speaker_combos", []) if str(c.get("id")) == combo_id), None)
    if dialog is None:
        raise ValueError(f"dialog not found: {dialog_id}")
    if combo is None:
        raise ValueError(f"speaker combo not found: {combo_id}")
    return dialog, combo


def _duration_budget_sec(n_phones: int) -> float:
    return max(2.0, min(8.0, 1.35 + 0.12 * float(max(1, n_phones))))


def _load_or_synth_utts(args, cfg: dict, dialog: dict, combo: dict) -> tuple[list[dict], int]:
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    from irodori_tts.text_normalization import normalize_text

    ckpt = _resolve_ckpt(cfg)
    sc = cfg["synth"]
    runtime = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=ckpt,
        model_device=args.device,
        codec_repo=str(cfg["paths"]["codec_repo"]),
        model_precision=str(sc.get("model_precision", "fp32")),
        codec_device=args.device,
        codec_precision=str(sc.get("codec_precision", "fp32")),
    ))
    sr = int(runtime.codec.sample_rate)

    ref_by_spk = {
        "A": args.ref_a,
        "B": args.ref_b,
    }
    utts = []
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, u in enumerate(dialog["utts"]):
        spk = str(u["speaker"])
        text = normalize_text(str(u["text"])).strip()
        n_phones = int(u.get("n_phones", len(u.get("phone_ids", []))))
        wav_path = cache_dir / f"{i:02d}_{spk}.wav"
        if not wav_path.exists():
            seconds = _duration_budget_sec(n_phones)
            res = runtime.synthesize(SamplingRequest(
                text=text,
                ref_wav=ref_by_spk[spk],
                seconds=seconds,
                num_steps=int(sc.get("num_steps", 40)),
                cfg_scale_text=float(sc.get("cfg_scale_text", 3.0)),
                cfg_scale_speaker=float(sc.get("cfg_scale_speaker", 5.0)),
                seed=int(args.seed) + i,
                trim_tail=True,
            ))
            torchaudio.save(str(wav_path), res.audio.cpu(), sr, channels_first=True)
            print(f"[utt] {i:02d} {spk} {text} -> {wav_path} ({res.audio.shape[-1] / sr:.2f}s)", flush=True)
        wav, wsr = torchaudio.load(str(wav_path))
        if wsr != sr:
            wav = torchaudio.functional.resample(wav, wsr, sr)
        x = wav.mean(dim=0).numpy().astype(np.float32)
        utts.append({
            "index": i,
            "speaker": spk,
            "text": text,
            "n_phones": n_phones,
            "wav_path": str(wav_path),
            "audio": x,
            "duration_sec": float(x.shape[0] / sr),
        })
    return utts, sr


def _transition_gap_sec(rng: random.Random, prev_spk: str, spk: str, overlap_prob: float) -> float:
    if prev_spk == spk:
        return max(0.0, min(1.1, rng.gauss(0.42, 0.18)))
    if rng.random() < overlap_prob:
        return -rng.uniform(0.08, 0.36)
    return max(-0.12, min(1.1, rng.gauss(0.30, 0.30)))


def _make_placements(utts: list[dict], mode: str, *, seed: int, gap_sec: float, overlap_prob: float) -> list[dict]:
    placements = []
    if mode == "sequential":
        cursor = 0.35
        for u in utts:
            placements.append({**u, "start_sec": cursor})
            cursor += float(u["duration_sec"]) + gap_sec
        return placements
    if mode != "statistical":
        raise ValueError(f"unknown mode: {mode}")
    rng = random.Random(seed)
    channel_cursor = {"A": 0.0, "B": 0.0}
    prev_spk = None
    prev_end = 0.35
    for i, u in enumerate(utts):
        spk = str(u["speaker"])
        if i == 0:
            start = 0.35
        else:
            gap = _transition_gap_sec(rng, str(prev_spk), spk, overlap_prob)
            start = max(channel_cursor[spk], prev_end + gap)
        placements.append({**u, "start_sec": start})
        end = start + float(u["duration_sec"])
        channel_cursor[spk] = max(channel_cursor[spk], end)
        prev_spk = spk
        prev_end = end
    return placements


def _place(placements: list[dict], sr: int, out: Path, *, tail_sec: float) -> dict:
    max_end = max((float(p["start_sec"]) + float(p["duration_sec"]) for p in placements), default=0.0)
    total_n = int(round((max_end + tail_sec) * sr))
    buf = {
        "A": np.zeros(total_n, dtype=np.float32),
        "B": np.zeros(total_n, dtype=np.float32),
    }
    for p in placements:
        x = p["audio"]
        s = int(round(float(p["start_sec"]) * sr))
        e = min(total_n, s + x.shape[0])
        if e > s:
            buf[p["speaker"]][s:e] += x[: e - s]
    stereo = np.stack([buf["A"], buf["B"]], axis=0)
    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if peak > 0.99:
        stereo *= 0.99 / peak
    out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out), torch.from_numpy(stereo), sr, channels_first=True)
    frame_hz = 25.0
    T = max(1, int(np.ceil((total_n / sr) * frame_hz)))
    act_A = np.zeros(T, dtype=np.float32)
    act_B = np.zeros(T, dtype=np.float32)
    for p in placements:
        s = max(0, int(round(float(p["start_sec"]) * frame_hz)))
        e = min(T, int(round((float(p["start_sec"]) + float(p["duration_sec"])) * frame_hz)))
        if e > s:
            (act_A if p["speaker"] == "A" else act_B)[s:e] = 1.0
    meta = {
        "wav": str(out),
        "sample_rate": sr,
        "num_channels": 2,
        "duration_sec": float(total_n / sr),
        "peak_before_norm": peak,
        "speech_ratio_A": float(act_A.mean()),
        "speech_ratio_B": float(act_B.mean()),
        "overlap_ratio": float(((act_A > 0.5) & (act_B > 0.5)).mean()),
        "placements": [
            {
                "index": p["index"],
                "speaker": p["speaker"],
                "text": p["text"],
                "start_sec": float(p["start_sec"]),
                "end_sec": float(p["start_sec"] + p["duration_sec"]),
                "duration_sec": float(p["duration_sec"]),
                "wav_path": p["wav_path"],
            }
            for p in placements
        ],
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="irodori baseline yaml (see configs/irodori_baseline.yaml)")
    ap.add_argument("--phones-json", required=True)
    ap.add_argument("--dialog-id", default="demo_meeting_dialogic")
    ap.add_argument("--speaker-combo-id", default=None)
    ap.add_argument("--ref-a", required=True, help="reference wav for speaker A")
    ap.add_argument("--ref-b", required=True, help="reference wav for speaker B")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--placement-seed", type=int, default=7)
    ap.add_argument("--seq-gap-sec", type=float, default=0.24)
    ap.add_argument("--stat-overlap-prob", type=float, default=0.22)
    ap.add_argument("--tail-sec", type=float, default=0.72)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dialog, combo = _load_dialog(Path(args.phones_json), args.dialog_id, args.speaker_combo_id)
    utts, sr = _load_or_synth_utts(args, cfg, dialog, combo)
    out_dir = Path(args.out_dir)
    rows = []
    for mode, filename in [
        ("sequential", "05_irodori_raw_sequential_stereo.wav"),
        ("statistical", "06_irodori_raw_statistical_stereo.wav"),
    ]:
        placements = _make_placements(
            utts, mode, seed=args.placement_seed,
            gap_sec=args.seq_gap_sec, overlap_prob=args.stat_overlap_prob,
        )
        meta = _place(placements, sr, out_dir / filename, tail_sec=args.tail_sec)
        meta.update({
            "condition": f"irodori_raw_{mode}",
            "dialog_id": args.dialog_id,
            "speaker_combo_id": args.speaker_combo_id,
            "speaker_A": combo.get("A"),
            "speaker_B": combo.get("B"),
            "ref_A": args.ref_a,
            "ref_B": args.ref_b,
            "seed": args.seed,
            "placement_seed": args.placement_seed,
        })
        rows.append(meta)
        print(f"[save] {filename} duration={meta['duration_sec']:.2f}s overlap={meta['overlap_ratio']:.4f}", flush=True)
    (out_dir / "irodori_raw_metadata.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

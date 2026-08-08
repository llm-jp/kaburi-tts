"""Synthesize KABURI-TTS demo audio with non-learned utterance placement baselines.

This script keeps the text, phone sequence, speaker references, acoustic
checkpoint, CFG, and diffusion seed fixed.  Only the phone/activity raster
placement is replaced with simple baselines:

  sequential  : all utterances are concatenated in dialog order, with no overlap.
  statistical : a small hand-coded length/transition model, with occasional
                turn-switch overlap.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torchaudio
import yaml

from irodori_tts.codec import DACVAECodec
from irodori_tts.tokenizer import PretrainedTextTokenizer
from kaburi_tts import ASSETS_DIR
from kaburi_tts.two_stream.dataset import Normal2StreamDataset, collate_eventmix
from kaburi_tts.two_stream.loss import build_phone_class_table_8, build_soft_phone_features
from kaburi_tts.acoustic.infer import build_model as build_acoustic_model, synth as synth_acoustic


SIL_PHONE_ID = 1
FPS = 25.0
OVERLAP_CAP_SEC = 0.3  # turn-switch の被り上限（秒）。 train の被り tail は同時発話混在で過大なため保守的に cap
DEFAULT_PLACEMENT_STATS_JSON = str(ASSETS_DIR / "placement_stats_train.json")


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _find_index(ds, chunk_id: str) -> int | None:
    for i, e in enumerate(ds.entries):
        if str(e.get("chunk_id")) == chunk_id:
            return i
    return None


def _swap_channel_refs(batch: dict) -> dict:
    for a_key, b_key in [
        ("ref_latent_A", "ref_latent_B"),
        ("ref_mask_A", "ref_mask_B"),
        ("text_A_input_ids", "text_B_input_ids"),
        ("text_A_mask", "text_B_mask"),
    ]:
        batch[a_key], batch[b_key] = batch[b_key], batch[a_key]
    return batch


def _replace_batch_raster(batch: dict, phone_A, phone_B, activity_A, activity_B, boundary_radius: int) -> dict:
    batch["phone_A"] = phone_A.unsqueeze(0)
    batch["phone_B"] = phone_B.unsqueeze(0)
    batch["activity_A"] = activity_A.unsqueeze(0)
    batch["activity_B"] = activity_B.unsqueeze(0)
    for ch, ph in [("A", phone_A), ("B", phone_B)]:
        feats = build_soft_phone_features(ph, boundary_radius=boundary_radius)
        for k, v in feats.items():
            batch[f"{k}_{ch}"] = v.unsqueeze(0)
    return batch


def _load_dialog(phones_json: Path, dialog_id: str, combo_id: str) -> tuple[dict, dict]:
    raw = json.loads(phones_json.read_text(encoding="utf-8"))
    dialog = next((d for d in raw["dialogs"] if str(d.get("id")) == dialog_id), None)
    if dialog is None:
        raise ValueError(f"dialog not found: {dialog_id}")
    combo = next((c for c in raw.get("speaker_combos", []) if str(c.get("id")) == combo_id), None)
    if combo is None:
        raise ValueError(f"speaker combo not found: {combo_id}")
    return dialog, combo


def _load_placement_stats(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        print(f"[warn] placement stats not found; using old heuristic defaults: {p}", flush=True)
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    print(f"[load] placement_stats={p}", flush=True)
    return data


def _clip_float(x: float, lo: float, hi: float) -> float:
    if hi < lo:
        lo, hi = hi, lo
    return max(float(lo), min(float(hi), float(x)))


def _durations_for_total(n_items: int, total_frames: int) -> list[int]:
    if n_items <= 0 or total_frames <= 0:
        return []
    total_frames = max(total_frames, n_items)
    base = total_frames // n_items
    rem = total_frames % n_items
    return [base + (1 if i < rem else 0) for i in range(n_items)]


def _weighted_durations(phone_ids, total_frames: int, phone_means: dict | None, gmean: float) -> list[int]:
    """発話内フレームを各 phone の train 平均長に **比例配分**（合計=total を厳守・各 >=1）。
    phone_means=None なら等長（_durations_for_total）にフォールバック。 = 棒読み回避の自然な緩急。"""
    n = len(phone_ids)
    if n <= 0 or total_frames <= 0:
        return []
    total_frames = max(total_frames, n)
    if not phone_means:
        return _durations_for_total(n, total_frames)
    g = float(gmean) if gmean else 1.0
    w = [max(1e-6, float(phone_means.get(str(int(p)), g))) for p in phone_ids]
    sw = sum(w)
    raw = [total_frames * x / sw for x in w]
    d = [max(1, int(round(x))) for x in raw]
    diff = total_frames - sum(d)
    if diff > 0:                                   # 余りを「平均長が長い phone」へ
        order = sorted(range(n), key=lambda i: raw[i] - d[i], reverse=True)
        for k in range(diff):
            d[order[k % n]] += 1
    elif diff < 0:                                 # 超過を「1 を超える phone」から削る
        order = sorted(range(n), key=lambda i: d[i] - raw[i], reverse=True)
        rem, k = -diff, 0
        while rem > 0 and k < 100000:
            i = order[k % n]
            if d[i] > 1:
                d[i] -= 1
                rem -= 1
            k += 1
    return d


def _write_utterance(
    phone_by_ch: dict[str, torch.Tensor],
    act_by_ch: dict[str, torch.Tensor],
    placement: dict,
    phone_vocab_size: int,
    phone_means: dict | None = None,
    gmean: float = 1.0,
) -> None:
    ch = placement["speaker"]
    cursor = int(placement["start"])
    T = int(phone_by_ch[ch].shape[0])
    phone_ids = [int(x) for x in placement["phone_ids"]]
    # phone_means があれば train 平均長に比例配分（発話内の自然な緩急）、 無ければ等長
    durations = _weighted_durations(phone_ids, int(placement["dur"]), phone_means, gmean)
    for ph, dur in zip(phone_ids, durations):
        s = max(0, cursor)
        e = min(T, cursor + int(dur))
        if e > s:
            if ph <= 0 or ph >= phone_vocab_size:
                ph = SIL_PHONE_ID
            phone_by_ch[ch][s:e] = ph
            if ph != SIL_PHONE_ID:
                act_by_ch[ch][s:e] = 1.0
        cursor += int(dur)


def _scale_placements_to_fit(placements: list[dict], T: int, end_margin: int) -> list[dict]:
    if not placements:
        return placements
    min_start = min(int(p["start"]) for p in placements)
    max_end = max(int(p["start"]) + int(p["dur"]) for p in placements)
    target_end = max(min_start + 1, T - end_margin)
    if max_end <= target_end:
        return placements
    scale = (target_end - min_start) / max(1.0, float(max_end - min_start))
    out = []
    for p in placements:
        q = dict(p)
        q["start"] = int(round(min_start + (int(p["start"]) - min_start) * scale))
        q["dur"] = max(len(q["phone_ids"]), int(round(int(p["dur"]) * scale)))
        out.append(q)
    return out


def make_sequential_placements(
    utts: list[dict],
    T: int,
    *,
    start_offset: int,
    gap_frames: int,
    end_margin: int,
    frames_per_phone: float | None,
) -> list[dict]:
    n_phone_total = sum(max(0, int(u.get("n_phones", len(u.get("phone_ids", []))))) for u in utts)
    if frames_per_phone is None or frames_per_phone <= 0.0:
        gap_total = max(0, len(utts) - 1) * gap_frames
        speech_budget = max(n_phone_total, T - start_offset - end_margin - gap_total)
        frames_per_phone = speech_budget / max(1, n_phone_total)
    cursor = start_offset
    placements = []
    for u in utts:
        phone_ids = [int(x) for x in u.get("phone_ids", [])]
        dur = max(len(phone_ids), int(round(len(phone_ids) * frames_per_phone)))
        placements.append({
            "speaker": str(u["speaker"]),
            "text": str(u.get("text", "")),
            "phone_ids": phone_ids,
            "start": cursor,
            "dur": dur,
        })
        cursor += dur + gap_frames
    return _scale_placements_to_fit(placements, T, end_margin)


def _inv_cdf_frames(rng: random.Random, d: dict) -> float:
    """方向別 gap 経験分布の **区分線形 逆CDF** サンプル（frames）。
    knot = (p05,p10,median,p90,p95)。 u~Uniform(0.05,0.95) を引いて補間（= p5–p95 truncate）。
    手設定の jitter/cap を使わず、 spread も truncate も train 実分布そのもの。"""
    qs = (0.05, 0.10, 0.50, 0.90, 0.95)
    vs = [float(d["p05"]), float(d["p10"]), float(d["median"]), float(d["p90"]), float(d["p95"])]
    for i in range(1, len(vs)):           # percentile の単調非減少を保証（数値ノイズ対策）
        if vs[i] < vs[i - 1]:
            vs[i] = vs[i - 1]
    u = rng.uniform(qs[0], qs[-1])
    for i in range(len(qs) - 1):
        if u <= qs[i + 1]:
            t = (u - qs[i]) / (qs[i + 1] - qs[i])
            return vs[i] + t * (vs[i + 1] - vs[i])
    return vs[-1]


def train_gap_sec(rng: random.Random, prev_spk: str, spk: str, stats: dict | None, **_ignored) -> float:
    """発話間 gap（秒）。 方向別 train 経験分布から **逆CDFサンプリング（p5–p95 truncate）**。

    正規性を仮定せず、 spread・truncate とも train 実分布に従う（手設定の jitter/cap を全廃）。
    turn-switch は経験分布の p05 が負（被り）なので、 そこから自然に overlap が発生する。
    gap_by_transition（A_to_A/A_to_B/B_to_A/B_to_B、 frames@25fps、 p05/p10/median/p90/p95）を読む。
    無ければ旧 heuristic（中央値近傍）にフォールバック。 ※旧 jitter_sec/gap_cap/overlap_cap/gap_scale
    引数は廃止（**_ignored で後方互換）。"""
    gbt = (stats or {}).get("gap_by_transition", {})
    d = gbt.get(f"{prev_spk}_to_{spk}")
    if d is None or "median" not in d:
        same = (prev_spk == spk)
        center = 0.42 if same else 0.30
        return _clip_float(center + rng.gauss(0.0, 0.18), (0.0 if same else -OVERLAP_CAP_SEC), 1.1)
    # 被り（負 gap）は OVERLAP_CAP_SEC で頭打ち。 train の被り側 tail は同時発話(相づち被り)混在で
    # 中央値でも 0.76s と大きく、 そのまま使うと長発話同士が潰し合うため。 = 唯一の手設定（えいや 0.3s）。
    return max(_inv_cdf_frames(rng, d) / FPS, -OVERLAP_CAP_SEC)


def _transition_gap(
    rng: random.Random,
    prev_spk: str,
    spk: str,
    *,
    overlap_prob: float | None,
    stats: dict | None,
    gap_scale: float = 1.0,
) -> int:
    """発話間 gap（frames）。 train stats があれば median ベース train_gap_sec×FPS。"""
    if stats is not None:
        return int(round(train_gap_sec(rng, prev_spk, spk, stats, gap_scale=gap_scale) * FPS))
    if prev_spk == spk:
        return max(0, min(28, int(round(rng.gauss(10.0, 4.0)))))
    if rng.random() < float(0.22 if overlap_prob is None else overlap_prob):
        return -rng.randint(2, 9)
    return max(-3, min(26, int(round(rng.gauss(7.0, 7.0)))))


def make_statistical_placements(
    utts: list[dict],
    T: int,
    *,
    seed: int,
    start_offset: int,
    end_margin: int,
    overlap_prob: float,
    duration_scale: float,
    stats: dict | None,
) -> list[dict]:
    rng = random.Random(seed)
    placements = []
    channel_cursor = {"A": 0, "B": 0}
    prev_spk = None
    prev_end = start_offset
    sb = (stats or {}).get("statistical_baseline", {})
    fpp_summary = (stats or {}).get("frames_per_phone_by_threshold_bin", {})
    for i, u in enumerate(utts):
        spk = str(u["speaker"])
        phone_ids = [int(x) for x in u.get("phone_ids", [])]
        n = len(phone_ids)
        fpp_table = (stats or {}).get("fpp_by_nphones")
        if fpp_table:
            # 音素数 n 別の train 中央値 fpp（連続 lookup）。 bin 境界(8/26)も jitter も無し＝完全データ駆動。
            # fpp は短発話ほど大（n=1,2 が特に長く、 n>=3 は ~2.2-2.7）。 3-bin の粗さ（短bin一律3.0）による
            # 中間長発話の不当な遅さを解消する。
            fpp = float(fpp_table.get(str(n), (stats or {}).get("fpp_tail", 2.3)))
        elif stats is not None:
            # フォールバック: 旧 3-bin median（fpp_by_nphones 未生成時のみ。 jitter なし）
            bin_name = "short" if n <= 8 else ("long" if n >= 26 else "mid")
            fpp = float(fpp_summary.get(bin_name, {}).get("median", 2.45))
        else:
            fpp = 3.0 if n <= 8 else (2.2 if n >= 26 else 2.3)
        dur = max(n, int(round(n * fpp * duration_scale)))
        if i == 0:
            start = start_offset
        else:
            gap = _transition_gap(rng, str(prev_spk), spk, overlap_prob=overlap_prob, stats=stats)
            start = max(channel_cursor[spk], prev_end + gap)
        placements.append({
            "speaker": spk,
            "text": str(u.get("text", "")),
            "phone_ids": phone_ids,
            "start": int(start),
            "dur": int(dur),
        })
        end = int(start) + int(dur)
        channel_cursor[spk] = max(channel_cursor[spk], end)
        prev_spk = spk
        prev_end = end
    return _scale_placements_to_fit(placements, T, end_margin)


def placements_to_raster(placements: list[dict], T: int, phone_vocab_size: int,
                         phone_means: dict | None = None, gmean: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    phone_by_ch = {
        "A": torch.full((T,), SIL_PHONE_ID, dtype=torch.long),
        "B": torch.full((T,), SIL_PHONE_ID, dtype=torch.long),
    }
    act_by_ch = {
        "A": torch.zeros(T, dtype=torch.float32),
        "B": torch.zeros(T, dtype=torch.float32),
    }
    for p in placements:
        _write_utterance(phone_by_ch, act_by_ch, p, phone_vocab_size, phone_means, gmean)
    activity_A = act_by_ch["A"]
    activity_B = act_by_ch["B"]
    meta = {
        "active_A": float(activity_A.mean().item()),
        "active_B": float(activity_B.mean().item()),
        "overlap": float(((activity_A > 0.5) & (activity_B > 0.5)).float().mean().item()),
        "n_utts": len(placements),
        "max_end_frame": max((int(p["start"]) + int(p["dur"]) for p in placements), default=0),
        "placements": [
            {
                "speaker": p["speaker"],
                "text": p["text"],
                "start": int(p["start"]),
                "end": int(p["start"]) + int(p["dur"]),
                "dur": int(p["dur"]),
                "n_phones": len(p["phone_ids"]),
            }
            for p in placements
        ],
    }
    return phone_by_ch["A"], phone_by_ch["B"], activity_A, activity_B, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acoustic-config", required=True)
    ap.add_argument("--acoustic-ckpt", default=None, help="local .pt/.safetensors; omit to download from HF")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--phones-json", required=True)
    ap.add_argument("--dialog-id", default="demo_meeting_dialogic")
    ap.add_argument("--speaker-combo-id", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--modes", default="sequential,statistical")
    ap.add_argument("--cfg-scale", type=float, default=2.5)
    ap.add_argument("--num-steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--placement-seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq-gap-frames", type=int, default=6)
    ap.add_argument("--start-offset-frames", type=int, default=10)
    ap.add_argument("--end-margin-frames", type=int, default=30)
    ap.add_argument("--stat-overlap-prob", type=float, default=None,
                    help="Override turn-switch overlap probability. If omitted with placement stats, use train estimate.")
    ap.add_argument("--seq-frames-per-phone", type=float, default=0.0,
                    help="If positive, do not stretch sequential timing to the full chunk; use this average phone duration.")
    ap.add_argument("--stat-duration-scale", type=float, default=1.0)
    ap.add_argument("--placement-stats-json", default=DEFAULT_PLACEMENT_STATS_JSON,
                    help="Train-estimated placement stats JSON. If missing/empty, old hand-coded heuristic is used.")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = yaml.safe_load(Path(args.acoustic_config).read_text())
    dcfg = cfg["data"]
    pccfg = cfg.get("phone_condition", {})
    T = int(dcfg.get("latent_T", 750))
    boundary_radius = int(pccfg.get("boundary_radius", 5))
    phone_vocab_size = int(cfg["model"]["phone_vocab_size"])

    dialog, combo = _load_dialog(Path(args.phones_json), args.dialog_id, args.speaker_combo_id)
    placement_stats = _load_placement_stats(args.placement_stats_json)
    template_chunk = str(combo["template_chunk"])
    combo_A = str(combo["A"])
    combo_B = str(combo["B"])

    tok = PretrainedTextTokenizer.from_pretrained("llm-jp/llm-jp-3-150m", local_files_only=False)
    ds = Normal2StreamDataset(
        args.manifest,
        text_tokenizer=tok,
        max_text_len=int(dcfg.get("max_text_len", 64)),
        latent_T=T,
        speaker_balanced=False,
        soft_phone=(str(pccfg.get("mode", "hard")) == "soft"),
        soft_boundary_radius=boundary_radius,
    )
    idx = _find_index(ds, template_chunk)
    if idx is None:
        raise ValueError(f"template chunk not found in manifest: {template_chunk}")
    base_item = ds[idx]
    base_batch = collate_eventmix([base_item])
    row = ds.entries[idx]
    template_A = str(row.get("speaker_A") or row.get("spk_A") or "")
    template_B = str(row.get("speaker_B") or row.get("spk_B") or "")
    if combo_A == template_B and combo_B == template_A:
        base_batch = _swap_channel_refs(base_batch)
    elif combo_A != template_A or combo_B != template_B:
        print(f"[warn] combo {combo_A}/{combo_B} does not match template {template_A}/{template_B}; using template refs as-is", flush=True)

    acoustic = build_acoustic_model(cfg, device)
    from kaburi_tts.acoustic.infer import load_acoustic_checkpoint
    acoustic_step = load_acoustic_checkpoint(acoustic, args.acoustic_ckpt, device)
    pct8 = build_phone_class_table_8(dcfg["phone_vocab_path"]).to(device)
    codec = DACVAECodec.load(device=str(device), dtype=torch.bfloat16)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_meta = []
    for offset, mode in enumerate(modes):
        if mode == "sequential":
            placements = make_sequential_placements(
                dialog["utts"], T,
                start_offset=args.start_offset_frames,
                gap_frames=args.seq_gap_frames,
                end_margin=args.end_margin_frames,
                frames_per_phone=args.seq_frames_per_phone,
            )
        elif mode == "statistical":
            placements = make_statistical_placements(
                dialog["utts"], T,
                seed=args.placement_seed,
                start_offset=args.start_offset_frames,
                end_margin=args.end_margin_frames,
                overlap_prob=args.stat_overlap_prob,
                duration_scale=args.stat_duration_scale,
                stats=placement_stats,
            )
        else:
            raise ValueError(f"unknown mode: {mode}")

        phone_A, phone_B, activity_A, activity_B, meta = placements_to_raster(placements, T, phone_vocab_size)
        batch = _replace_batch_raster(dict(base_batch), phone_A, phone_B, activity_A, activity_B, boundary_radius)
        batch = _move_batch(batch, device)
        wav = synth_acoustic(
            acoustic, codec, batch, device=device, num_steps=args.num_steps,
            seed=args.seed + offset, cfg_scale=args.cfg_scale, pct8=pct8,
        )
        out = out_dir / (
            f"{args.dialog_id}__{args.speaker_combo_id}__ac{acoustic_step}_"
            f"{mode}_cfg{args.cfg_scale}_n{args.num_steps}.wav"
        )
        torchaudio.save(str(out), wav, codec.sample_rate, channels_first=True)
        row_meta = {
            "mode": mode,
            "wav": str(out),
            "dialog_id": args.dialog_id,
            "speaker_combo_id": args.speaker_combo_id,
            "phones_json": str(args.phones_json),
            "template_chunk": template_chunk,
            "speaker_A": combo_A,
            "speaker_B": combo_B,
            "acoustic_ckpt": str(args.acoustic_ckpt),
            "acoustic_step": acoustic_step,
            "cfg_scale": args.cfg_scale,
            "num_steps": args.num_steps,
            "seed": args.seed + offset,
            "placement_seed": args.placement_seed,
            "seq_gap_frames": args.seq_gap_frames,
            "stat_overlap_prob": args.stat_overlap_prob,
            "placement_stats_json": str(args.placement_stats_json) if args.placement_stats_json else None,
            "placement_stats_source": (placement_stats or {}).get("source"),
            "seq_frames_per_phone": args.seq_frames_per_phone,
            "stat_duration_scale": args.stat_duration_scale,
            **meta,
        }
        (out.with_suffix(".json")).write_text(json.dumps(row_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        all_meta.append(row_meta)
        print(f"[save] {out} overlap={meta['overlap']:.4f} active_A={meta['active_A']:.3f} active_B={meta['active_B']:.3f}", flush=True)

    (out_dir / "metadata.json").write_text(json.dumps(all_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()

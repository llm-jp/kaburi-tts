"""text dialog をまとめて合成（model 1 回ロード）。

--mode kaburi_pred  : predictor timing（text→phones→predictor raster→acoustic）
--mode kaburi_stat  : train 推定統計配置 raster（placement_stats_train.json）
--mode irodori_stat  : 生 Irodori 発話合成 + train 推定 gap 配置（2ch）。 短発話の間延びを抑制。

出力 <out_root>/<mode>/<dialog_id>.wav。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torchaudio
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kaburi_tts import ASSETS_DIR  # noqa: E402

FPS = 25.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["kaburi_pred", "kaburi_stat", "irodori_stat"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest", required=True, help="chunk manifest jsonl（template_chunk の参照元）")
    ap.add_argument("--acoustic_config", default=str(REPO_ROOT / "configs/acoustic.yaml"))
    ap.add_argument("--acoustic_ckpt", default=None, help="local .pt/.safetensors; omit to download from HF")
    ap.add_argument("--cfg_scale", type=float, default=2.5)
    ap.add_argument("--num_steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    # kaburi_stat
    ap.add_argument("--placement_seed", type=int, default=7)
    ap.add_argument("--stat_overlap_prob", type=float, default=None, help="None なら train stats の overlap 確率")
    ap.add_argument("--stat_duration_scale", type=float, default=1.0)  # = Eval A と統一（手調整除去）
    ap.add_argument("--start_offset_frames", type=int, default=10)
    ap.add_argument("--end_margin_frames", type=int, default=30)
    ap.add_argument("--placement_stats_json", default=str(ASSETS_DIR / "placement_stats_train.json"))
    # kaburi_pred
    ap.add_argument("--predictor_config", default=str(REPO_ROOT / "configs/predictor.yaml"))
    ap.add_argument("--predictor_ckpt", default=None, help="local .pt/.safetensors; omit to download from HF")
    ap.add_argument("--utt_timing_jsonls", nargs="+", default=None,
                    help="per-utt timing record jsonl 群 (= 既定は predictor config の data.utt_timing_jsonls。 ref pack 使用時は pack の utt_timing.jsonl を渡す)")
    ap.add_argument("--gap_blend", type=float, default=0.5)
    ap.add_argument("--gap_bias_frames", type=float, default=0.0)
    ap.add_argument("--fit_only", action="store_true",
                    help="タイミング配置 (fit) の計算だけ行い音響合成をスキップ"
                         " (台本が 30 秒ラスタに収まるかの事前検証用)")
    ap.add_argument("--phone_duration_scale", type=float, default=1.0)
    ap.add_argument("--dur_sample_temp", type=float, default=0.0,
                    help=">0 で音素継続長を予測分布からサンプル (0 = 期待値 = 従来)。"
                         "期待値は継続長ジッタを平均化で消すため、学習分布の分散を復元する用途")
    ap.add_argument("--presil_scale", type=float, default=1.0,
                    help="発話内ポーズ直前音素の継続長スケール (MFA 実測比の較正値 2.0)")
    ap.add_argument("--final_scale", type=float, default=1.0,
                    help="発話末音素の継続長スケール (MFA 実測比の較正値 1.45)")
    ap.add_argument("--dur_gamma", type=float, default=1.0,
                    help="予測偏差のクラス条件付き増幅率 (1.0 = 従来)。乱数なしで nPVI を復元")
    ap.add_argument("--dur_lm", default=None,
                    help="DurationLM ckpt。指定時は音素継続長を MFA 実測分布模倣の AR サンプルに置換")
    ap.add_argument("--dur_lm_temp", type=float, default=1.0)
    ap.add_argument("--inserted_sil_scale", type=float, default=1.0,
                    help="発話内 <sil> トークンの継続長スケール (「区切りました感」の緩和用)")
    ap.add_argument("--duration_calibration", default=None,
                    help="posterior calibration 資産 JSON。指定時は境界文脈の exponential tilt で"
                         "継続長をデコード (検証完了まで既定 OFF)")
    ap.add_argument("--duration_calibration_total", choices=["none", "raw", "lenfix", "lenfix_up"],
                    default="none", help="発話 total の projection モード")
    # irodori_stat
    ap.add_argument("--iro_config", default=str(REPO_ROOT / "configs/irodori_baseline.yaml"))
    ap.add_argument("--iro_refs_dir", default=None,
                    help="irodori_stat: 話者参照 wav のディレクトリ（<speaker_id>.wav）")
    ap.add_argument("--iro_tail_sec", type=float, default=0.4)
    ap.add_argument("--iro_fit_target_sec", type=float, default=40.0,
                    help="irodori_stat: 末尾がこの秒を超えたら発話間 gap(正のslack)だけを一様圧縮して収める（発話長・overlap は不変）")
    ap.add_argument("--force_iro_utts", action="store_true",
                    help="既存の iro_utts cache を無視して発話を再合成（= budget 変更時）")
    ap.add_argument("--out_root", required=True, help="出力先ルート")
    ap.add_argument("--phones_json", required=True, help="g2p 済 phones json")
    ap.add_argument("--speakers_json", required=True, help="dialog_id -> {A,B,template_chunk} の json")
    ap.add_argument("--meta_jsonl", default=None,
                    help="write per-dialog placement/timing metadata as JSONL")
    args = ap.parse_args()

    raw = json.loads(Path(args.phones_json).read_text())
    dialogs = raw["dialogs"]
    speakers = json.loads(Path(args.speakers_json).read_text())  # rpc_id -> {A,B,template_chunk}
    if args.limit > 0:
        dialogs = dialogs[: args.limit]
    out_dir = Path(args.out_root) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{args.mode}] dialogs={len(dialogs)} -> {out_dir}", flush=True)

    if args.mode == "irodori_stat":
        _run_iro(args, dialogs, speakers, out_dir)
    else:
        _run_kaburi(args, dialogs, speakers, torch.device(args.device), out_dir)


# gap は SPB.train_gap_sec（train 分布の median 中心・現実的 cap）に一本化 → _run_iro 内で呼ぶ


def _run_kaburi(args, dialogs, speakers, device, out_dir):
    from irodori_tts.tokenizer import PretrainedTextTokenizer
    from kaburi_tts.two_stream.dataset import Normal2StreamDataset, collate_eventmix
    from kaburi_tts.two_stream.loss import build_phone_class_table_8
    import kaburi_tts.placement.baselines as SPB
    import synth_with_predictor as SPRED

    cfg = yaml.safe_load(Path(args.acoustic_config).read_text())
    dcfg = cfg["data"]; pccfg = cfg.get("phone_condition", {})
    T = int(dcfg.get("latent_T", 750)); boundary_radius = int(pccfg.get("boundary_radius", 5))
    phone_vocab_size = int(cfg["model"]["phone_vocab_size"])

    tok = PretrainedTextTokenizer.from_pretrained("llm-jp/llm-jp-3-150m", local_files_only=False)
    acoustic_ds = Normal2StreamDataset(
        args.manifest, text_tokenizer=tok, max_text_len=int(dcfg.get("max_text_len", 64)),
        latent_T=T, speaker_balanced=False,
        soft_phone=(str(pccfg.get("mode", "hard")) == "soft"), soft_boundary_radius=boundary_radius)

    acoustic = codec = pct8 = synth_acoustic = None
    if not args.fit_only:
        from irodori_tts.codec import DACVAECodec
        from kaburi_tts.acoustic.infer import (
            build_model as build_acoustic_model, load_acoustic_checkpoint, synth as synth_acoustic,
        )

        acoustic = build_acoustic_model(cfg, device)
        load_acoustic_checkpoint(acoustic, args.acoustic_ckpt, device)
        pct8 = build_phone_class_table_8(dcfg["phone_vocab_path"]).to(device)
        codec = DACVAECodec.load(device=str(device), dtype=torch.bfloat16)

    placement_stats = None
    predictor = pred_ds = None
    if args.mode == "kaburi_stat":
        placement_stats = SPB._load_placement_stats(args.placement_stats_json)
        print(f"[stat] placement_stats loaded: {placement_stats is not None}", flush=True)
    if args.mode == "kaburi_pred":
        pred_cfg = yaml.safe_load(Path(args.predictor_config).read_text())
        utj = args.utt_timing_jsonls or list(pred_cfg["data"]["utt_timing_jsonls"])
        pred_ds = SPRED.TimingPredictorDataset(
            args.manifest, utj, text_tokenizer=tok, max_text_len=64,
            latent_T=T, soft_radius=boundary_radius,
            max_collapsed_seq_len=int(pred_cfg["context"]["max_collapsed_seq_len"]),
            phone_class_lookup=build_phone_class_table_8(pred_cfg["data"]["phone_vocab_path"]),
            T_frames=T, max_timing_seq_len=int(pred_cfg["timing"]["max_timing_seq_len"]),
            max_utt_text_len=int(pred_cfg["model"].get("max_utt_text_len", 64)),
            utt_manifest_path=pred_cfg["data"].get("utt_manifest_path"))
        predictor, pstep = SPRED._load_predictor(pred_cfg, args.predictor_ckpt, device)
        dur_lm = None
        if args.dur_lm:
            from kaburi_tts.predictor.duration_lm import load_duration_lm
            dur_lm = load_duration_lm(args.dur_lm, device=str(device))
            print(f"[durlm] loaded {args.dur_lm}", flush=True)
        dur_calib = None
        if args.duration_calibration:
            from kaburi_tts.predictor.duration_calibration import file_sha1, load_calibration
            pred_id = f"{Path(args.predictor_ckpt).name}:step{pstep}" if args.predictor_ckpt else None
            dur_calib = load_calibration(
                args.duration_calibration, predictor_id=pred_id,
                phone_vocab_hash=file_sha1(REPO_ROOT / "assets/phone_vocab.json"))
            print(f"[calib] loaded {args.duration_calibration} "
                  f"(l2={dur_calib.get('l2_weight')}, total={args.duration_calibration_total})", flush=True)
        print(f"[load] predictor step={pstep}", flush=True)

    meta_f = None
    if args.meta_jsonl:
        meta_path = Path(args.meta_jsonl)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_f = meta_path.open("w", encoding="utf-8")

    n_ok = n_miss = 0
    for off, d in enumerate(dialogs):
        did = d["id"]
        out_wav = out_dir / f"{did}.wav"
        if out_wav.exists():
            n_ok += 1; continue
        sp = speakers.get(did)
        if sp is None:
            print(f"  [miss] {did}: speaker 割当なし", flush=True); n_miss += 1; continue
        cA, cB, template_chunk = str(sp["A"]), str(sp["B"]), str(sp["template_chunk"])
        aidx = SPB._find_index(acoustic_ds, template_chunk)
        if aidx is None:
            print(f"  [miss] {did}: template {template_chunk} not in manifest", flush=True); n_miss += 1; continue
        row = acoustic_ds.entries[aidx]
        tA = str(row.get("speaker_A") or row.get("spk_A") or ""); tB = str(row.get("speaker_B") or row.get("spk_B") or "")
        base_batch = collate_eventmix([acoustic_ds[aidx]])
        if cA == tB and cB == tA:
            base_batch = SPB._swap_channel_refs(base_batch)

        if args.mode == "kaburi_stat":
            placements = SPB.make_statistical_placements(
                d["utts"], T, seed=args.placement_seed, start_offset=args.start_offset_frames,
                end_margin=args.end_margin_frames, overlap_prob=args.stat_overlap_prob,
                duration_scale=args.stat_duration_scale, stats=placement_stats)
            pA, pB, aA, aB, timing_meta = SPB.placements_to_raster(
                placements, T, phone_vocab_size,
                phone_means=(placement_stats or {}).get("phone_mean_frames"),
                gmean=float((placement_stats or {}).get("phone_mean_frames_global", 1.0)))
            batch = SPB._replace_batch_raster(dict(base_batch), pA, pB, aA, aB, boundary_radius)
        else:  # kaburi_pred
            pidx = SPRED._find_predictor_index(pred_ds, template_chunk)
            template_rec = pred_ds.utt_timing[template_chunk]
            new_rec = SPRED._build_modified_record(template_rec, d["utts"], cA, cB)
            pred_item = pred_ds._attach_timing(pred_ds[pidx], new_rec)
            pA, pB, aA, aB, timing_meta = SPRED._predict_raster_from_item(
                predictor, pred_item, device, T=T, gap_blend=args.gap_blend,
                gap_bias_frames=args.gap_bias_frames, phone_duration_scale=args.phone_duration_scale,
                phone_vocab_size=phone_vocab_size,
                dur_sample_temp=args.dur_sample_temp, dur_seed=args.seed + off,
                presil_scale=args.presil_scale, final_scale=args.final_scale,
                dur_gamma=args.dur_gamma,
                class_table=(pct8.cpu() if pct8 is not None else
                             build_phone_class_table_8(dcfg["phone_vocab_path"])),
                dur_lm=dur_lm, dur_lm_temp=args.dur_lm_temp,
                inserted_sil_scale=args.inserted_sil_scale,
                duration_calibration=dur_calib,
                duration_calibration_total=args.duration_calibration_total)
            batch = SPRED._replace_batch_raster(dict(base_batch), pA, pB, aA, aB, boundary_radius)
        if meta_f is not None:
            meta_row = {
                "dialog_id": did,
                "mode": args.mode,
                "speaker_A": cA,
                "speaker_B": cB,
                "template_chunk": template_chunk,
                "seed": int(args.seed + off),
                "gap_blend": float(args.gap_blend),
                "gap_bias_frames": float(args.gap_bias_frames),
                "phone_duration_scale": float(args.phone_duration_scale),
                "n_utts": int(len(d["utts"])),
                "n_phones": int(sum(int(u.get("n_phones", len(u.get("phone_ids", [])))) for u in d["utts"])),
                **(timing_meta or {}),
            }
            meta_f.write(json.dumps(meta_row, ensure_ascii=False) + "\n")
            meta_f.flush()
        if args.fit_only:
            print(f"[fitonly] {did} placed", flush=True)
            n_ok += 1
            continue
        assert acoustic is not None and codec is not None and pct8 is not None and synth_acoustic is not None
        batch = SPB._move_batch(batch, device)
        wav = synth_acoustic(acoustic, codec, batch, device=device, num_steps=args.num_steps,
                        seed=args.seed + off, cfg_scale=args.cfg_scale, pct8=pct8)
        wav = _tail_trim_activity(wav.cpu(), aA, aB, codec.sample_rate)  # 末尾 garbage 除去
        torchaudio.save(str(out_wav), wav, codec.sample_rate, channels_first=True)
        n_ok += 1
        print(f"  [{args.mode}] {did} ({cA}x{cB}) done", flush=True)
    print(f"[{args.mode}] done ok={n_ok} miss={n_miss} -> {out_dir}", flush=True)
    if meta_f is not None:
        meta_f.close()


# text-input は GT 長が無いので、 phone 数から発話長を推定（train fpp median ~2.5 frames/phone ÷25fps）。
# = 一般規則（手調整なし）。 SEC_PER_PHONE を介して iro_gt baseline の seconds 式に代入する。
SEC_PER_PHONE = 0.10


def _iro_budget_sec(n_phones: int, sc: dict) -> float:
    # Eval A (synth_chunk_utts) と **完全同一式**: clamp(D*margin + pad, seconds_min, seconds_max)。
    # 共有 baseline.yaml の seconds_margin/pad/min/max を参照。 D だけが A/B で異なる
    # （A=GT長 gt_dur / B=phone推定 est_dur=SEC_PER_PHONE*n、 B には GT が無いため）。
    # ※これは over-generation budget（最終発話長ではない）。 生成後 _trim_energy で前後無音を除去し
    #   trim 後長で配置する。 pad は短発話を延ばさない安全余白（短発話の間延びを防ぐ）。
    est_dur = SEC_PER_PHONE * float(max(1, n_phones))
    return max(float(sc.get("seconds_min", 1.2)),
               min(float(sc.get("seconds_max", 8.0)),
                   est_dur * float(sc.get("seconds_margin", 1.3)) + float(sc.get("seconds_pad", 0.8))))


def _trim_energy(wav, sr, thr_rel=0.05, win_sec=0.02, pad=0.05):
    """単純なエネルギー閾値による endpoint trimming。
    20ms 包絡でピーク振幅の 5% を閾値とし、 **最初に閾値を超える窓〜最後に超える窓**＋小 pad を残す
    （= 先頭・末尾の無音を除去するだけ。 間の無音や発話内の間はそのまま保持）。
    セグメント連結・junk 判定などの heuristic は行わない（論文記述可能な単一閾値の標準処理）。
    閾値超えの末尾 junk がまれに残る/薄い語尾がまれに切れる場合があるが許容する。"""
    import numpy as np
    x = wav.abs().mean(0).cpu().numpy()
    w = max(1, int(win_sec * sr))
    n = len(x) // w
    if n == 0:
        return wav
    env = np.array([x[i * w:(i + 1) * w].max() for i in range(n)])
    peak = float(env.max())
    if peak <= 0:
        return wav
    idx = np.where(env > thr_rel * peak)[0]
    if len(idx) == 0:
        return wav
    s = max(0, int((int(idx[0]) * win_sec - pad) * sr))
    e = min(wav.shape[1], int(((int(idx[-1]) + 1) * win_sec + pad) * sr))
    return wav[:, s:e] if e > s else wav


def _tail_trim_activity(wav, aA, aB, sr, fps=25.0, margin_frames=8):
    """acoustic 出力の末尾 garbage 除去: activity ラスタの最終活性フレーム+margin で切る。
    内容 timeline が 30s 固定 canvas より早く終わる場合、 残りの無活動域に生じる garbage を消す。"""
    act = (aA.reshape(-1) > 0) | (aB.reshape(-1) > 0)
    nz = torch.nonzero(act)
    if len(nz) == 0:
        return wav
    last = int(nz[-1].item())
    cut = int((last + 1 + margin_frames) * sr / fps)
    return wav[:, :cut] if 0 < cut < wav.shape[1] else wav


def _run_iro(args, dialogs, speakers, out_dir):
    import numpy as np
    import kaburi_tts.placement.irodori as IRO
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    from irodori_tts.text_normalization import normalize_text
    from huggingface_hub import hf_hub_download

    if args.iro_refs_dir is None:
        raise SystemExit("--iro_refs_dir (speaker reference wav directory) is required for --mode irodori_stat")
    refs_dir = Path(args.iro_refs_dir)

    cfg = yaml.safe_load(Path(args.iro_config).read_text())
    sc = cfg["synth"]
    ckpt = hf_hub_download(repo_id=str(cfg["paths"]["base_checkpoint_hf"]).strip(), filename="model.safetensors")
    runtime = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=ckpt, model_device=args.device, codec_repo=str(cfg["paths"]["codec_repo"]),
        model_precision=str(sc.get("model_precision", "fp32")), codec_device=args.device,
        codec_precision=str(sc.get("codec_precision", "fp32"))))
    import kaburi_tts.placement.baselines as SPB
    sr = int(runtime.codec.sample_rate)
    stats = json.loads(Path(args.placement_stats_json).read_text())
    print(f"[iro] train gap stats: gap_by_transition={'gap_by_transition' in stats}", flush=True)
    cache_root = Path(args.out_root) / "iro_utts"  # = out_root 追従（試聴用 別root で別 cache）

    n_ok = n_miss = 0
    for off, d in enumerate(dialogs):
        did = d["id"]
        out_wav = out_dir / f"{did}.wav"
        if out_wav.exists() and not args.force_iro_utts:
            n_ok += 1; continue
        sp = speakers.get(did)
        if sp is None:
            print(f"  [miss] {did}: speaker なし", flush=True); n_miss += 1; continue
        ref_by = {"A": str(refs_dir / f"{sp['A']}.wav"), "B": str(refs_dir / f"{sp['B']}.wav")}
        spk_id = {"A": sp["A"], "B": sp["B"]}
        cdir = cache_root / did; cdir.mkdir(parents=True, exist_ok=True)
        utts = []
        for i, u in enumerate(d["utts"]):
            ch = str(u["speaker"]); text = normalize_text(str(u["text"])).strip()
            n_ph = int(u.get("n_phones", len(u.get("phone_ids", []))))
            wp = cdir / f"{i:02d}_{spk_id[ch]}.wav"   # = 実話者 id を含めて cache（話者変更で再合成）
            if (not wp.exists()) or args.force_iro_utts:
                # over-generation budget（est ベース）＋ trim_tail=True ＋ _trim_energy（前後無音除去）。
                # 最終発話長は trim 後 wav で決まる（pad は短発話を延ばさない安全余白）。
                budget = _iro_budget_sec(n_ph, sc)
                res = runtime.synthesize(SamplingRequest(
                    text=text, ref_wav=ref_by[ch], seconds=budget,
                    num_steps=int(sc.get("num_steps", 40)),
                    cfg_scale_text=float(sc.get("cfg_scale_text", 3.0)),
                    cfg_scale_speaker=float(sc.get("cfg_scale_speaker", 5.0)),
                    seed=args.seed + i, trim_tail=True))
                clip = _trim_energy(res.audio.cpu(), sr)
                torchaudio.save(str(wp), clip, sr, channels_first=True)
                if args.limit > 0:  # = smoke 時のみ budget/trim 後長を log
                    print(f"[iro-utt] {did} idx={i} text={text[:10]} n={n_ph} "
                          f"budget={budget:.2f}s trimmed={clip.shape[-1]/sr:.2f}s", flush=True)
            w, wsr = torchaudio.load(str(wp))
            if wsr != sr:
                w = torchaudio.functional.resample(w, wsr, sr)
            x = w.mean(0).numpy().astype(np.float32)
            utts.append({"index": i, "speaker": ch, "text": text, "n_phones": n_ph,
                         "wav_path": str(wp), "audio": x, "duration_sec": float(x.shape[0] / sr)})
        # train 推定 gap で配置（IRO._place で 2ch レンダリング）
        # gap は rng で一度だけ引いて固定 → gap-only fit で gap_scale を変えて再配置できるように。
        rng = random.Random(args.placement_seed)
        gaps = [0.0] + [SPB.train_gap_sec(rng, str(utts[i-1]["speaker"]), utts[i]["speaker"], stats)
                        for i in range(1, len(utts))]  # median 中心・現実的 cap（秒）。 負=turn-switch overlap

        def _place_with(gap_scale):
            # 正の gap（無音 slack）だけ gap_scale 倍に圧縮。 負の gap（overlap）と発話長は不変。
            ch_cursor = {"A": 0.0, "B": 0.0}; prev_end = 0.35; pls = []
            for i, u in enumerate(utts):
                ch = u["speaker"]
                if i == 0:
                    start = 0.35
                else:
                    g = gaps[i] * gap_scale if gaps[i] > 0 else gaps[i]
                    start = max(ch_cursor[ch], prev_end + g)
                pls.append({**u, "start_sec": start})
                end = start + u["duration_sec"]; ch_cursor[ch] = max(ch_cursor[ch], end); prev_end = end
            return pls

        placements = _place_with(1.0)
        raw_end = max(p["start_sec"] + p["duration_sec"] for p in placements)
        target = float(args.iro_fit_target_sec)
        if raw_end > target:  # 末尾が target 超 → gap(正のslack)だけ一様圧縮して fit（発話長・話速・overlap は不変）
            lo, hi = 0.0, 1.0
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                e = max(p["start_sec"] + p["duration_sec"] for p in _place_with(mid))
                if e > target:
                    hi = mid
                else:
                    lo = mid
            placements = _place_with(lo)
            fit_end = max(p["start_sec"] + p["duration_sec"] for p in placements)
            print(f"  [iro-fit] {did} raw={raw_end:.1f}s -> {fit_end:.1f}s (gap_scale={lo:.3f}, target={target:.0f})", flush=True)
        IRO._place(placements, sr, out_wav, tail_sec=args.iro_tail_sec)
        n_ok += 1
        print(f"  [irodori_stat] {did} ({sp['A']}x{sp['B']}) done", flush=True)
    print(f"[irodori_stat] done ok={n_ok} miss={n_miss} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

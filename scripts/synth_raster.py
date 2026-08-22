"""音素ラスタ生成経路の合成ランナー: 対話テキスト spec -> 2ch wav。

kaburi_cli.py (pred モード既定) から呼ばれる。タイミング生成は
kaburi_tts.raster (realizer: 実現形+duration / gap model: 間・かぶり) が行い、
音響合成は従来と同一 (acoustic/codec/参照は無変更)。

spec 形式は kaburi_cli が作る authored spec と同じ:
  {"dialogs": [{"id": ..., "utts": [{"speaker": "A", "text": ...}, ...]}]}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="authored spec json (対話テキスト)")
    ap.add_argument("--manifest", required=True, help="ref pack / chunk manifest jsonl")
    ap.add_argument("--speakers_json", required=True,
                    help="dialog_id -> {A,B,template_chunk}")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg_scale", type=float, default=2.5)
    ap.add_argument("--num_steps", type=int, default=32)
    ap.add_argument("--acoustic_ckpt", default=None)
    ap.add_argument("--realizer_ckpt", default=None)
    ap.add_argument("--gap_ckpt", default=None)
    ap.add_argument("--raster-release", dest="raster_release", default=None,
                    help="使用する raster release (既定 = 最新)")
    ap.add_argument("--no-activity-gate", dest="no_activity_gate", action="store_true",
                    help="無発話区間の activity gate を無効化 (debug 用。公開既定は ON)")
    ap.add_argument("--decode_json", default=None)
    ap.add_argument("--acoustic_config", default=str(REPO / "configs/acoustic.yaml"))
    ap.add_argument("--timing_meta_json", default=None,
                    help="配置メタ (overlap/fit) の出力先 (任意)")
    args = ap.parse_args()

    import torch
    import torchaudio
    import yaml
    from irodori_tts.codec import DACVAECodec
    from irodori_tts.tokenizer import PretrainedTextTokenizer
    from kaburi_tts.acoustic.infer import build_model, load_acoustic_checkpoint, synth
    from kaburi_tts.two_stream.dataset import Normal2StreamDataset, collate_eventmix
    from kaburi_tts.two_stream.loss import build_phone_class_table_8
    import kaburi_tts.placement.baselines as SPB
    from kaburi_tts.raster import RasterGenerator

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    speakers = json.loads(Path(args.speakers_json).read_text(encoding="utf-8"))
    out_dir = Path(args.out_root) / "raster"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    config = yaml.safe_load(Path(args.acoustic_config).read_text())
    data_config = config["data"]
    boundary_radius = int(config.get("phone_condition", {}).get("boundary_radius", 5))
    tokenizer = PretrainedTextTokenizer.from_pretrained(
        "llm-jp/llm-jp-3-150m", local_files_only=False)
    dataset = Normal2StreamDataset(
        args.manifest, text_tokenizer=tokenizer,
        max_text_len=int(data_config.get("max_text_len", 64)),
        latent_T=750, speaker_balanced=False,
        soft_phone=(str(config.get("phone_condition", {}).get("mode", "hard")) == "soft"),
        soft_boundary_radius=boundary_radius)
    acoustic = build_model(config, device)
    load_acoustic_checkpoint(acoustic, args.acoustic_ckpt, device)
    pct8 = build_phone_class_table_8(data_config["phone_vocab_path"]).to(device)
    codec = DACVAECodec.load(device=str(device), dtype=torch.bfloat16)

    # タイミングモデルは CPU で十分軽い (合成の主コストは acoustic)
    timing = RasterGenerator(device="cpu", realizer_ckpt=args.realizer_ckpt,
                             gap_ckpt=args.gap_ckpt, decode_json=args.decode_json,
                             release=args.raster_release)
    gate_cfg = dict(timing.decode_cfg.get("activity_gate", {}))
    gate_on = bool(gate_cfg.get("enabled", True)) and not args.no_activity_gate
    print(f"[raster] release={timing.release} realizer={type(timing.realizer).__name__} "
          f"activity_gate={'on' if gate_on else 'off'}", flush=True)

    metas = []
    for offset, dialog in enumerate(spec["dialogs"]):
        did = str(dialog["id"])
        sp = speakers.get(did)
        if sp is None:
            print(f"[miss] {did}: speaker 割当なし", flush=True)
            continue
        aidx = SPB._find_index(dataset, str(sp["template_chunk"]))
        if aidx is None:
            print(f"[miss] {did}: template {sp['template_chunk']} not in manifest", flush=True)
            continue
        placed, meta = timing.timeline(dialog["utts"], chunk_id=did)
        phone_a, phone_b, act_a, act_b = RasterGenerator.rasterize(placed)
        batch = collate_eventmix([dataset[aidx]])
        batch["latent_mask"] = torch.ones_like(batch["latent_mask"], dtype=torch.bool)
        row = dataset.entries[aidx]
        t_a = str(row.get("speaker_A") or row.get("spk_A") or "")
        t_b = str(row.get("speaker_B") or row.get("spk_B") or "")
        if str(sp["A"]) == t_b and str(sp["B"]) == t_a:
            batch = SPB._swap_channel_refs(batch)
        batch = SPB._replace_batch_raster(dict(batch), phone_a, phone_b, act_a, act_b,
                                          boundary_radius)
        batch = SPB._move_batch(batch, device)
        wav = synth(acoustic, codec, batch, device=device, num_steps=args.num_steps,
                    seed=args.seed + offset, cfg_scale=args.cfg_scale, pct8=pct8).cpu()
        act = torch.maximum(act_a, act_b)
        nz = torch.nonzero(act > 0)
        if len(nz):
            cut = int((int(nz[-1]) + 1 + 8) * codec.sample_rate / 25.0)
            if 0 < cut < wav.shape[-1]:
                wav = wav[:, :cut]
        if gate_on:   # 無発話区間の acoustic 漏れ抑制 (active 区間は bit 不変)
            from kaburi_tts.raster.activity_gate import gate_channels_first
            wav, _env = gate_channels_first(
                wav, torch.stack([act_a, act_b]), codec.sample_rate,
                fps=int(gate_cfg.get("fps", 25)),
                pad_frames=int(gate_cfg.get("pad_frames", 2)),
                fade_ms=float(gate_cfg.get("fade_ms", 40.0)))
        out_wav = out_dir / f"{did}.wav"
        torchaudio.save(str(out_wav), wav, codec.sample_rate, channels_first=True)
        overlap = float(((act_a > 0.5) & (act_b > 0.5)).float().mean())
        metas.append({"dialog_id": did, **meta, "overlap": overlap,
                      "raster_release": timing.release,
                      "activity_gate": (gate_cfg if gate_on else {"enabled": False})})
        print(f"[raster] {did} utts={meta['n_utts']} overlap={overlap:.3f} "
              f"fit={meta['fit_scale']:.2f} done", flush=True)

    if args.timing_meta_json:
        Path(args.timing_meta_json).write_text(
            json.dumps(metas, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[raster] done ok={len(metas)} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

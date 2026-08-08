"""Synthesize KABURI-TTS audio using predictor timing.

This bridges the timing predictor and the acoustic model:

  manifest chunk -> predictor tokens -> predicted phone/activity raster
                 -> frozen acoustic inference -> wav

It can also emit a GT-raster wav for the same chunk for quick listening
comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from irodori_tts.codec import DACVAECodec
from irodori_tts.tokenizer import PretrainedTextTokenizer
from kaburi_tts.two_stream.dataset import Normal2StreamDataset, collate_eventmix
from kaburi_tts.two_stream.loss import build_phone_class_table_8, build_soft_phone_features
from kaburi_tts.predictor import (
    PHONE_BIN_VALUES, SIL_BIN_VALUES, GAP_BIN_VALUES,
    TOKEN_TYPE_PHONE, TOKEN_TYPE_PRE_SIL, TOKEN_TYPE_FIRST_SIL,
)
from kaburi_tts.predictor.dataset import TimingPredictorDataset, collate_timing
from kaburi_tts.predictor.model import KaburiTimingPredictor
from kaburi_tts.acoustic.infer import (
    build_model as build_acoustic_model, load_acoustic_checkpoint, synth as synth_acoustic,
)


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def _decode_durations(phone_logits, sil_logits, token_type, *, phone_mode="expected", sil_mode="argmax",
                      sample_temp: float = 1.0, generator=None):
    device = phone_logits.device
    phone_values = torch.tensor(PHONE_BIN_VALUES, device=device, dtype=torch.float32)
    sil_values = torch.tensor(SIL_BIN_VALUES, device=device, dtype=torch.float32)
    phone_probs = torch.softmax(phone_logits.float(), dim=-1)
    sil_probs = torch.softmax(sil_logits.float(), dim=-1)
    if phone_mode == "argmax":
        phone_dur = phone_values[phone_probs.argmax(dim=-1)]
    elif phone_mode == "sample":
        # 予測分布からのサンプリング。expected (期待値) は継続長のジッタを平均化で
        # 消してしまう — 学習済み分布の分散をそのまま使うことで揺らぎを復元する。
        p = torch.softmax(phone_logits.float() / max(sample_temp, 1e-3), dim=-1)
        idx = torch.multinomial(p, 1, generator=generator)[:, 0]
        phone_dur = phone_values[idx]
    else:
        phone_dur = (phone_probs * phone_values).sum(dim=-1)
    if sil_mode == "expected":
        sil_dur = (sil_probs * sil_values).sum(dim=-1)
    else:
        sil_dur = sil_values[sil_probs.argmax(dim=-1)]
    out = torch.zeros(token_type.shape[0], device=device, dtype=torch.float32)
    out = torch.where(token_type == TOKEN_TYPE_PHONE, phone_dur, out)
    out = torch.where(
        (token_type == TOKEN_TYPE_PRE_SIL) | (token_type == TOKEN_TYPE_FIRST_SIL),
        sil_dur, out,
    )
    return out


def _decode_any_gaps(any_gap_logits, is_utt_first_token, *, mode="expected"):
    device = any_gap_logits.device
    gap_values = torch.tensor(GAP_BIN_VALUES, device=device, dtype=torch.float32)
    probs = torch.softmax(any_gap_logits.float(), dim=-1)
    if mode == "argmax":
        gap = gap_values[probs.argmax(dim=-1)]
    else:
        gap = (probs * gap_values).sum(dim=-1)
    return torch.where(is_utt_first_token.bool(), gap, torch.zeros_like(gap))


def _channel_total_len(durations, token_type, channel_id, attn_mask, target_ch: int) -> float:
    mask = ((channel_id == target_ch) & attn_mask).float()
    return float((durations * mask).sum().item())


def _fit_durations_to_T(durations, token_type, channel_id, attn_mask, T: int):
    fitted = durations.clone()
    info = {}
    for ch in [0, 1]:
        overflow_before = max(0.0, _channel_total_len(fitted, token_type, channel_id, attn_mask, ch) - T)
        reduced = 0.0
        sil_mask = (
            ((token_type == TOKEN_TYPE_PRE_SIL) | (token_type == TOKEN_TYPE_FIRST_SIL))
            & (channel_id == ch) & attn_mask
        )
        sil_indices = sil_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        sil_indices.sort(key=lambda i: -float(fitted[i].item()))
        overflow = overflow_before
        for idx in sil_indices:
            if overflow <= 0:
                break
            cur = float(fitted[idx].item())
            take = min(cur, overflow)
            fitted[idx] = cur - take
            overflow -= take
            reduced += take
        remaining = max(0.0, _channel_total_len(fitted, token_type, channel_id, attn_mask, ch) - T)
        clamped = 0.0
        if remaining > 0:
            ch_indices = ((channel_id == ch) & attn_mask).nonzero(as_tuple=False).squeeze(-1).tolist()
            for idx in reversed(ch_indices):
                if remaining <= 0:
                    break
                cur = float(fitted[idx].item())
                take = min(cur, remaining)
                fitted[idx] = cur - take
                remaining -= take
                clamped += take
        info[f"ch{ch}_overflow_before"] = float(overflow_before)
        info[f"ch{ch}_sil_reduced"] = float(reduced)
        info[f"ch{ch}_clamped"] = float(clamped)
    return fitted, info


def _reconstruct_with_any_gap(
    durations, gaps, token_type, channel_id, utt_index, attn_mask, global_utt_order, T: int, *,
    gap_blend: float, gap_scale: float = 1.0,
):
    # gap_scale<1 は **発話間の正の gap（pause=同一channel の無音 ideal_start−cursor）のみ**を縮める。
    # phone duration は不変。 cross-channel overlap（負の any-gap）は ideal_start の計算に内包され保持。
    N = token_type.shape[0]
    device = durations.device
    starts = torch.zeros(N, dtype=torch.float32, device=device)
    ends = torch.zeros(N, dtype=torch.float32, device=device)
    if global_utt_order:
        first_positions = [
            int(k) for k in global_utt_order
            if 0 <= int(k) < N and bool(attn_mask[int(k)].item())
        ]
    else:
        first_positions = [
            k for k in range(N)
            if bool(attn_mask[k].item())
            and int(token_type[k].item()) in (TOKEN_TYPE_PRE_SIL, TOKEN_TYPE_FIRST_SIL)
        ]
    channel_cursor = {0: 0.0, 1: 0.0}
    last_any_end = 0.0
    placed_phone = torch.zeros(N, dtype=torch.bool, device=device)
    for first_pos in first_positions:
        ch = int(channel_id[first_pos].item())
        u = int(utt_index[first_pos].item())
        same_start = channel_cursor[ch] + float(durations[first_pos].item())
        any_start = last_any_end + float(gaps[first_pos].item())
        ideal_start = max(channel_cursor[ch], (1.0 - gap_blend) * same_start + gap_blend * any_start)
        # 正の gap（ideal_start − cursor ≥ 0）だけを gap_scale で縮める。 phone 長は不変。
        utt_start = channel_cursor[ch] + gap_scale * (ideal_start - channel_cursor[ch])
        starts[first_pos] = channel_cursor[ch]
        ends[first_pos] = utt_start
        cursor = utt_start
        for k in range(first_pos + 1, N):
            if not bool(attn_mask[k].item()):
                continue
            if int(channel_id[k].item()) != ch:
                continue
            if int(utt_index[k].item()) != u:
                if int(token_type[k].item()) in (TOKEN_TYPE_PRE_SIL, TOKEN_TYPE_FIRST_SIL):
                    break
                continue
            if int(token_type[k].item()) != TOKEN_TYPE_PHONE:
                continue
            d = max(0.0, float(durations[k].item()))
            starts[k] = cursor
            cursor += d
            ends[k] = cursor
            placed_phone[k] = True
        channel_cursor[ch] = max(channel_cursor[ch], cursor)
        last_any_end = max(last_any_end, cursor)
    for ch in [0, 1]:
        cursor = channel_cursor[ch]
        for k in range(N):
            if not bool(attn_mask[k].item()):
                continue
            if int(channel_id[k].item()) != ch:
                continue
            if int(token_type[k].item()) != TOKEN_TYPE_PHONE or bool(placed_phone[k].item()):
                continue
            d = max(0.0, float(durations[k].item()))
            starts[k] = cursor
            cursor += d
            ends[k] = cursor
            placed_phone[k] = True
    return {"token_starts": starts.cpu(), "token_ends": ends.cpu()}


def _timeline_to_raster(timeline, token_type, phone_id, channel_id, attn_mask, T: int, phone_vocab_size: int,
                        fit_margin: int = 5):
    phone = {
        0: torch.full((T,), 1, dtype=torch.long),
        1: torch.full((T,), 1, dtype=torch.long),
    }
    activity = {
        0: torch.zeros(T, dtype=torch.float32),
        1: torch.zeros(T, dtype=torch.float32),
    }
    starts = timeline["token_starts"].clone()
    ends = timeline["token_ends"].clone()
    # --- scale-to-fit: timeline が T を超える場合のみ線形圧縮して末尾発話の切れを防ぐ（kaburi_stat の _scale_placements_to_fit と同思想）---
    max_end = 0.0
    for k in range(token_type.shape[0]):
        if bool(attn_mask[k].item()) and int(token_type[k].item()) == TOKEN_TYPE_PHONE:
            max_end = max(max_end, float(ends[k].item()))
    target = float(max(1, T - fit_margin))
    if max_end > target:
        sc = target / max_end
        starts = starts * sc
        ends = ends * sc
        print(f"[fit] timeline {max_end:.0f} > {T} → scale x{sc:.3f}", flush=True)
    for k in range(token_type.shape[0]):
        if not bool(attn_mask[k].item()) or int(token_type[k].item()) != TOKEN_TYPE_PHONE:
            continue
        ch = int(channel_id[k].item())
        s = max(0, int(round(float(starts[k].item()))))
        e = min(T, int(round(float(ends[k].item()))))
        if e <= s:
            continue
        ph = int(phone_id[k].item())
        if ph <= 0 or ph >= phone_vocab_size:
            ph = 1
        phone[ch][s:e] = ph
        if ph != 1:  # 学習データでは activity ≡ (phone != sil)。sil に 1 を塗ると分布外
            activity[ch][s:e] = 1.0
    return phone[0], phone[1], activity[0], activity[1]


def _replace_batch_raster(batch: dict, phone_A, phone_B, activity_A, activity_B, boundary_radius: int):
    batch["phone_A"] = phone_A.unsqueeze(0)
    batch["phone_B"] = phone_B.unsqueeze(0)
    batch["activity_A"] = activity_A.unsqueeze(0)
    batch["activity_B"] = activity_B.unsqueeze(0)
    for ch, ph in [("A", phone_A), ("B", phone_B)]:
        feats = build_soft_phone_features(ph, boundary_radius=boundary_radius)
        for k, v in feats.items():
            batch[f"{k}_{ch}"] = v.unsqueeze(0)
    return batch


def _apply_ref_pack(batch: dict, pack_dir: str) -> dict:
    """batch の話者参照 latent を ref pack (= make_ref_pack.py 出力) のものに差し替える。"""
    pack = Path(pack_dir)
    row = json.loads((pack / "manifest.jsonl").read_text().splitlines()[0])
    for ch, key in [("A", "ref_A_path"), ("B", "ref_B_path")]:
        p = Path(row[key])
        if not p.is_absolute():
            p = pack / p
        d = torch.load(str(p), map_location="cpu", weights_only=True)
        lat = d.get("ref_latent", d.get("latent")).float().unsqueeze(0)
        batch[f"ref_latent_{ch}"] = lat
        batch[f"ref_mask_{ch}"] = torch.ones(1, lat.shape[1], dtype=torch.bool)
    return batch


def _swap_channel_refs(batch: dict):
    for a_key, b_key in [
        ("ref_latent_A", "ref_latent_B"),
        ("ref_mask_A", "ref_mask_B"),
        ("text_A_input_ids", "text_B_input_ids"),
        ("text_A_mask", "text_B_mask"),
    ]:
        batch[a_key], batch[b_key] = batch[b_key], batch[a_key]
    return batch


def _find_index(ds, chunk_id: str):
    for i, e in enumerate(ds.entries):
        if str(e.get("chunk_id")) == chunk_id:
            return i
    return None


def _find_predictor_index(ds: TimingPredictorDataset, chunk_id: str):
    for k, inner_idx in enumerate(ds.valid_indices):
        if str(ds.inner.entries[inner_idx].get("chunk_id")) == chunk_id:
            return k
    return None


def _build_modified_record(template_rec: dict, new_utts_spec: list[dict], combo_A: str, combo_B: str) -> dict:
    new_rec = dict(template_rec)
    new_utts = []
    for i, u in enumerate(new_utts_spec):
        speaker = str(u["speaker"])
        n_phones = int(u.get("n_phones", len(u.get("phone_ids", []))))
        if n_phones <= 8:
            length_bin = "short"
        elif n_phones >= 26:
            length_bin = "long"
        else:
            length_bin = "mid"
        prev_speaker = new_utts[-1]["speaker"] if new_utts else None
        transition_type = "chunk_first" if prev_speaker is None else f"{prev_speaker}_to_{speaker}"
        start = i * 10
        end = start + 1
        new_utts.append({
            "utt_id": f"text_u{i:03d}",
            "speaker": speaker,
            "text": str(u.get("text", "")),
            "phone_ids": [int(x) for x in u.get("phone_ids", [])],
            "n_phones": n_phones,
            "gt_start_frame": start,
            "gt_end_frame": end,
            "gt_region_len": 1,
            "gt_phone_durations": [1] * n_phones,
            "gt_inter_utt_gap": 0,
            "gt_same_spk_inter_utt_gap": 0,
            "prev_speaker": prev_speaker,
            "next_speaker": None,
            "transition_type": transition_type,
            "length_bin": length_bin,
            "position_in_chunk": i / max(1, len(new_utts_spec) - 1),
            "is_chunk_first": i == 0,
        })
    for i in range(len(new_utts) - 1):
        new_utts[i]["next_speaker"] = new_utts[i + 1]["speaker"]
    new_rec["utterances"] = new_utts
    new_rec["n_utts"] = len(new_utts)
    new_rec["speaker_A"] = combo_A
    new_rec["speaker_B"] = combo_B
    return new_rec


def _load_text_dialog(phones_json: str, dialog_id: str | None, combo_id: str | None):
    raw = json.load(open(phones_json, encoding="utf-8"))
    dialogs = raw.get("dialogs", [])
    if not dialogs:
        raise ValueError(f"no dialogs in {phones_json}")
    if dialog_id:
        dialog = next((d for d in dialogs if str(d.get("id")) == dialog_id), None)
        if dialog is None:
            raise ValueError(f"dialog not found: {dialog_id}")
    else:
        dialog = dialogs[0]
    combos = raw.get("speaker_combos", [])
    combo = None
    if combo_id:
        combo = next((c for c in combos if str(c.get("id")) == combo_id), None)
        if combo is None:
            raise ValueError(f"speaker combo not found: {combo_id}")
    elif combos:
        combo = combos[0]
    return dialog, combo


def _load_predictor(cfg, ckpt_path: str, device: torch.device):
    mcfg = cfg["model"]
    model = KaburiTimingPredictor(
        phone_vocab_size=int(mcfg["phone_vocab_size"]),
        n_speakers_max=int(mcfg["n_speakers_max"]),
        hidden=int(mcfg["hidden"]),
        layers=int(mcfg["layers"]),
        heads=int(mcfg["heads"]),
        ff_dim=int(mcfg["ff_dim"]),
        dropout=float(mcfg["dropout"]),
        max_pos=int(mcfg["max_pos"]),
        use_text=bool(mcfg.get("use_text", False)),
        text_vocab_size=int(mcfg.get("text_vocab_size", 100000)),
        text_emb_dim=int(mcfg.get("text_emb_dim", 64)),
        use_utt_context=bool(mcfg.get("use_utt_context", False)),
        use_dialog_order_context=bool(mcfg.get("use_dialog_order_context", False)),
        utt_context_layers=int(mcfg.get("utt_context_layers", 1)),
        max_dialog_pos=int(mcfg.get("max_dialog_pos", 256)),
    ).to(device)
    if ckpt_path is None:
        from huggingface_hub import hf_hub_download
        from kaburi_tts import HF_PREDICTOR_FILE, HF_REPO
        ckpt_path = hf_hub_download(repo_id=HF_REPO, filename=HF_PREDICTOR_FILE)
    if str(ckpt_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict, step = load_file(str(ckpt_path), device=str(device)), -1
    else:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict, step = state["model_state_dict"], int(state.get("step", -1))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, step


@torch.no_grad()
def _predict_raster_from_item(model, pred_item: dict, device: torch.device, *, T: int, gap_blend: float,
                              gap_bias_frames: float, phone_duration_scale: float, phone_vocab_size: int,
                              dur_sample_temp: float = 0.0, dur_seed: int = 0,
                              presil_scale: float = 1.0, final_scale: float = 1.0,
                              dur_gamma: float = 1.0, class_table=None,
                              dur_lm=None, dur_lm_temp: float = 1.0,
                              inserted_sil_scale: float = 1.0,
                              duration_calibration=None,
                              duration_calibration_total: str = "none"):
    bd = collate_timing([pred_item])
    bd = _move_batch(bd, device)
    phone_logits, sil_logits, gap_logits = model(
        bd["timing_token_type"],
        bd["timing_phone_id"],
        bd["timing_channel_id"],
        bd["timing_speaker_id_per_token"],
        bd["timing_pos_in_channel"],
        bd["timing_mask"],
        bd.get("timing_utt_text_ids"),
        bd.get("timing_utt_text_mask"),
        bd.get("timing_utt_index"),
        bd.get("timing_is_utt_first_token"),
        bd.get("timing_dialog_rank"),
        bd.get("timing_transition_type_id"),
        bd.get("timing_length_bin_id"),
    )
    token_type = bd["timing_token_type"][0]
    channel_id = bd["timing_channel_id"][0]
    phone_id = bd["timing_phone_id"][0]
    utt_index = bd["timing_utt_index"][0]
    mask = bd["timing_mask"][0]
    if dur_sample_temp > 0:
        gen = torch.Generator(device=phone_logits.device)
        gen.manual_seed(int(dur_seed))
        durations = _decode_durations(phone_logits[0], sil_logits[0], token_type,
                                      phone_mode="sample", sample_temp=dur_sample_temp,
                                      generator=gen)
    else:
        durations = _decode_durations(phone_logits[0], sil_logits[0], token_type)
    calib_meta = None
    if duration_calibration is not None:
        # --- MFA 制約付き posterior calibration (境界文脈への exponential tilt) ---
        # 旧 lenfix 系スケールとの二重適用は仕様違反 (CLAUDE_CODE_MFA_DURATION_CALIBRATION.md §10)
        if any(x != 1.0 for x in (presil_scale, final_scale, inserted_sil_scale, dur_gamma)) \
                or dur_sample_temp > 0 or dur_lm is not None:
            print("[calib][WARNING] duration_calibration と旧補正 (presil/final/sil_scale/"
                  "gamma/sample/dur_lm) の二重適用が指定されています", flush=True)
        from kaburi_tts.predictor.duration_calibration import (
            classify_contexts, project_total, tilt_expected,
        )
        utt_idx_c = bd["timing_utt_index"][0]
        is_ph_c = (token_type == TOKEN_TYPE_PHONE) & mask.bool()
        raw_total = float(durations[is_ph_c].sum())
        n_tok = token_type.shape[0]
        deltas = []
        unmet_total = 0.0
        ctx_counts: dict[str, int] = {}
        k = 0
        while k < n_tok:
            if not bool(is_ph_c[k]):
                k += 1
                continue
            u = int(utt_idx_c[k])
            ks = []
            while k < n_tok and bool(is_ph_c[k]) and int(utt_idx_c[k]) == u:
                ks.append(k)
                k += 1
            ids = [int(phone_id[j]) for j in ks]
            ctxs = classify_contexts(ids)
            for c in ctxs:
                ctx_counts[c] = ctx_counts.get(c, 0) + 1
            lg = phone_logits[0][torch.tensor(ks)]
            exp, var = tilt_expected(lg, ctxs, duration_calibration)
            if duration_calibration_total == "raw":
                target = float(sum(float(durations[j]) for j in ks))
                exp, unmet = project_total(exp, ctxs, var, target)
                unmet_total += unmet
            elif duration_calibration_total in ("lenfix", "lenfix_up"):
                target = 0.0
                for j, c in zip(ks, ctxs):
                    d = float(durations[j])
                    if c == "pre_internal_sil_1":
                        d *= 2.0
                    elif c == "utterance_final_1":
                        d *= 1.45
                    target += d
                if duration_calibration_total == "lenfix_up" and target <= float(exp.sum()):
                    pass  # 較正 posterior 平均より短くする方向の projection はしない
                          # (境界トークンの押し潰し = 局所激速の防止, 2026-07-19)
                else:
                    exp, unmet = project_total(exp, ctxs, var, target)
                    unmet_total += unmet
            for j, e in zip(ks, exp.tolist()):
                deltas.append(abs(e - float(durations[j])))
                durations[j] = max(1.0, float(e))
        calib_meta = {
            "duration_calibration_total_mode": duration_calibration_total,
            "raw_phone_duration_total": raw_total,
            "calibrated_phone_duration_total_after_projection": float(durations[is_ph_c].sum()),
            "duration_projection_unmet_frames": unmet_total,
            "context_counts": ctx_counts,
            "mean_abs_duration_delta": float(sum(deltas) / max(len(deltas), 1)),
            "max_abs_duration_delta": float(max(deltas) if deltas else 0.0),
        }
    if dur_lm is not None:
        # --- DurationLM: MFA 実測分布を模倣する AR モデルで音素継続長を置換 ---
        # (発話単位で、直前継続長に条件付けた首尾一貫サンプル。sil 長も同モデル)
        from kaburi_tts.predictor.duration_lm import sample_durations
        gen = torch.Generator(device=str(next(dur_lm.parameters()).device))
        gen.manual_seed(int(dur_seed))
        utt_idx_t = bd["timing_utt_index"][0]
        is_ph_t = (token_type == TOKEN_TYPE_PHONE) & mask.bool()
        k = 0
        n_tok = token_type.shape[0]
        while k < n_tok:
            if not bool(is_ph_t[k]):
                k += 1
                continue
            u = int(utt_idx_t[k])
            ks = []
            while k < n_tok and bool(is_ph_t[k]) and int(utt_idx_t[k]) == u:
                ks.append(k)
                k += 1
            ids = [int(phone_id[j]) for j in ks]
            durs = sample_durations(dur_lm, ids, device=str(next(dur_lm.parameters()).device),
                                    temperature=dur_lm_temp, generator=gen)
            # テンポは v1 (expected) に合わせ、ミクロリズムだけ LM から採る:
            # MFA 実測分布は自発対話の実速度 (2.18 fr/ph) で、合成でそのまま鳴らすと
            # 速すぎて不自然 (2026-07-18 E6)。発話合計を expected 合計へ再スケール
            # (nPVI・伸長の相対構造はスケール不変)。
            exp_sum = float(sum(float(durations[j]) for j in ks))
            lm_sum = float(sum(durs))
            sc = exp_sum / lm_sum if lm_sum > 0 else 1.0
            sc = min(max(sc, 0.7), 1.4)
            for j, d in zip(ks, durs):
                durations[j] = max(1.0, float(d) * sc)
    if phone_duration_scale != 1.0:
        phone_mask = token_type == TOKEN_TYPE_PHONE
        durations = torch.where(phone_mask, durations * float(phone_duration_scale), durations)
    if inserted_sil_scale != 1.0:
        # 発話内 <sil> トークン (= phone_id 1 の PHONE トークン) の継続長スケール。
        # 「区切りました感」の緩和用 (外部レビュー提案、E8 グリッド)。
        sil_mask = (token_type == TOKEN_TYPE_PHONE) & (phone_id == 1) & mask.bool()
        durations = torch.where(sil_mask,
                                (durations * float(inserted_sil_scale)).clamp(min=1.0),
                                durations)
    # --- MFA 実測との系統差の補正 (2026-07-18 dur_analysis.py 較正) ---
    # expected デコードは条件付き平均のため、実測比で (a) ポーズ直前音素 (GT 5.89 vs
    # pred 2.94)・発話末音素 (4.20 vs 2.89) の伸長を過小予測し、(b) 分散が 1/4 に潰れて
    # nPVI (リズム局所変動) が半減する。(a) は実測比の決定的スケール、(b) はモデル自身の
    # 予測偏差のクラス条件付き増幅 (dur_gamma) で復元する — 乱数は注入しない。
    if dur_gamma != 1.0 or presil_scale != 1.0 or final_scale != 1.0:
        is_ph = (token_type == TOKEN_TYPE_PHONE) & mask.bool()
        utt_idx = bd["timing_utt_index"][0]
        n_tok = token_type.shape[0]
        if dur_gamma != 1.0 and class_table is not None:
            cls = class_table.to(durations.device)[phone_id.clamp(0, class_table.shape[0] - 1)]
            new_d = durations.clone()
            for c in torch.unique(cls[is_ph]):
                m = is_ph & (cls == c) & (phone_id != 1)
                if int(m.sum()) < 3:
                    continue
                mu = durations[m].mean()
                new_d[m] = mu + float(dur_gamma) * (durations[m] - mu)
            durations = new_d.clamp(min=1.0)
        if presil_scale != 1.0 or final_scale != 1.0:
            for k in range(n_tok):
                if not bool(is_ph[k]) or int(phone_id[k]) == 1:
                    continue
                nk = k + 1
                same_utt_next = (nk < n_tok and bool(is_ph[nk])
                                 and int(utt_idx[nk]) == int(utt_idx[k]))
                if same_utt_next and int(phone_id[nk]) == 1:
                    durations[k] = durations[k] * float(presil_scale)
                elif not same_utt_next:
                    durations[k] = durations[k] * float(final_scale)
    durations, fit_info = _fit_durations_to_T(durations, token_type, channel_id, mask, T)
    gaps = _decode_any_gaps(gap_logits[0], bd["timing_is_utt_first_token"][0])
    order = bd["timing_global_utt_order"][0]
    if gap_bias_frames != 0.0:
        for pos in order[1:]:
            pos = int(pos)
            if 0 <= pos < gaps.shape[0]:
                gaps[pos] = gaps[pos] + float(gap_bias_frames)
    # --- timeline が T 超なら _timeline_to_raster で **全体一様 scale-to-fit**（A/B 共通・gap-only fit は廃止）。---
    # phone も gap も均一に圧縮されるため間が均一（gap だけが詰まらない）。
    FIT_MARGIN = 5
    tt_cpu, ph_cpu, ch_cpu, am_cpu = token_type.cpu(), phone_id.cpu(), channel_id.cpu(), mask.cpu()
    timeline = _reconstruct_with_any_gap(
        durations, gaps, token_type, channel_id, utt_index, mask, order, T, gap_blend=gap_blend,
    )
    phone_sel = (tt_cpu == TOKEN_TYPE_PHONE) & am_cpu
    raw_max_end = float(timeline["token_ends"][phone_sel].max().item()) if bool(phone_sel.any()) else 0.0
    print(f"[fit] raw_max_end={raw_max_end:.0f} target={T-FIT_MARGIN} "
          f"(>{T-FIT_MARGIN} なら全体一様 scale)", flush=True)
    phone_A, phone_B, activity_A, activity_B = _timeline_to_raster(
        timeline, tt_cpu, ph_cpu, ch_cpu, am_cpu, T, phone_vocab_size, fit_margin=FIT_MARGIN,
    )
    meta = {
        **fit_info,
        "raw_max_end": float(raw_max_end),
        "active_A": float(activity_A.mean().item()),
        "active_B": float(activity_B.mean().item()),
        "overlap": float(((activity_A > 0.5) & (activity_B > 0.5)).float().mean().item()),
        "n_tokens": int(mask.sum().item()),
        "phone_duration_scale": float(phone_duration_scale),
    }
    if calib_meta is not None:
        meta.update(calib_meta)
    return phone_A, phone_B, activity_A, activity_B, meta


@torch.no_grad()
def _predict_raster(model, pred_ds, pred_idx: int, device: torch.device, *, T: int, gap_blend: float,
                    gap_bias_frames: float, phone_duration_scale: float, phone_vocab_size: int):
    return _predict_raster_from_item(
        model, pred_ds[pred_idx], device, T=T, gap_blend=gap_blend,
        gap_bias_frames=gap_bias_frames, phone_duration_scale=phone_duration_scale,
        phone_vocab_size=phone_vocab_size,
    )


def main():
    ap = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    ap.add_argument("--predictor-config", default=str(repo_root / "configs/predictor.yaml"))
    ap.add_argument("--predictor-ckpt", default=None, help="local .pt/.safetensors; omit to download from HF")
    ap.add_argument("--acoustic-config", default=str(repo_root / "configs/acoustic.yaml"))
    ap.add_argument("--acoustic-ckpt", default=None, help="local .pt/.safetensors; omit to download from HF")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--chunk-ids", default="", help="合成対象の chunk ID (カンマ区切り)")
    ap.add_argument("--phones-json", default=None,
                    help="Hand-authored text converted to phone ids. If set, synthesize this dialog from text.")
    ap.add_argument("--dialog-id", default=None)
    ap.add_argument("--speaker-combo-id", default=None)
    ap.add_argument("--template-chunk", default=None,
                    help="Reference/template chunk for text synthesis. Defaults to combo template_chunk or first chunk-id.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cfg-scale", type=float, default=2.5)
    ap.add_argument("--num-steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gap-blend", type=float, default=0.5)
    ap.add_argument("--gap-bias-frames", type=float, default=0.0)
    ap.add_argument("--phone-duration-scale", type=float, default=1.0,
                    help="Scale predicted PHONE durations before fitting to the raster budget.")
    ap.add_argument("--also-gt", action="store_true")
    ap.add_argument("--ref-pack", default=None,
                    help="話者参照を ref pack (make_ref_pack.py 出力) の latent に差し替える")
    args = ap.parse_args()

    device = torch.device(args.device)
    pred_cfg = yaml.safe_load(Path(args.predictor_config).read_text())
    timing_cfg = yaml.safe_load(Path(args.acoustic_config).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = PretrainedTextTokenizer.from_pretrained("llm-jp/llm-jp-3-150m", local_files_only=False)
    pccfg = timing_cfg.get("phone_condition", {})
    T = int(timing_cfg["data"].get("latent_T", 750))
    boundary_radius = int(pccfg.get("boundary_radius", 5))
    phone_vocab_size = int(timing_cfg["model"]["phone_vocab_size"])

    acoustic_ds = Normal2StreamDataset(
        args.manifest, text_tokenizer=tok, max_text_len=int(timing_cfg["data"].get("max_text_len", 64)),
        latent_T=T, speaker_balanced=False,
        soft_phone=(str(pccfg.get("mode", "hard")) == "soft"),
        soft_boundary_radius=boundary_radius,
    )
    pred_ds = TimingPredictorDataset(
        args.manifest, list(pred_cfg["data"]["utt_timing_jsonls"]),
        text_tokenizer=tok, max_text_len=64, latent_T=T,
        soft_radius=boundary_radius,
        max_collapsed_seq_len=int(pred_cfg["context"]["max_collapsed_seq_len"]),
        phone_class_lookup=build_phone_class_table_8(pred_cfg["data"]["phone_vocab_path"]),
        T_frames=T, max_timing_seq_len=int(pred_cfg["timing"]["max_timing_seq_len"]),
        max_utt_text_len=int(pred_cfg["model"].get("max_utt_text_len", 64)),
        utt_manifest_path=pred_cfg["data"].get("utt_manifest_path"),
    )

    predictor, pred_step = _load_predictor(pred_cfg, args.predictor_ckpt, device)
    acoustic = build_acoustic_model(timing_cfg, device)
    acoustic_step = load_acoustic_checkpoint(acoustic, args.acoustic_ckpt, device)
    pct8 = build_phone_class_table_8(timing_cfg["data"]["phone_vocab_path"]).to(device)
    codec = DACVAECodec.load(device=str(device), dtype=torch.bfloat16)

    meta_rows = []
    text_dialog = None
    text_combo = None
    text_chunk_ids = None
    if args.phones_json:
        text_dialog, text_combo = _load_text_dialog(args.phones_json, args.dialog_id, args.speaker_combo_id)
        template_chunk = args.template_chunk
        if template_chunk is None and text_combo is not None:
            template_chunk = text_combo.get("template_chunk")
        if template_chunk is None:
            template_chunk = next((c.strip() for c in args.chunk_ids.split(",") if c.strip()), None)
        if template_chunk is None:
            raise ValueError("text synthesis requires --template-chunk or --chunk-ids")
        text_chunk_ids = [str(template_chunk)]

    run_chunk_ids = text_chunk_ids or [c.strip() for c in args.chunk_ids.split(",") if c.strip()]
    for offset, chunk_id in enumerate(run_chunk_ids):
        aidx = _find_index(acoustic_ds, chunk_id)
        pidx = _find_predictor_index(pred_ds, chunk_id)
        if aidx is None or pidx is None:
            print(f"[skip] {chunk_id}: acoustic_idx={aidx} predictor_idx={pidx}", flush=True)
            continue
        base_item = acoustic_ds[aidx]
        batch = collate_eventmix([base_item])
        if args.ref_pack:
            batch = _apply_ref_pack(batch, args.ref_pack)
        sample_id = chunk_id
        if text_dialog is not None:
            template_rec = pred_ds.utt_timing[chunk_id]
            manifest_row = acoustic_ds.entries[aidx]
            template_A = str(manifest_row.get("speaker_A") or manifest_row.get("spk_A") or template_rec.get("speaker_A", ""))
            template_B = str(manifest_row.get("speaker_B") or manifest_row.get("spk_B") or template_rec.get("speaker_B", ""))
            combo_A = str((text_combo or {}).get("A") or template_A)
            combo_B = str((text_combo or {}).get("B") or template_B)
            if combo_A == template_B and combo_B == template_A:
                batch = _swap_channel_refs(batch)
            elif combo_A != template_A or combo_B != template_B:
                print(
                    f"[warn] combo {combo_A}/{combo_B} does not match template {template_A}/{template_B}; "
                    "using template refs as-is",
                    flush=True,
                )
                combo_A, combo_B = template_A, template_B
            new_rec = _build_modified_record(template_rec, text_dialog["utts"], combo_A, combo_B)
            pred_item = pred_ds._attach_timing(pred_ds[pidx], new_rec)
            sample_id = f"{text_dialog.get('id', 'text_dialog')}__{(text_combo or {}).get('id', combo_A + 'x' + combo_B)}"
            phone_A, phone_B, activity_A, activity_B, timing_meta = _predict_raster_from_item(
                predictor, pred_item, device, T=T, gap_blend=args.gap_blend,
                gap_bias_frames=args.gap_bias_frames,
                phone_duration_scale=args.phone_duration_scale,
                phone_vocab_size=phone_vocab_size,
            )
        else:
            phone_A, phone_B, activity_A, activity_B, timing_meta = _predict_raster(
                predictor, pred_ds, pidx, device, T=T, gap_blend=args.gap_blend,
                gap_bias_frames=args.gap_bias_frames,
                phone_duration_scale=args.phone_duration_scale,
                phone_vocab_size=phone_vocab_size,
            )
        pred_batch = _replace_batch_raster(dict(batch), phone_A, phone_B, activity_A, activity_B, boundary_radius)
        pred_batch = _move_batch(pred_batch, device)
        seed = int(args.seed + offset)
        wav = synth_acoustic(
            acoustic, codec, pred_batch, device=device, num_steps=args.num_steps,
            seed=seed, cfg_scale=args.cfg_scale, pct8=pct8,
        )
        out = out_dir / (
            f"{sample_id}__pred{pred_step}_ac{acoustic_step}"
            f"_cfg{args.cfg_scale}_gb{args.gap_blend}_bias{args.gap_bias_frames}"
            f"_pdur{args.phone_duration_scale}.wav"
        )
        torchaudio.save(str(out), wav, codec.sample_rate, channels_first=True)
        print(f"[save] {out}", flush=True)
        row = {
            "chunk_id": chunk_id,
            "sample_id": sample_id,
            "phones_json": args.phones_json,
            "pred_step": pred_step,
            "acoustic_step": acoustic_step,
            "cfg_scale": args.cfg_scale,
            "num_steps": args.num_steps,
            "seed": seed,
            "gap_blend": args.gap_blend,
            "gap_bias_frames": args.gap_bias_frames,
            "phone_duration_scale": args.phone_duration_scale,
            "pred_wav": str(out),
            **timing_meta,
        }
        if args.also_gt:
            gt_batch = _move_batch(batch, device)
            gt_wav = synth_acoustic(
                acoustic, codec, gt_batch, device=device, num_steps=args.num_steps,
                seed=seed, cfg_scale=args.cfg_scale, pct8=pct8,
            )
            gt_out = out_dir / f"{chunk_id}__gt_ac{acoustic_step}_cfg{args.cfg_scale}.wav"
            torchaudio.save(str(gt_out), gt_wav, codec.sample_rate, channels_first=True)
            print(f"[save] {gt_out}", flush=True)
            row["gt_wav"] = str(gt_out)
        meta_rows.append(row)

    (out_dir / "metadata.json").write_text(json.dumps(meta_rows, indent=2, ensure_ascii=False))
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()

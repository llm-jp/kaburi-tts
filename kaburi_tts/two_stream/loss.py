"""2-stream RF loss + 診断指標。

normal sample (= chunk全体): active-frame weighted MSE (silence_weight=0.05)。
event sample (= margin 付き短窓): core_mask に基づく weighted MSE
  weight = core_mask + margin_weight * (latent_mask & ~core_mask)
  margin_weight=0 で margin frames は loss に寄与しない (input としては与える)。

valid breakdown:
  normal loss vs event loss (sample_kind)
  per-event-type loss (boundary / short / overlap / turn_switch)
  既存診断 (nonoverlap_active / overlap_active / consonant / vowel / boundary)
"""

from __future__ import annotations

import torch


def rf_velocity_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return noise - x0


def rf_interpolate(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (1.0 - t[:, None, None]) * x0 + t[:, None, None] * noise


def _stream_normal_loss(
    v_pred: torch.Tensor, v_target: torch.Tensor,
    activity: torch.Tensor, silence_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """active/silence weighted MSE。"""
    err = (v_pred - v_target).pow(2).mean(dim=-1)   # (B, T)
    sil = 1.0 - activity
    w = activity + silence_weight * sil
    weighted = (err * w).sum() / w.sum().clamp_min(1e-6)
    L_active = (err * activity).sum() / activity.sum().clamp_min(1e-6)
    L_silence = (err * sil).sum() / sil.sum().clamp_min(1e-6)
    return weighted, L_active, L_silence


def _stream_event_loss(
    v_pred: torch.Tensor, v_target: torch.Tensor,
    latent_mask: torch.Tensor, core_mask: torch.Tensor,
    margin_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """core_mask weighted MSE (event sample 用)。"""
    err = (v_pred - v_target).pow(2).mean(dim=-1)
    m_valid = latent_mask
    m_core = core_mask & m_valid
    m_margin = m_valid & ~m_core
    w = m_core.float() + float(margin_weight) * m_margin.float()
    weighted = (err * w).sum() / w.sum().clamp_min(1e-6)
    L_core = (err * m_core.float()).sum() / m_core.float().sum().clamp_min(1e-6)
    L_margin = (err * m_margin.float()).sum() / m_margin.float().sum().clamp_min(1e-6) if m_margin.sum() > 0 else torch.tensor(0.0, device=err.device)
    return weighted, L_core, L_margin


def compute_2stream_loss_eventmix(
    *,
    v_pred_A: torch.Tensor, v_pred_B: torch.Tensor,
    v_target_A: torch.Tensor, v_target_B: torch.Tensor,
    activity_A: torch.Tensor, activity_B: torch.Tensor,
    sample_kind: list[str],            # batch ごとの "normal" or "event"
    core_mask: torch.Tensor,           # (B, T) bool: event は core True、 normal は全 True
    latent_mask: torch.Tensor,         # (B, T) bool: padding 除外
    silence_weight: float = 0.05,      # normal sample 用
    margin_weight: float = 0.0,        # event sample 用 (margin frame の重み)
) -> dict[str, torch.Tensor]:
    """sample_kind に応じて行ごとに normal/event 経路で loss を計算、 batch 平均。"""
    B = v_pred_A.shape[0]
    is_event = torch.tensor([sk == "event" for sk in sample_kind], device=v_pred_A.device)
    # normal mask は activity ベース、 event は core ベース。 各 sample (行) の loss を計算して平均。
    # 実装: 単純に各行を loop (B≦8 想定で問題なし)、 あるいは 1 つの式に統合
    losses_A, losses_B = [], []
    diag = {"loss_A_active": [], "loss_A_silence": [], "loss_B_active": [], "loss_B_silence": [],
            "loss_core_A": [], "loss_core_B": [], "loss_margin_A": [], "loss_margin_B": []}
    for b in range(B):
        v_pA = v_pred_A[b : b + 1]; v_pB = v_pred_B[b : b + 1]
        v_tA = v_target_A[b : b + 1]; v_tB = v_target_B[b : b + 1]
        if not is_event[b]:
            # normal path
            wA, lA_act, lA_sil = _stream_normal_loss(v_pA, v_tA, activity_A[b : b + 1], silence_weight)
            wB, lB_act, lB_sil = _stream_normal_loss(v_pB, v_tB, activity_B[b : b + 1], silence_weight)
            losses_A.append(wA); losses_B.append(wB)
            diag["loss_A_active"].append(lA_act.detach()); diag["loss_A_silence"].append(lA_sil.detach())
            diag["loss_B_active"].append(lB_act.detach()); diag["loss_B_silence"].append(lB_sil.detach())
        else:
            # event: core_mask weighted
            wA, lA_core, lA_margin = _stream_event_loss(
                v_pA, v_tA, latent_mask[b : b + 1], core_mask[b : b + 1], margin_weight,
            )
            wB, lB_core, lB_margin = _stream_event_loss(
                v_pB, v_tB, latent_mask[b : b + 1], core_mask[b : b + 1], margin_weight,
            )
            losses_A.append(wA); losses_B.append(wB)
            diag["loss_core_A"].append(lA_core.detach()); diag["loss_margin_A"].append(lA_margin.detach())
            diag["loss_core_B"].append(lB_core.detach()); diag["loss_margin_B"].append(lB_margin.detach())
    loss_A = torch.stack(losses_A).mean()
    loss_B = torch.stack(losses_B).mean()
    loss = 0.5 * (loss_A + loss_B)
    out = {"loss": loss, "loss_A": loss_A.detach(), "loss_B": loss_B.detach()}
    # diagnostic averages (該当 sample が無ければ 0)
    for k, lst in diag.items():
        out[k] = torch.stack(lst).mean().detach() if lst else torch.tensor(0.0, device=loss.device)
    return out


def compute_diagnostic_per_event(
    *,
    v_pred_A: torch.Tensor, v_pred_B: torch.Tensor,
    v_target_A: torch.Tensor, v_target_B: torch.Tensor,
    sample_kind: list[str], event_type: list[str],
    core_mask: torch.Tensor, latent_mask: torch.Tensor,
    activity_A: torch.Tensor, activity_B: torch.Tensor,
    phone_A: torch.Tensor, phone_B: torch.Tensor,
    phone_class_table: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """各 sample 行を event_type に振り分け、 sample 単位 mean loss を計算。

    返り値:
      normal_loss              : normal sample の active-weighted loss (silence_weight=0.05)
      event_loss               : event sample の core-weighted loss
      event_<category>_loss    : event の各 category (boundary/short/overlap/turn_switch)
      nonoverlap_active_loss、 overlap_active_loss : normal sample 内のみ計算
      consonant_loss、 vowel_loss、 boundary_loss : normal + event 全体、 core 内のみ
    """
    err_A = (v_pred_A - v_target_A).pow(2).mean(dim=-1)   # (B, T)
    err_B = (v_pred_B - v_target_B).pow(2).mean(dim=-1)
    B, T = err_A.shape
    out: dict[str, list[torch.Tensor]] = {
        "normal_loss": [], "event_loss": [],
        "boundary_loss": [], "short_loss": [], "overlap_loss": [], "turn_switch_loss": [],
    }
    # core 内 frame ベースの phone / boundary 集計用 buffer
    cons_all_err: list[torch.Tensor] = []
    vow_all_err: list[torch.Tensor] = []
    bd_all_err: list[torch.Tensor] = []
    nonov_all_err: list[torch.Tensor] = []
    ov_all_err: list[torch.Tensor] = []

    pc_A = phone_class_table.to(phone_A.device)[phone_A]
    pc_B = phone_class_table.to(phone_B.device)[phone_B]
    act_A_bool = activity_A > 0.5
    act_B_bool = activity_B > 0.5

    for b in range(B):
        m_valid = latent_mask[b]
        m_core = core_mask[b] & m_valid
        n_core = m_core.sum()
        if n_core == 0:
            continue
        per = ((err_A[b] + err_B[b]) * 0.5)  # 2-stream avg per frame
        sample_mean = per[m_core].mean().detach()
        sk = sample_kind[b]; et = event_type[b]
        if sk == "normal":
            out["normal_loss"].append(sample_mean)
        else:
            out["event_loss"].append(sample_mean)
            if et.startswith("boundary_"):
                out["boundary_loss"].append(sample_mean)
            elif et.startswith("short_"):
                out["short_loss"].append(sample_mean)
            elif et == "overlap":
                out["overlap_loss"].append(sample_mean)
            elif et.startswith("turn_switch_"):
                out["turn_switch_loss"].append(sample_mean)
        # phone breakdown (core 内のみ)
        for err_x, pc_x, act_x in [(err_A[b], pc_A[b], act_A_bool[b]), (err_B[b], pc_B[b], act_B_bool[b])]:
            cm = m_core & act_x   # active frame に限定
            cons_m = (pc_x == 2) & cm
            vow_m = (pc_x == 1) & cm
            if cons_m.sum() > 0:
                cons_all_err.append(err_x[cons_m].mean().detach())
            if vow_m.sum() > 0:
                vow_all_err.append(err_x[vow_m].mean().detach())
        # boundary frame (activity 境界) — core 内のみ
        ap_A = torch.cat([torch.zeros(1, device=activity_A.device), activity_A[b, :-1]])
        ap_B = torch.cat([torch.zeros(1, device=activity_B.device), activity_B[b, :-1]])
        bd_A = ((activity_A[b] > 0.5) != (ap_A > 0.5)) & m_core
        bd_B = ((activity_B[b] > 0.5) != (ap_B > 0.5)) & m_core
        if bd_A.sum() > 0:
            bd_all_err.append(err_A[b, bd_A].mean().detach())
        if bd_B.sum() > 0:
            bd_all_err.append(err_B[b, bd_B].mean().detach())
        # nonoverlap / overlap (normal sample 内で意味あり; event だと偏るが計算は継続)
        ov_m = act_A_bool[b] & act_B_bool[b] & m_core
        nonov_A = act_A_bool[b] & ~act_B_bool[b] & m_core
        nonov_B = act_B_bool[b] & ~act_A_bool[b] & m_core
        if ov_m.sum() > 0:
            ov_all_err.append(((err_A[b, ov_m] + err_B[b, ov_m]) * 0.5).mean().detach())
        if (nonov_A | nonov_B).sum() > 0:
            va = err_A[b, nonov_A].sum() if nonov_A.sum() > 0 else torch.tensor(0.0, device=err_A.device)
            vb = err_B[b, nonov_B].sum() if nonov_B.sum() > 0 else torch.tensor(0.0, device=err_B.device)
            n = nonov_A.sum() + nonov_B.sum()
            nonov_all_err.append(((va + vb) / n.clamp_min(1)).detach())

    final: dict[str, torch.Tensor] = {}
    for k, lst in out.items():
        if lst:
            final[k] = torch.stack(lst).mean().detach()
    if cons_all_err:
        final["consonant_loss"] = torch.stack(cons_all_err).mean().detach()
    if vow_all_err:
        final["vowel_loss"] = torch.stack(vow_all_err).mean().detach()
    if bd_all_err:
        final["boundary_loss"] = torch.stack(bd_all_err).mean().detach()
    if nonov_all_err:
        final["nonoverlap_active_loss"] = torch.stack(nonov_all_err).mean().detach()
    if ov_all_err:
        final["overlap_active_loss"] = torch.stack(ov_all_err).mean().detach()
    return final


def predicted_silence_ratio_in_active(
    *,
    x0_pred_A: torch.Tensor, x0_pred_B: torch.Tensor,
    activity_A: torch.Tensor, activity_B: torch.Tensor,
    energy_threshold: float = 0.1,
) -> dict[str, torch.Tensor]:
    energy_A = x0_pred_A.pow(2).mean(dim=-1).sqrt()
    energy_B = x0_pred_B.pow(2).mean(dim=-1).sqrt()
    sil_A = (energy_A < energy_threshold).float()
    sil_B = (energy_B < energy_threshold).float()
    pr_A = (sil_A * activity_A).sum() / activity_A.sum().clamp_min(1e-6)
    pr_B = (sil_B * activity_B).sum() / activity_B.sum().clamp_min(1e-6)
    return {"pred_silence_ratio_in_active_A": pr_A, "pred_silence_ratio_in_active_B": pr_B}


def _resolve_repo_path(p):
    """相対パスは repo root 基準で解決 (= config の assets/... 記法用)。"""
    from pathlib import Path
    from kaburi_tts import REPO_ROOT
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def build_phone_class_table(phone_vocab_path: str) -> torch.Tensor:
    """3-class table (0=silence, 1=vowel, 2=consonant)。"""
    import json
    obj = json.loads(_resolve_repo_path(phone_vocab_path).read_text())
    vocab = obj["phone_vocab"]
    inv = {v: k for k, v in vocab.items()}
    SILENCE = {"<sil>", "<pad>", "<eos>", "<bos>", "sp", "sil", "spn", ""}
    VOWELS = {"a", "i", "u", "e", "o", "I", "U"}
    n = max(vocab.values()) + 1
    table = torch.zeros(n, dtype=torch.long)
    for idx in range(n):
        s = inv.get(idx, "").strip().lower().replace("ː", "").replace(":", "")
        if s in SILENCE or not s:
            table[idx] = 0
        elif s in VOWELS or s.lower() in {"a", "i", "u", "e", "o"}:
            table[idx] = 1
        else:
            table[idx] = 2
    return table


# ---------------------------------------------------------------------------
# 8-class phone class table for SoftPhoneConditioner
# ---------------------------------------------------------------------------

PHONE_CLASS_8 = {
    "silence": 0, "vowel": 1, "consonant_general": 2,
    "nasal": 3, "fricative": 4, "plosive": 5, "approximant": 6, "special": 7,
}
N_PHONE_CLASSES_8 = 8
_SIL_SET = {"<sil>", "<pad>", "<eos>", "<bos>", "sp", "sil", "spn", ""}
# IPA-precise Unicode で日本語頻出 phone を網羅 (= 'ɡ' U+0261、 'ɨ' 中舌高、 拗音/無声化母音)
_VOWELS = {"a", "i", "u", "e", "o", "ɯ", "ɨ", "i̥", "ɯ̥", "ɨ̥"}
_NASALS = {"m", "mʲ", "n", "ɲ", "ŋ", "n̩", "ɴ", "ɰ̃"}
_FRICATIVES = {"s", "ɕ", "z", "ʑ", "h", "ç", "ɸ", "ɸʲ", "f", "x", "v", "vʲ"}
# 'g' (U+0067) と 'ɡ' (U+0261) を両方含める
_PLOSIVES = {"p", "t", "k", "b", "d", "g", "ɡ", "ts", "tɕ", "dz", "dʑ", "c", "ɟ", "ʔ"}
_APPROX = {"r", "ɾ", "ɾʲ", "j", "w", "l"}


def _classify_phone_str(s: str) -> int:
    s = s.strip()
    if not s or s.lower() in _SIL_SET:
        return PHONE_CLASS_8["silence"]
    base = s.replace("ː", "").replace(":", "")
    if base in _VOWELS:
        return PHONE_CLASS_8["vowel"]
    if base in _NASALS:
        return PHONE_CLASS_8["nasal"]
    if base in _FRICATIVES:
        return PHONE_CLASS_8["fricative"]
    if base in _PLOSIVES:
        return PHONE_CLASS_8["plosive"]
    if base in _APPROX:
        return PHONE_CLASS_8["approximant"]
    if base and base[0] in {"p", "t", "k", "b", "d", "g", "ɡ", "c", "ɟ"}:
        return PHONE_CLASS_8["plosive"]
    if base and base[0] in {"s", "ɕ", "z", "ʑ", "h", "ç", "ɸ", "f", "v"}:
        return PHONE_CLASS_8["fricative"]
    if base and base[0] in {"m", "n", "ɲ", "ŋ", "ɴ"}:
        return PHONE_CLASS_8["nasal"]
    if base and base[0] in {"r", "ɾ", "j", "w", "l"}:
        return PHONE_CLASS_8["approximant"]
    if base and base[0] in {"a", "i", "u", "e", "o", "ɯ", "ɨ"}:
        return PHONE_CLASS_8["vowel"]
    return PHONE_CLASS_8["special"]


def build_phone_class_table_8(phone_vocab_path: str) -> torch.Tensor:
    """8-class phone class table for phone-soft conditioning。"""
    import json
    obj = json.loads(_resolve_repo_path(phone_vocab_path).read_text())
    vocab = obj["phone_vocab"]
    inv = {v: k for k, v in vocab.items()}
    n = max(vocab.values()) + 1
    table = torch.zeros(n, dtype=torch.long)
    for idx in range(n):
        table[idx] = _classify_phone_str(inv.get(idx, ""))
    return table


# ---------------------------------------------------------------------------
# phone segment 復元 + soft phone feature 生成 (frame-level)
# ---------------------------------------------------------------------------

def build_soft_phone_features(
    phone: torch.Tensor,
    *,
    boundary_radius: int = 5,
    log1p_dur_norm: float = 5.0,
    fallback_phone: int = 1,    # <sil>=1 が semantically 正しい (「neighbor 無し」 = silence 隣接)。 <pad>=0 ではない。
) -> dict[str, torch.Tensor]:
    """frame-level phone tensor (T,) long → soft conditioning feature 群。

    返り値:
      phone_cur            (T,) long
      phone_prev           (T,) long
      phone_next           (T,) long
      phone_pos_frac       (T,) float
      phone_dur_norm       (T,) float in [0,1]   log1p(dur_frames) / log1p_dur_norm
      dist_start_norm      (T,) float in [0,1]   (t - seg_start)/5、 clip
      dist_end_norm        (T,) float in [0,1]
      is_phone_boundary    (T,) float in [0,1]   端 ±boundary_radius で linear decay
    """
    T = int(phone.shape[0])
    if T == 0:
        empty_l = torch.zeros(0, dtype=torch.long)
        empty_f = torch.zeros(0, dtype=torch.float32)
        return {k: empty_l if k.startswith("phone_") and not k.endswith("_frac") and not k.endswith("_norm") else empty_f
                for k in ["phone_cur", "phone_prev", "phone_next",
                          "phone_pos_frac", "phone_dur_norm", "dist_start_norm",
                          "dist_end_norm", "is_phone_boundary"]}
    ph_list = phone.long().tolist()
    seg_starts: list[int] = [0]
    seg_phones: list[int] = [ph_list[0]]
    for t in range(1, T):
        if ph_list[t] != ph_list[t - 1]:
            seg_starts.append(t)
            seg_phones.append(ph_list[t])
    seg_starts.append(T)
    n_seg = len(seg_phones)
    phone_cur = phone.long().clone()
    phone_prev = torch.full((T,), fallback_phone, dtype=torch.long)
    phone_next = torch.full((T,), fallback_phone, dtype=torch.long)
    phone_pos_frac = torch.zeros(T, dtype=torch.float32)
    phone_dur_frames = torch.zeros(T, dtype=torch.float32)
    dist_to_start_f = torch.zeros(T, dtype=torch.float32)
    dist_to_end_f = torch.zeros(T, dtype=torch.float32)
    for si in range(n_seg):
        s, e = seg_starts[si], seg_starts[si + 1]
        seg_len = e - s
        prev_ph = seg_phones[si - 1] if si - 1 >= 0 else fallback_phone
        next_ph = seg_phones[si + 1] if si + 1 < n_seg else fallback_phone
        for t in range(s, e):
            phone_prev[t] = prev_ph
            phone_next[t] = next_ph
            if seg_len == 1:
                phone_pos_frac[t] = 0.5
            else:
                phone_pos_frac[t] = (t - s) / (seg_len - 1)
            phone_dur_frames[t] = float(seg_len)
            dist_to_start_f[t] = float(t - s)
            dist_to_end_f[t] = float(e - 1 - t)
    phone_dur_norm = (torch.log1p(phone_dur_frames) / float(log1p_dur_norm)).clamp(0.0, 1.0)
    dist_start_norm = (dist_to_start_f / 5.0).clamp(0.0, 1.0)
    dist_end_norm = (dist_to_end_f / 5.0).clamp(0.0, 1.0)
    is_boundary = torch.zeros(T, dtype=torch.float32)
    r = int(boundary_radius)
    if r > 0:
        for si in range(n_seg):
            s, e = seg_starts[si], seg_starts[si + 1]
            for k in range(min(r, e - s)):
                v = 1.0 - k / r
                if v > is_boundary[s + k]:
                    is_boundary[s + k] = v
            for k in range(min(r, e - s)):
                v = 1.0 - k / r
                if v > is_boundary[e - 1 - k]:
                    is_boundary[e - 1 - k] = v
    else:
        for si in range(n_seg):
            s = seg_starts[si]; e = seg_starts[si + 1] - 1
            is_boundary[s] = 1.0
            is_boundary[e] = 1.0
    return {
        "phone_cur": phone_cur,
        "phone_prev": phone_prev,
        "phone_next": phone_next,
        "phone_pos_frac": phone_pos_frac,
        "phone_dur_norm": phone_dur_norm,
        "dist_start_norm": dist_start_norm,
        "dist_end_norm": dist_end_norm,
        "is_phone_boundary": is_boundary,
    }


# ---------------------------------------------------------------------------
# timing auxiliary targets + BCE loss
# ---------------------------------------------------------------------------

def build_timing_aux_targets(
    activity_A: torch.Tensor,
    activity_B: torch.Tensor,
    *,
    turn_switch_window_frames: int = 13,
    soft_boundary_width: int = 1,
) -> dict[str, torch.Tensor]:
    a = (activity_A > 0.5).float()
    b = (activity_B > 0.5).float()
    B, T = a.shape
    a_prev = torch.cat([torch.zeros(B, 1, device=a.device), a[:, :-1]], dim=1)
    b_prev = torch.cat([torch.zeros(B, 1, device=b.device), b[:, :-1]], dim=1)
    onset_A = ((a == 1) & (a_prev == 0)).float()
    onset_B = ((b == 1) & (b_prev == 0)).float()
    offset_A = ((a == 0) & (a_prev == 1)).float()
    offset_B = ((b == 0) & (b_prev == 1)).float()
    overlap = a * b
    w_ts = int(turn_switch_window_frames)
    if w_ts > 0:
        kernel = torch.ones(1, 1, w_ts, device=a.device)
        pad = torch.nn.functional.pad(offset_A.unsqueeze(1), (w_ts - 1, 0))
        a_off_recent = torch.nn.functional.conv1d(pad, kernel).squeeze(1).clamp(0, 1)
        pad = torch.nn.functional.pad(offset_B.unsqueeze(1), (w_ts - 1, 0))
        b_off_recent = torch.nn.functional.conv1d(pad, kernel).squeeze(1).clamp(0, 1)
        ts = (onset_B * a_off_recent + onset_A * b_off_recent).clamp(0, 1)
    else:
        ts = torch.zeros_like(a)
    out = {
        "activity_A_tgt": a, "activity_B_tgt": b,
        "onset_A_tgt": onset_A, "onset_B_tgt": onset_B,
        "offset_A_tgt": offset_A, "offset_B_tgt": offset_B,
        "overlap_tgt": overlap, "turn_switch_tgt": ts,
    }
    sw = int(soft_boundary_width)
    if sw > 0:
        kernel = torch.ones(1, 1, 2 * sw + 1, device=a.device)
        for k_ in ("onset_A_tgt", "onset_B_tgt", "offset_A_tgt", "offset_B_tgt", "turn_switch_tgt"):
            x = out[k_].unsqueeze(1)
            x = torch.nn.functional.pad(x, (sw, sw))
            x = torch.nn.functional.conv1d(x, kernel).squeeze(1).clamp(0, 1)
            out[k_] = x
    return out


def compute_timing_aux_loss(
    *,
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    latent_mask: torch.Tensor,
    pos_weights: dict[str, float],
    loss_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total = torch.zeros((), device=latent_mask.device)
    diag: dict[str, torch.Tensor] = {}
    m = latent_mask.float()
    n_valid = m.sum().clamp_min(1.0)
    for tgt_key, target in targets.items():
        head_key = tgt_key.replace("_tgt", "")
        if head_key not in logits:
            continue
        logit = logits[head_key]
        pw = float(pos_weights.get(head_key, 1.0))
        weight = target * (pw - 1.0) + 1.0
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, target, weight=weight, reduction="none",
        )
        per_loss = (bce * m).sum() / n_valid
        w = float(loss_weights.get(head_key, 0.0))
        if w != 0.0:
            total = total + w * per_loss
        diag[f"aux_{head_key}_loss"] = per_loss.detach()
        diag[f"aux_{head_key}_pos_ratio"] = (target * m).sum().detach() / n_valid.detach()
    return total, diag

"""timing predictor loss: L_phone_dur + L_sil_dur + L_any_gap + L_total_len + L_frame_state。

- L_phone_dur: PHONE token のみ の mean CE
- L_sil_dur:   PRE_SIL / FIRST_SIL token の mean CE
- L_any_gap:   utterance first token の cross-channel previous-any-end gap CE
- L_total_len: per-channel 累積長 と GT 累積長 の |Δ|/T
- L_frame_state: per-channel soft activity raster → 4-state probability → 重み付き CE

各 loss は weight 1.0 で和を取るが、 値域は意図的に揃えてある。
overflow loss は 初期実装では入れない。 overflow 量は ログには残す。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kaburi_tts.predictor import (
    N_PHONE_CLASSES, N_SIL_CLASSES, N_GAP_CLASSES,
    SIL_BIN_VALUES, PHONE_BIN_VALUES, GAP_BIN_VALUES,
    TOKEN_TYPE_PHONE, TOKEN_TYPE_PRE_SIL, TOKEN_TYPE_FIRST_SIL,
    N_STATES,
)


def _phone_value_tensor(device) -> torch.Tensor:
    return torch.arange(1, N_PHONE_CLASSES + 1, device=device, dtype=torch.float32)


def _sil_value_tensor(device) -> torch.Tensor:
    return torch.tensor(SIL_BIN_VALUES, device=device, dtype=torch.float32)


def _gap_value_tensor(device) -> torch.Tensor:
    return torch.tensor(GAP_BIN_VALUES, device=device, dtype=torch.float32)


def compute_phone_dur_ce(
    phone_logits: torch.Tensor,        # [B, N, n_phone]
    phone_dur_target: torch.Tensor,    # [B, N] long, -100 for non-PHONE / invalid
) -> torch.Tensor:
    """PHONE token の mean CE (= ignore_index=-100 で 自動的に PHONE のみ)。"""
    return F.cross_entropy(
        phone_logits.float().transpose(1, 2),
        phone_dur_target,
        ignore_index=-100,
        reduction="mean",
    )


def compute_sil_dur_ce(
    sil_logits: torch.Tensor,          # [B, N, n_sil]
    sil_dur_target: torch.Tensor,      # [B, N] long, -100 for non-PRE_SIL / FIRST_SIL
) -> torch.Tensor:
    """PRE_SIL / FIRST_SIL token の mean CE。"""
    return F.cross_entropy(
        sil_logits.float().transpose(1, 2),
        sil_dur_target,
        ignore_index=-100,
        reduction="mean",
    )


def compute_any_gap_ce(
    any_gap_logits: torch.Tensor,       # [B, N, n_gap]
    any_gap_target: torch.Tensor,       # [B, N] long, -100 for non-utt-first
) -> torch.Tensor:
    """utterance first token の cross-channel any-gap CE。"""
    return F.cross_entropy(
        any_gap_logits.float().transpose(1, 2),
        any_gap_target,
        ignore_index=-100,
        reduction="mean",
    )


def compute_expected_durations(
    phone_logits: torch.Tensor,        # [B, N, n_phone]
    sil_logits: torch.Tensor,          # [B, N, n_sil]
    token_type: torch.Tensor,          # [B, N] long
) -> torch.Tensor:
    """各 token の expected duration (= soft sum_c p(c) * value(c))、 token_type に応じて選択。

    Returns:
      expected_dur [B, N] float
    """
    device = phone_logits.device
    phone_values = _phone_value_tensor(device)         # [n_phone]
    sil_values = _sil_value_tensor(device)              # [n_sil]

    phone_probs = torch.softmax(phone_logits.float(), dim=-1)
    sil_probs = torch.softmax(sil_logits.float(), dim=-1)

    e_phone = (phone_probs * phone_values).sum(dim=-1)  # [B, N]
    e_sil = (sil_probs * sil_values).sum(dim=-1)         # [B, N]

    is_phone = token_type == TOKEN_TYPE_PHONE
    is_sil_any = (token_type == TOKEN_TYPE_PRE_SIL) | (token_type == TOKEN_TYPE_FIRST_SIL)

    expected = torch.zeros_like(e_phone)
    expected = torch.where(is_phone, e_phone, expected)
    expected = torch.where(is_sil_any, e_sil, expected)
    return expected


def per_channel_cumsum(
    expected_dur: torch.Tensor,        # [B, N]
    channel_id: torch.Tensor,           # [B, N] long
    attn_mask: torch.Tensor,            # [B, N] bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """各 token の per-channel 累積 start / end frame。

    A の token は A の cumsum、 B の token は B の cumsum を使う。
    Returns:
      start [B, N], end [B, N]
    """
    dur_masked = expected_dur * attn_mask.float()
    mask_A = ((channel_id == 0) & attn_mask).float()
    mask_B = ((channel_id == 1) & attn_mask).float()

    cum_A = (dur_masked * mask_A).cumsum(dim=-1)
    cum_B = (dur_masked * mask_B).cumsum(dim=-1)

    is_A = (channel_id == 0).float()
    end_in_ch = is_A * cum_A + (1.0 - is_A) * cum_B
    start_in_ch = end_in_ch - dur_masked
    return start_in_ch, end_in_ch


def build_soft_activity(
    start_in_ch: torch.Tensor,         # [B, N]
    end_in_ch: torch.Tensor,            # [B, N]
    token_type: torch.Tensor,           # [B, N] long
    channel_id: torch.Tensor,           # [B, N] long
    attn_mask: torch.Tensor,            # [B, N] bool
    T: int,
    tau: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """soft activity raster (= per-frame [0,1])、 PHONE token のみ contribution。

    activity_k(t) = sigmoid((t - start_k) / tau) * sigmoid((end_k - t) / tau)
    A_active[t] = clamp(sum over PHONE tokens on ch A, 0, 1)
    Returns:
      A_active [B, T], B_active [B, T]
    """
    device = start_in_ch.device
    B, N = start_in_ch.shape

    is_phone = ((token_type == TOKEN_TYPE_PHONE) & attn_mask).float()  # [B, N]
    mask_A_phone = is_phone * (channel_id == 0).float()
    mask_B_phone = is_phone * (channel_id == 1).float()

    t_idx = torch.arange(T, device=device, dtype=start_in_ch.dtype).view(1, T, 1)  # [1, T, 1]
    s = start_in_ch.unsqueeze(1)   # [B, 1, N]
    e = end_in_ch.unsqueeze(1)

    activity = torch.sigmoid((t_idx - s) / tau) * torch.sigmoid((e - t_idx) / tau)  # [B, T, N]

    A_act = (activity * mask_A_phone.unsqueeze(1)).sum(dim=-1).clamp(0.0, 1.0)
    B_act = (activity * mask_B_phone.unsqueeze(1)).sum(dim=-1).clamp(0.0, 1.0)
    return A_act, B_act


def compute_frame_state_ce(
    A_act: torch.Tensor,                # [B, T] float in [0,1]
    B_act: torch.Tensor,                # [B, T]
    gt_state_label: torch.Tensor,        # [B, T] long {0..3}
    class_weight: torch.Tensor,          # [4] float
    prob_clamp_min: float = 1e-2,
    frame_mask: torch.Tensor | None = None,  # [B, T] bool
) -> tuple[torch.Tensor, dict]:
    """4-state probability + weighted CE。

    指示書 §5: 各 loss を 同 scale に揃える。
    prob_clamp_min=1e-2 で per-frame CE を max -log(1e-2)≈4.6 に cap、
    init で L_phone_dur (= ~log(30)≈3.4) と 同程度の scale に。

    Returns:
      loss (scalar)
      info dict: per-class CE 等 (= logging 用)
    """
    p_none = (1.0 - A_act) * (1.0 - B_act)
    p_a = A_act * (1.0 - B_act)
    p_b = (1.0 - A_act) * B_act
    p_both = A_act * B_act
    probs = torch.stack([p_none, p_a, p_b, p_both], dim=-1).clamp_min(prob_clamp_min)  # [B, T, 4]
    log_probs = torch.log(probs)

    nll = -log_probs.gather(2, gt_state_label.unsqueeze(-1)).squeeze(-1)  # [B, T]
    w_per_frame = class_weight.to(nll.device)[gt_state_label]              # [B, T]
    weighted = nll * w_per_frame
    if frame_mask is not None:
        mask = frame_mask.to(weighted.device).bool()
        if mask.any():
            loss = weighted[mask].mean()
        else:
            loss = weighted.mean() * 0.0
    else:
        mask = None
        loss = weighted.mean()

    # per-class CE for logging
    info = {}
    with torch.no_grad():
        for c in range(N_STATES):
            mask_c = (gt_state_label == c)
            if mask is not None:
                mask_c = mask_c & mask
            if mask_c.any():
                info[f"ce_class_{c}"] = float(nll[mask_c].mean().item())
            else:
                info[f"ce_class_{c}"] = float("nan")
            if mask is not None and mask.any():
                info[f"pred_frac_class_{c}"] = float(((probs.argmax(-1) == c) & mask).float().sum().item() / mask.float().sum().item())
                info[f"gt_frac_class_{c}"] = float(mask_c.float().sum().item() / mask.float().sum().item())
            else:
                info[f"pred_frac_class_{c}"] = float((probs.argmax(-1) == c).float().mean().item())
                info[f"gt_frac_class_{c}"] = float(mask_c.float().mean().item())

    return loss, info


def compute_total_len_loss(
    expected_dur: torch.Tensor,        # [B, N]
    channel_id: torch.Tensor,
    attn_mask: torch.Tensor,
    gt_total_A: torch.Tensor,           # [B] long
    gt_total_B: torch.Tensor,           # [B] long
    T: int,
) -> tuple[torch.Tensor, dict]:
    """L_total_len = (|pred_A - gt_A| + |pred_B - gt_B|) / (2*T)、 batch mean。"""
    dur_masked = expected_dur * attn_mask.float()
    mask_A = ((channel_id == 0) & attn_mask).float()
    mask_B = ((channel_id == 1) & attn_mask).float()
    pred_A = (dur_masked * mask_A).sum(dim=-1)  # [B]
    pred_B = (dur_masked * mask_B).sum(dim=-1)  # [B]

    err_A = (pred_A - gt_total_A.float()).abs()
    err_B = (pred_B - gt_total_B.float()).abs()
    loss = ((err_A + err_B) / (2.0 * T)).mean()

    info = {
        "pred_total_A_mean": float(pred_A.mean().item()),
        "pred_total_B_mean": float(pred_B.mean().item()),
        "gt_total_A_mean": float(gt_total_A.float().mean().item()),
        "gt_total_B_mean": float(gt_total_B.float().mean().item()),
        "err_A_mean": float(err_A.mean().item()),
        "err_B_mean": float(err_B.mean().item()),
        "overflow_A_count": int(((pred_A - T).clamp_min(0) > 0).sum().item()),
        "overflow_B_count": int(((pred_B - T).clamp_min(0) > 0).sum().item()),
        "overflow_A_mean_frames": float((pred_A - T).clamp_min(0).mean().item()),
        "overflow_B_mean_frames": float((pred_B - T).clamp_min(0).mean().item()),
    }
    return loss, info


def compute_utt_len_loss(
    expected_dur: torch.Tensor,        # [B, N]
    token_type: torch.Tensor,
    channel_id: torch.Tensor,
    utt_index: torch.Tensor,
    is_utt_first_token: torch.Tensor,
    attn_mask: torch.Tensor,
    gt_utt_region_len: torch.Tensor,   # [B, N] float, same value on each utt token
    T: int,
) -> tuple[torch.Tensor, dict]:
    """Utterance-level phone span loss.

    L_total_len only constrains per-channel totals. This loss makes each
    utterance's summed PHONE duration match its GT region length so dialog text
    context can affect speaking-rate variation within the existing raster format.
    """
    losses = []
    pred_vals = []
    gt_vals = []
    B, _N = token_type.shape
    first_mask = attn_mask & is_utt_first_token.bool()
    phone_mask = attn_mask & (token_type == TOKEN_TYPE_PHONE)
    for b in range(B):
        first_pos = first_mask[b].nonzero(as_tuple=False).squeeze(-1)
        for fp_t in first_pos:
            fp = int(fp_t.item())
            same_utt_phone = (
                phone_mask[b]
                & (channel_id[b] == channel_id[b, fp])
                & (utt_index[b] == utt_index[b, fp])
            )
            if not same_utt_phone.any():
                continue
            pred = expected_dur[b, same_utt_phone].sum()
            gt = gt_utt_region_len[b, fp].float().clamp_min(1.0)
            losses.append((pred - gt).abs() / float(T))
            pred_vals.append(pred.detach())
            gt_vals.append(gt.detach())
    if not losses:
        zero = expected_dur.sum() * 0.0
        return zero, {
            "pred_utt_len_mean": 0.0,
            "gt_utt_len_mean": 0.0,
            "err_utt_len_mean": 0.0,
            "n_utts": 0,
        }
    loss = torch.stack(losses).mean()
    pred_t = torch.stack(pred_vals)
    gt_t = torch.stack(gt_vals)
    err_t = (pred_t - gt_t).abs()
    return loss, {
        "pred_utt_len_mean": float(pred_t.mean().item()),
        "gt_utt_len_mean": float(gt_t.mean().item()),
        "err_utt_len_mean": float(err_t.mean().item()),
        "n_utts": int(pred_t.numel()),
    }


def compute_utt_speed_loss(
    expected_dur: torch.Tensor,        # [B, N]
    token_type: torch.Tensor,
    channel_id: torch.Tensor,
    utt_index: torch.Tensor,
    is_utt_first_token: torch.Tensor,
    attn_mask: torch.Tensor,
    gt_utt_region_len: torch.Tensor,   # [B, N]
    *,
    norm_frames_per_phone: float = 10.0,
) -> tuple[torch.Tensor, dict]:
    """Utterance-level speaking-rate loss.

    L_utt_len constrains absolute utterance length.  This term constrains
    frames/phone so short backchannels and long explanatory turns keep their
    relative speed differences instead of drifting toward a flat average.
    """
    losses = []
    pred_speeds = []
    gt_speeds = []
    B, _N = token_type.shape
    first_mask = attn_mask & is_utt_first_token.bool()
    phone_mask = attn_mask & (token_type == TOKEN_TYPE_PHONE)
    for b in range(B):
        first_pos = first_mask[b].nonzero(as_tuple=False).squeeze(-1)
        for fp_t in first_pos:
            fp = int(fp_t.item())
            same_utt_phone = (
                phone_mask[b]
                & (channel_id[b] == channel_id[b, fp])
                & (utt_index[b] == utt_index[b, fp])
            )
            n_phone = int(same_utt_phone.sum().item())
            if n_phone <= 0:
                continue
            pred = expected_dur[b, same_utt_phone].sum() / float(n_phone)
            gt = gt_utt_region_len[b, fp].float().clamp_min(1.0) / float(n_phone)
            losses.append((pred - gt).abs() / float(norm_frames_per_phone))
            pred_speeds.append(pred.detach())
            gt_speeds.append(gt.detach())
    if not losses:
        zero = expected_dur.sum() * 0.0
        return zero, {
            "pred_utt_speed_mean": 0.0,
            "gt_utt_speed_mean": 0.0,
            "err_utt_speed_mean": 0.0,
        }
    loss = torch.stack(losses).mean()
    pred_t = torch.stack(pred_speeds)
    gt_t = torch.stack(gt_speeds)
    err_t = (pred_t - gt_t).abs()
    return loss, {
        "pred_utt_speed_mean": float(pred_t.mean().item()),
        "gt_utt_speed_mean": float(gt_t.mean().item()),
        "err_utt_speed_mean": float(err_t.mean().item()),
    }


def compute_gap_l1_loss(
    any_gap_logits: torch.Tensor,       # [B, N, n_gap]
    any_gap_gt: torch.Tensor,           # [B, N]
    any_gap_target: torch.Tensor,       # [B, N], -100 invalid
    attn_mask: torch.Tensor,
    *,
    norm_frames: float = 100.0,
) -> tuple[torch.Tensor, dict]:
    """Expected any-gap L1 loss.

    CE learns the gap bin classification.  This term gives a smoother penalty
    for over/under-shooting pauses and overlaps, which matters for gb=0.55
    inference where expected gaps are used.
    """
    mask = (any_gap_target >= 0) & attn_mask
    if not mask.any():
        zero = any_gap_logits.sum() * 0.0
        return zero, {"gap_l1_mae_frames": 0.0}
    gap_values = _gap_value_tensor(any_gap_logits.device)
    gap_probs = torch.softmax(any_gap_logits.float(), dim=-1)
    e_gap = (gap_probs * gap_values).sum(dim=-1)
    err = (e_gap - any_gap_gt.float()).abs()[mask]
    return (err / float(norm_frames)).mean(), {
        "gap_l1_mae_frames": float(err.mean().item()),
    }


class TimingPredictorLoss(nn.Module):
    """timing predictor loss module。 forward は (logits, batch) を取り、 (total_loss, dict) を返す。

    Args:
      lambda_phone, lambda_sil, lambda_total_len, lambda_frame_state: 重み (= 全 1.0 デフォルト)
      lambda_any_gap: turn-taking gap CE の重み
      lambda_utt_len: utterance-level region length loss の重み
      tau: soft boundary の温度 (= 2.0 デフォルト)
      class_weight: [4] 4-state class weight tensor
      T: chunk frame 長 (= 750)
    """

    def __init__(
        self,
        *,
        T: int = 750,
        tau: float = 2.0,
        lambda_phone: float = 1.0,
        lambda_sil: float = 1.0,
        lambda_any_gap: float = 1.0,
        lambda_total_len: float = 1.0,
        lambda_frame_state: float = 1.0,
        lambda_utt_len: float = 0.0,
        lambda_utt_speed: float = 0.0,
        lambda_gap_l1: float = 0.0,
        lambda_frame_tail: float = 0.0,
        tail_start_frac: float = 0.55,
        gap_l1_norm_frames: float = 100.0,
        utt_speed_norm_frames: float = 10.0,
        class_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.T = T
        self.tau = tau
        self.lambda_phone = lambda_phone
        self.lambda_sil = lambda_sil
        self.lambda_any_gap = lambda_any_gap
        self.lambda_total_len = lambda_total_len
        self.lambda_frame_state = lambda_frame_state
        self.lambda_utt_len = lambda_utt_len
        self.lambda_utt_speed = lambda_utt_speed
        self.lambda_gap_l1 = lambda_gap_l1
        self.lambda_frame_tail = lambda_frame_tail
        self.tail_start_frac = tail_start_frac
        self.gap_l1_norm_frames = gap_l1_norm_frames
        self.utt_speed_norm_frames = utt_speed_norm_frames
        if class_weight is None:
            class_weight = torch.ones(N_STATES, dtype=torch.float32)
        self.register_buffer("class_weight", class_weight.float())

    def forward(
        self,
        phone_logits: torch.Tensor,    # [B, N, n_phone]
        sil_logits: torch.Tensor,       # [B, N, n_sil]
        any_gap_logits: torch.Tensor,   # [B, N, n_gap]
        batch: dict,
    ) -> tuple[torch.Tensor, dict]:
        token_type = batch["timing_token_type"]
        channel_id = batch["timing_channel_id"]
        attn_mask = batch["timing_mask"]
        phone_target = batch["timing_phone_dur_target"]
        sil_target = batch["timing_sil_dur_target"]
        any_gap_target = batch["timing_any_gap_target"]
        gt_state = batch["timing_frame_state_label"]
        gt_A = batch["timing_gt_total_len_A"]
        gt_B = batch["timing_gt_total_len_B"]

        L_phone = compute_phone_dur_ce(phone_logits, phone_target)
        L_sil = compute_sil_dur_ce(sil_logits, sil_target)
        L_any_gap = compute_any_gap_ce(any_gap_logits, any_gap_target)

        # expected duration (= L_total_len と soft activity 両方で使う)
        e_dur = compute_expected_durations(phone_logits, sil_logits, token_type)

        L_total_len, total_len_info = compute_total_len_loss(
            e_dur, channel_id, attn_mask, gt_A, gt_B, self.T,
        )
        if (
            self.lambda_utt_len != 0.0
            and "timing_utt_region_len_gt" in batch
            and "timing_is_utt_first_token" in batch
            and "timing_utt_index" in batch
        ):
            L_utt_len, utt_len_info = compute_utt_len_loss(
                e_dur, token_type, channel_id, batch["timing_utt_index"],
                batch["timing_is_utt_first_token"], attn_mask,
                batch["timing_utt_region_len_gt"], self.T,
            )
        else:
            L_utt_len = e_dur.sum() * 0.0
            utt_len_info = {
                "pred_utt_len_mean": 0.0,
                "gt_utt_len_mean": 0.0,
                "err_utt_len_mean": 0.0,
                "n_utts": 0,
            }

        if (
            self.lambda_utt_speed != 0.0
            and "timing_utt_region_len_gt" in batch
            and "timing_is_utt_first_token" in batch
            and "timing_utt_index" in batch
        ):
            L_utt_speed, utt_speed_info = compute_utt_speed_loss(
                e_dur, token_type, channel_id, batch["timing_utt_index"],
                batch["timing_is_utt_first_token"], attn_mask,
                batch["timing_utt_region_len_gt"],
                norm_frames_per_phone=self.utt_speed_norm_frames,
            )
        else:
            L_utt_speed = e_dur.sum() * 0.0
            utt_speed_info = {
                "pred_utt_speed_mean": 0.0,
                "gt_utt_speed_mean": 0.0,
                "err_utt_speed_mean": 0.0,
            }

        if self.lambda_gap_l1 != 0.0 and "timing_any_gap_gt" in batch:
            L_gap_l1, gap_l1_info = compute_gap_l1_loss(
                any_gap_logits, batch["timing_any_gap_gt"], any_gap_target, attn_mask,
                norm_frames=self.gap_l1_norm_frames,
            )
        else:
            L_gap_l1 = e_dur.sum() * 0.0
            gap_l1_info = {"gap_l1_mae_frames": 0.0}

        start_ch, end_ch = per_channel_cumsum(e_dur, channel_id, attn_mask)
        A_act, B_act = build_soft_activity(
            start_ch, end_ch, token_type, channel_id, attn_mask, self.T, tau=self.tau,
        )
        L_frame, frame_info = compute_frame_state_ce(
            A_act, B_act, gt_state, self.class_weight,
        )
        if self.lambda_frame_tail != 0.0:
            B = gt_state.shape[0]
            t0 = int(round(float(self.T) * float(self.tail_start_frac)))
            frame_mask = torch.zeros(B, self.T, dtype=torch.bool, device=gt_state.device)
            frame_mask[:, max(0, min(self.T, t0)):] = True
            L_frame_tail, frame_tail_info = compute_frame_state_ce(
                A_act, B_act, gt_state, self.class_weight, frame_mask=frame_mask,
            )
        else:
            L_frame_tail = e_dur.sum() * 0.0
            frame_tail_info = {}

        L_total = (
            self.lambda_phone * L_phone
            + self.lambda_sil * L_sil
            + self.lambda_any_gap * L_any_gap
            + self.lambda_gap_l1 * L_gap_l1
            + self.lambda_total_len * L_total_len
            + self.lambda_utt_len * L_utt_len
            + self.lambda_utt_speed * L_utt_speed
            + self.lambda_frame_state * L_frame
            + self.lambda_frame_tail * L_frame_tail
        )

        info = {
            "L_phone_dur": float(L_phone.item()),
            "L_sil_dur": float(L_sil.item()),
            "L_any_gap": float(L_any_gap.item()),
            "L_gap_l1": float(L_gap_l1.item()),
            "L_total_len": float(L_total_len.item()),
            "L_utt_len": float(L_utt_len.item()),
            "L_utt_speed": float(L_utt_speed.item()),
            "L_frame_state": float(L_frame.item()),
            "L_frame_tail": float(L_frame_tail.item()),
            "L_total": float(L_total.item()),
            **{f"total_len/{k}": v for k, v in total_len_info.items()},
            **{f"utt_len/{k}": v for k, v in utt_len_info.items()},
            **{f"utt_speed/{k}": v for k, v in utt_speed_info.items()},
            **{f"gap_l1/{k}": v for k, v in gap_l1_info.items()},
            **{f"frame_state/{k}": v for k, v in frame_info.items()},
            **{f"frame_tail/{k}": v for k, v in frame_tail_info.items()},
        }
        # phone / sil token 数 (= for sanity check)
        with torch.no_grad():
            info["n_phone_tokens"] = int(((token_type == TOKEN_TYPE_PHONE) & attn_mask).sum().item())
            info["n_sil_tokens"] = int(((token_type == TOKEN_TYPE_PRE_SIL) & attn_mask).sum().item())
            info["n_first_sil_tokens"] = int(((token_type == TOKEN_TYPE_FIRST_SIL) & attn_mask).sum().item())
            info["n_any_gap_tokens"] = int(((batch["timing_any_gap_target"] >= 0) & attn_mask).sum().item())
            # dur MAE proxy
            phone_mask_b = (token_type == TOKEN_TYPE_PHONE) & attn_mask & (batch["timing_phone_dur_gt"] > 0)
            sil_mask_b = ((token_type == TOKEN_TYPE_PRE_SIL) | (token_type == TOKEN_TYPE_FIRST_SIL)) & attn_mask
            gap_mask_b = (batch["timing_any_gap_target"] >= 0) & attn_mask
            if phone_mask_b.any():
                info["phone_dur_mae"] = float((e_dur - batch["timing_phone_dur_gt"]).abs()[phone_mask_b].mean().item())
            if sil_mask_b.any():
                info["sil_dur_mae"] = float((e_dur - batch["timing_sil_dur_gt"]).abs()[sil_mask_b].mean().item())
            if gap_mask_b.any():
                gap_values = _gap_value_tensor(any_gap_logits.device)
                gap_probs = torch.softmax(any_gap_logits.float(), dim=-1)
                e_gap = (gap_probs * gap_values).sum(dim=-1)
                pred_gap = gap_probs.argmax(dim=-1)
                info["any_gap_mae"] = float((e_gap - batch["timing_any_gap_gt"]).abs()[gap_mask_b].mean().item())
                info["any_gap_argmax_acc"] = float((pred_gap[gap_mask_b] == batch["timing_any_gap_target"][gap_mask_b]).float().mean().item())

        return L_total, info


__all__ = [
    "TimingPredictorLoss",
    "compute_phone_dur_ce",
    "compute_sil_dur_ce",
    "compute_any_gap_ce",
    "compute_expected_durations",
    "per_channel_cumsum",
    "build_soft_activity",
    "compute_frame_state_ce",
    "compute_total_len_loss",
    "compute_utt_len_loss",
    "compute_utt_speed_loss",
    "compute_gap_l1_loss",
]

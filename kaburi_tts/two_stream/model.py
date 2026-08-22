"""
Irodori 1ch two-stream wrapper.

設計方針:
  - A/B を batch 次元で stack して 1 model 呼び出しで同時 forward
  - model 内部 (self-attention 等) で A と B は別 sample として扱われ、 直接混ざらない
  - in_proj / out_proj は Irodori 原型の 32-dim latent shape を保持
  - 新 module (activity_proj, phone_emb, phone_proj, phone_temporal_conv) は per-stream 1ch 用 shape
  - speaker_encoder は Irodori frozen を A/B でそれぞれ呼び、 stream に対応する speaker_state を渡す
  - text_encoder は Irodori frozen、 A/B で別々の text_input_ids (= speaker marker) を渡す

forward 入力:
  latent_A_noisy, latent_B_noisy: (B, T, 32) 各 stream
  t: (B,) RF timestep (A と B 共通)
  phone_A, phone_B: (B, T) long
  activity_A, activity_B: (B, T) float ∈ {0, 1}
  ref_latent_A, ref_latent_B: (B, T_ref, 32)
  ref_mask_A, ref_mask_B: (B, T_ref) bool
  text_A_input_ids, text_A_mask: (B, L_A)
  text_B_input_ids, text_B_mask: (B, L_B)

forward 出力:
  v_pred_A, v_pred_B: (B, T, 32)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from irodori_tts.model import TextToLatentRFDiT


# ---------------------------------------------------------------------------
# SoftPhoneConditioner (phone-soft / boundary-aware condition)
# ---------------------------------------------------------------------------

class SoftPhoneConditioner(nn.Module):
    """1 frame に対し phone_cur / prev / next embedding + class embedding + 5 scalar features を統合し、
    model_dim 空間に projection。 last layer zero-init で訓練初期は legacy 挙動を保つ。"""

    def __init__(
        self,
        *,
        n_phones: int,
        n_phone_classes: int = 8,
        model_dim: int = 1280,
        phone_emb_dim: int = 128,
        class_emb_dim: int = 32,
        scalar_dim: int = 5,
        hidden_dim: int = 512,
        temporal_conv_kernel: int = 5,
        use_temporal_conv: bool = True,
        zero_init_last: bool = True,
    ) -> None:
        super().__init__()
        self.n_phones = int(n_phones)
        self.phone_emb_dim = int(phone_emb_dim)
        self.class_emb_dim = int(class_emb_dim)
        self.scalar_dim = int(scalar_dim)
        self.phone_emb = nn.Embedding(self.n_phones, self.phone_emb_dim, padding_idx=0)
        nn.init.normal_(self.phone_emb.weight, std=0.02)
        self.class_emb = nn.Embedding(int(n_phone_classes), self.class_emb_dim)
        nn.init.normal_(self.class_emb.weight, std=0.02)
        self.scalar_proj = nn.Sequential(
            nn.Linear(self.scalar_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        in_dim = self.phone_emb_dim * 3 + self.class_emb_dim + 64
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, model_dim, bias=False),
        )
        if zero_init_last:
            with torch.no_grad():
                self.proj[-1].weight.zero_()
        if use_temporal_conv:
            k = int(temporal_conv_kernel)
            assert k % 2 == 1, "temporal_conv_kernel must be odd"
            self.temporal_conv = nn.Conv1d(
                model_dim, model_dim, kernel_size=k,
                padding=k // 2, groups=model_dim, bias=False,
            )
            with torch.no_grad():
                self.temporal_conv.weight.zero_()
                self.temporal_conv.weight[:, :, k // 2] = 1.0    # delta-init = identity
        else:
            self.temporal_conv = None

    def forward(
        self,
        phone_cur: torch.Tensor,      # (B, T) long
        phone_prev: torch.Tensor,     # (B, T) long
        phone_next: torch.Tensor,     # (B, T) long
        phone_class: torch.Tensor,    # (B, T) long
        scalars: torch.Tensor,        # (B, T, 5) float
    ) -> torch.Tensor:
        """(B, T, model_dim) を返す。"""
        emb_cur = self.phone_emb(phone_cur)
        emb_prev = self.phone_emb(phone_prev)
        emb_next = self.phone_emb(phone_next)
        cls = self.class_emb(phone_class)
        target_dtype = self.scalar_proj[0].weight.dtype
        sc = self.scalar_proj(scalars.to(dtype=target_dtype))
        feat = torch.cat([emb_cur, emb_prev, emb_next, cls, sc], dim=-1)
        out = self.proj(feat.to(dtype=target_dtype))
        if self.temporal_conv is not None:
            out = self.temporal_conv(out.transpose(1, 2)).transpose(1, 2)
        return out


# ---------------------------------------------------------------------------
# TimingAuxHead (timing 補助 loss head)
# ---------------------------------------------------------------------------

class TimingAuxHead(nn.Module):
    """final hidden state (B, T, model_dim) から activity/onset/offset/overlap/turn_switch logit を予測。
    A/B は per-stream で stream_head を共有、 overlap/turn_switch は pair_head。"""

    def __init__(self, model_dim: int = 1280) -> None:
        super().__init__()
        h = max(model_dim // 4, 64)
        self.stream_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, h),
            nn.SiLU(),
            nn.Linear(h, 3),    # activity, onset, offset
        )
        self.pair_head = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, h),
            nn.SiLU(),
            nn.Linear(h, 2),    # overlap, turn_switch
        )

    def forward(
        self,
        hidden_A: torch.Tensor,      # (B, T, D)
        hidden_B: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sh_A = self.stream_head(hidden_A)    # (B, T, 3)
        sh_B = self.stream_head(hidden_B)
        ph = self.pair_head(torch.cat([hidden_A, hidden_B], dim=-1))    # (B, T, 2)
        return {
            "activity_A": sh_A[..., 0],
            "onset_A":    sh_A[..., 1],
            "offset_A":   sh_A[..., 2],
            "activity_B": sh_B[..., 0],
            "onset_B":    sh_B[..., 1],
            "offset_B":   sh_B[..., 2],
            "overlap":      ph[..., 0],
            "turn_switch":  ph[..., 1],
        }


def compute_interaction_state(activity_A: torch.Tensor, activity_B: torch.Tensor) -> torch.Tensor:
    """activity_A, activity_B: (B, T) ∈ {0, 1} float → state (B, T) long ∈ [0, 9]。

    state codes:
      0: silence (both inactive)
      1: A_only (A=1, B=0)
      2: B_only (A=0, B=1)
      3: overlap (both active)
      4: A_onset (A frame で 0→1 transition)
      5: B_onset (B 同様)
      6: A_offset (A: 1→0)
      7: B_offset (B: 1→0)
      8: A_to_B_switch (A_offset と B_onset が同 frame)
      9: B_to_A_switch (B_offset と A_onset が同 frame)

    境界 frame は base state (0-3) を上書きする。 switch > onset/offset > base state の優先順。
    """
    a = (activity_A > 0.5).long()   # (B, T)
    b = (activity_B > 0.5).long()
    # Base state: 0:silence, 1:A_only, 2:B_only, 3:overlap
    state = a * 1 + b * 2

    # Previous frame (left-pad with 0)
    a_prev = torch.cat([torch.zeros_like(a[:, :1]), a[:, :-1]], dim=1)
    b_prev = torch.cat([torch.zeros_like(b[:, :1]), b[:, :-1]], dim=1)
    a_onset = (a_prev == 0) & (a == 1)
    b_onset = (b_prev == 0) & (b == 1)
    a_offset = (a_prev == 1) & (a == 0)
    b_offset = (b_prev == 1) & (b == 0)
    a_to_b = a_offset & b_onset    # 両者同 frame で switch (A 切→B 開)
    b_to_a = b_offset & a_onset

    # 優先順 (低→高で重ね、 高優先が後勝ち):
    state = torch.where(a_onset & ~a_to_b, torch.full_like(state, 4), state)
    state = torch.where(b_onset & ~b_to_a, torch.full_like(state, 5), state)
    state = torch.where(a_offset & ~a_to_b, torch.full_like(state, 6), state)
    state = torch.where(b_offset & ~b_to_a, torch.full_like(state, 7), state)
    state = torch.where(a_to_b, torch.full_like(state, 8), state)
    state = torch.where(b_to_a, torch.full_like(state, 9), state)
    return state


class Irodori1chTwoStreamModel(nn.Module):
    """A/B を独立 stream として処理し、 内部で batch dim stack して 1 model 呼び出しで同時 forward。

    base config はそのまま (latent_dim=32、 latent_patch_size=1)。 Irodori の in_proj 32→1280、
    out_proj 1280→32 は **置換せず原型を維持** する。

    新 module はすべて per-stream 1ch shape:
      activity_proj: Linear(1, model_dim) zero-init
      phone_emb:     Embedding(V, phone_emb_dim) normal init
      phone_proj:    Linear or 2-layer MLP (phone_emb_dim → model_dim)、 last layer zero-init
      phone_temporal_conv: depthwise Conv1d (delta-init)、 optional
    """

    def __init__(
        self,
        base: TextToLatentRFDiT,
        *,
        phone_vocab_size: int,
        phone_emb_dim: int = 128,
        use_activity: bool = True,
        use_phone: bool = True,
        phone_proj_hidden_dim: int | None = None,    # None = 1-layer Linear、 int = 2-layer MLP
        use_phone_temporal_conv: bool = False,
        phone_temporal_conv_kernel: int = 5,
        use_interaction_state: bool = False,         # activity_A/B から 10-class state
        interaction_state_vocab: int = 10,
        text_scale: float = 1.0,                     # text_state magnitude scale (1.0 = legacy)
        phone_condition_mode: str = "hard",          # "hard" (legacy) or "soft"
        soft_phone_class_size: int = 8,              # soft mode
        soft_phone_class_emb_dim: int = 32,
        soft_phone_hidden_dim: int = 512,
        soft_phone_temporal_conv_kernel: int = 5,
        soft_phone_zero_init_last: bool = True,
        use_timing_aux_head: bool = False,
    ) -> None:
        super().__init__()
        self.base = base
        self.cfg = base.cfg
        self.use_activity = use_activity
        self.use_phone = use_phone
        self.phone_vocab_size = int(phone_vocab_size)
        self.phone_emb_dim = int(phone_emb_dim)

        model_dim = self.cfg.model_dim  # 1280

        # ---- in_proj / out_proj は Irodori 原型維持 (置換しない) ----

        # ---- activity_proj (per-stream, 1-channel) ----
        if use_activity:
            self.activity_proj = nn.Linear(1, model_dim, bias=False)
            nn.init.zeros_(self.activity_proj.weight)
        else:
            self.activity_proj = None

        # ---- phone path (per-stream, 1-channel) ----
        if use_phone:
            self.phone_emb = nn.Embedding(self.phone_vocab_size, self.phone_emb_dim, padding_idx=0)
            nn.init.normal_(self.phone_emb.weight, std=0.02)

            if use_phone_temporal_conv:
                k = int(phone_temporal_conv_kernel)
                assert k % 2 == 1, "phone_temporal_conv_kernel must be odd"
                self.phone_temporal_conv = nn.Conv1d(
                    self.phone_emb_dim, self.phone_emb_dim,
                    kernel_size=k, padding=k // 2,
                    groups=self.phone_emb_dim, bias=False,    # depthwise
                )
                with torch.no_grad():
                    self.phone_temporal_conv.weight.zero_()
                    self.phone_temporal_conv.weight[:, :, k // 2] = 1.0  # delta = identity
            else:
                self.phone_temporal_conv = None

            # phone_proj: legacy 1-layer or 2-layer MLP
            if phone_proj_hidden_dim is None:
                self.phone_proj = nn.Linear(self.phone_emb_dim, model_dim, bias=False)
                nn.init.zeros_(self.phone_proj.weight)
            else:
                hidden = int(phone_proj_hidden_dim)
                self.phone_proj = nn.Sequential(
                    nn.Linear(self.phone_emb_dim, hidden),
                    nn.SiLU(),
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, model_dim, bias=False),
                )
                with torch.no_grad():
                    self.phone_proj[-1].weight.zero_()
        else:
            self.phone_emb = None
            self.phone_temporal_conv = None
            self.phone_proj = None

        # ---- interaction_state: activity_A/B → 10-class context embedding ----
        # 0:silence, 1:A_only, 2:B_only, 3:overlap, 4:A_onset, 5:B_onset,
        # 6:A_offset, 7:B_offset, 8:A_to_B_switch, 9:B_to_A_switch
        self.use_interaction_state = use_interaction_state
        if use_interaction_state:
            self.interaction_state_emb = nn.Embedding(int(interaction_state_vocab), model_dim)
            nn.init.zeros_(self.interaction_state_emb.weight)   # zero-init: 学習で立ち上がる
        else:
            self.interaction_state_emb = None

        # ---- text_scale: text_state magnitude scaling、 1.0 = legacy ----
        self.text_scale = float(text_scale)

        # ---- soft phone conditioner (= 5 scalar + prev/cur/next phone + class) ----
        self.phone_condition_mode = str(phone_condition_mode)
        if self.phone_condition_mode == "soft":
            assert use_phone, "soft phone conditioner requires use_phone=True"
            self.soft_phone_conditioner = SoftPhoneConditioner(
                n_phones=self.phone_vocab_size,
                n_phone_classes=int(soft_phone_class_size),
                model_dim=model_dim,
                phone_emb_dim=self.phone_emb_dim,
                class_emb_dim=int(soft_phone_class_emb_dim),
                scalar_dim=5,
                hidden_dim=int(soft_phone_hidden_dim),
                temporal_conv_kernel=int(soft_phone_temporal_conv_kernel),
                use_temporal_conv=True,
                zero_init_last=bool(soft_phone_zero_init_last),
            )
        else:
            self.soft_phone_conditioner = None

        # ---- timing aux head (= activity/onset/offset/overlap/turn_switch logits) ----
        self.use_timing_aux_head = bool(use_timing_aux_head)
        if self.use_timing_aux_head:
            self.timing_aux_head = TimingAuxHead(model_dim=model_dim)
        else:
            self.timing_aux_head = None

    # ------------------------------------------------------------------
    # property pass-through
    # ------------------------------------------------------------------
    @property
    def text_encoder(self):
        return self.base.text_encoder

    @property
    def speaker_encoder(self):
        return self.base.speaker_encoder

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        latent_A: torch.Tensor,                  # (B, T, 32)
        latent_B: torch.Tensor,                  # (B, T, 32)
        t: torch.Tensor,                         # (B,)
        *,
        phone_A: torch.Tensor,                   # (B, T) long
        phone_B: torch.Tensor,
        activity_A: torch.Tensor,                # (B, T) float ∈ {0,1}
        activity_B: torch.Tensor,
        ref_latent_A: torch.Tensor,              # (B, T_ref, 32)
        ref_mask_A: torch.Tensor,
        ref_latent_B: torch.Tensor,
        ref_mask_B: torch.Tensor,
        text_A_input_ids: torch.Tensor,          # (B, L)
        text_A_mask: torch.Tensor,
        text_B_input_ids: torch.Tensor,
        text_B_mask: torch.Tensor,
        latent_mask: torch.Tensor | None = None,
        # soft phone inputs (= dict of tensors per stream; mode=soft 時に使用)
        soft_phone_A: dict[str, torch.Tensor] | None = None,
        soft_phone_B: dict[str, torch.Tensor] | None = None,
        return_timing_aux_logits: bool = False,
    ):
        """A/B を batch dim で stack → 1 forward → split back。

        拡張:
          phone_condition_mode=="soft" の場合、 phone bias は SoftPhoneConditioner で
          soft_phone_A/B 内の tensor を用いて計算。 hard phone path は使わない。
          return_timing_aux_logits=True の場合、 (v_A, v_B, aux_logits) を返す。
        """
        from irodori_tts.model import get_timestep_embedding

        B = latent_A.shape[0]

        # ---- batch dim で stack (1 forward で 2*B sample 処理) ----
        latent_stk = torch.cat([latent_A, latent_B], dim=0)              # (2B, T, 32)
        phone_stk = torch.cat([phone_A, phone_B], dim=0)                  # (2B, T)
        activity_stk = torch.cat([activity_A, activity_B], dim=0).unsqueeze(-1)  # (2B, T, 1)
        # ref_A と ref_B は T_ref が違う可能性 (per-speaker 固定 ref の長さ違い) → 共通最大に右 pad
        T_ref_max = max(ref_latent_A.shape[1], ref_latent_B.shape[1])
        if ref_latent_A.shape[1] < T_ref_max:
            pad = T_ref_max - ref_latent_A.shape[1]
            ref_latent_A = torch.cat([ref_latent_A,
                                      ref_latent_A.new_zeros(B, pad, ref_latent_A.shape[-1])], dim=1)
            ref_mask_A = torch.cat([ref_mask_A,
                                    ref_mask_A.new_zeros(B, pad, dtype=torch.bool)], dim=1)
        if ref_latent_B.shape[1] < T_ref_max:
            pad = T_ref_max - ref_latent_B.shape[1]
            ref_latent_B = torch.cat([ref_latent_B,
                                      ref_latent_B.new_zeros(B, pad, ref_latent_B.shape[-1])], dim=1)
            ref_mask_B = torch.cat([ref_mask_B,
                                    ref_mask_B.new_zeros(B, pad, dtype=torch.bool)], dim=1)
        ref_stk = torch.cat([ref_latent_A, ref_latent_B], dim=0)
        ref_mask_stk = torch.cat([ref_mask_A, ref_mask_B], dim=0)
        # text は A/B で長さが違う可能性があるので、 max 長で右 pad してから cat
        L = max(text_A_input_ids.shape[1], text_B_input_ids.shape[1])
        if text_A_input_ids.shape[1] < L:
            pad_A = L - text_A_input_ids.shape[1]
            text_A_input_ids = torch.cat([text_A_input_ids,
                                          text_A_input_ids.new_zeros(B, pad_A)], dim=1)
            text_A_mask = torch.cat([text_A_mask,
                                     text_A_mask.new_zeros(B, pad_A, dtype=torch.bool)], dim=1)
        if text_B_input_ids.shape[1] < L:
            pad_B = L - text_B_input_ids.shape[1]
            text_B_input_ids = torch.cat([text_B_input_ids,
                                          text_B_input_ids.new_zeros(B, pad_B)], dim=1)
            text_B_mask = torch.cat([text_B_mask,
                                     text_B_mask.new_zeros(B, pad_B, dtype=torch.bool)], dim=1)
        text_stk = torch.cat([text_A_input_ids, text_B_input_ids], dim=0)
        text_mask_stk = torch.cat([text_A_mask, text_B_mask], dim=0)
        t_stk = torch.cat([t, t], dim=0)
        latent_mask_stk = None
        if latent_mask is not None:
            latent_mask_stk = torch.cat([latent_mask, latent_mask], dim=0)

        # ---- Irodori 通常 encode_conditions (text + speaker、 1ch 用そのまま) ----
        (
            text_state,
            text_mask,
            speaker_state,
            speaker_mask,
            _caption_state,
            _caption_mask,
        ) = self.base.encode_conditions(
            text_input_ids=text_stk,
            text_mask=text_mask_stk,
            ref_latent=ref_stk,
            ref_mask=ref_mask_stk,
        )

        # text_scale: text 影響を調整。 1.0 = legacy 動作。 < 1.0 で TTS-prior の引きを弱める
        if self.text_scale != 1.0:
            text_state = text_state * self.text_scale

        t_embed = get_timestep_embedding(t_stk, self.cfg.timestep_embed_dim).to(dtype=latent_stk.dtype)
        cond_embed = self.base.cond_module(t_embed)[:, None, :]

        # ---- Irodori 原型 in_proj 32→1280 ----
        x = self.base.in_proj(latent_stk)  # (2B, T, 1280)

        # activity bias
        if self.use_activity and self.activity_proj is not None:
            x = x + self.activity_proj(activity_stk.to(dtype=self.activity_proj.weight.dtype)).to(dtype=x.dtype)

        # phone bias 経路: hard (legacy) or soft
        if self.phone_condition_mode == "soft" and self.soft_phone_conditioner is not None:
            assert soft_phone_A is not None and soft_phone_B is not None, \
                "soft phone mode requires soft_phone_A/B dict in forward kwargs"
            # A/B それぞれ SoftPhoneConditioner で (B,T,model_dim) → stack して 2B に
            def _soft_phone_bias(sp_dict):
                scalars = torch.stack([
                    sp_dict["phone_pos_frac"], sp_dict["phone_dur_norm"],
                    sp_dict["dist_start_norm"], sp_dict["dist_end_norm"],
                    sp_dict["is_phone_boundary"],
                ], dim=-1)
                return self.soft_phone_conditioner(
                    sp_dict["phone_cur"], sp_dict["phone_prev"], sp_dict["phone_next"],
                    sp_dict["phone_class"], scalars,
                )
            ph_bias_A = _soft_phone_bias(soft_phone_A)
            ph_bias_B = _soft_phone_bias(soft_phone_B)
            ph_bias_stk = torch.cat([ph_bias_A, ph_bias_B], dim=0)
            x = x + ph_bias_stk.to(dtype=x.dtype)
        elif self.use_phone and self.phone_emb is not None:
            ph_emb = self.phone_emb(phone_stk)
            if self.phone_temporal_conv is not None:
                ph_emb = self.phone_temporal_conv(ph_emb.transpose(1, 2)).transpose(1, 2)
            target_dtype = next(self.phone_proj.parameters()).dtype
            x = x + self.phone_proj(ph_emb.to(dtype=target_dtype)).to(dtype=x.dtype)

        # interaction_state bias: 10-class state from activity_A/B、 chunk 単位で計算 → 2B に複製
        if self.use_interaction_state and self.interaction_state_emb is not None:
            inter_state = compute_interaction_state(activity_A, activity_B)  # (B, T) long
            inter_emb = self.interaction_state_emb(inter_state)              # (B, T, model_dim)
            inter_emb_stk = torch.cat([inter_emb, inter_emb], dim=0)         # (2B, T, model_dim)
            x = x + inter_emb_stk.to(dtype=x.dtype)

        # ---- DiT blocks (Irodori 原型、 LoRA がのる) ----
        freqs = self.base._rope_freqs(x.shape[1], x.device)
        for block in self.base.blocks:
            x = block(
                x=x,
                cond_embed=cond_embed,
                text_state=text_state,
                text_mask=text_mask,
                speaker_state=speaker_state,
                speaker_mask=speaker_mask,
                caption_state=None,
                caption_mask=None,
                freqs_cis=freqs,
                self_mask=latent_mask_stk,
                context_kv=None,
            )
        x = self.base.out_norm(x)
        # ---- Irodori 原型 out_proj 1280→32 ----
        v_stk = self.base.out_proj(x)  # (2B, T, 32)

        # ---- A/B に分割 ----
        v_A = v_stk[:B]
        v_B = v_stk[B:]
        out_A = v_A.to(dtype=latent_A.dtype)
        out_B = v_B.to(dtype=latent_B.dtype)

        # ---- timing aux logits (= final hidden を head に渡す) ----
        if return_timing_aux_logits:
            if self.timing_aux_head is None:
                raise RuntimeError("return_timing_aux_logits=True but timing_aux_head is None")
            hidden_A = x[:B]
            hidden_B = x[B:]
            aux_logits = self.timing_aux_head(hidden_A.float(), hidden_B.float())
            return out_A, out_B, aux_logits
        return out_A, out_B


def load_base_with_pretrained(
    base_config_path: str | Path,
    pretrained_repo: str = "Aratako/Irodori-TTS-500M-v2",
    pretrained_filename: str = "model.safetensors",
) -> TextToLatentRFDiT:
    """ModelConfig を yaml から構築 → HF から重み load (1ch latent_dim=32 base)。"""
    import yaml
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from irodori_tts.config import ModelConfig

    cfg_dict = yaml.safe_load(Path(base_config_path).read_text())["model"]
    cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})
    base = TextToLatentRFDiT(cfg)

    if "/" in str(pretrained_repo) and not Path(str(pretrained_repo)).exists():
        weights_path = hf_hub_download(repo_id=pretrained_repo, filename=pretrained_filename)
    else:
        weights_path = str(pretrained_repo)

    state = load_file(weights_path)
    missing, unexpected = base.load_state_dict(state, strict=False)
    if missing:
        print(f"[load_base] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[load_base] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    return base

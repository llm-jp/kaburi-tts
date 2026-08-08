"""KaburiAcousticModel = 公開用クリーン dialog TTS acoustic model。

Irodori1chTwoStreamModel (= 2-channel conditioning の base wrapper) を継承し、
実証済の 2 機能のみを追加:
  - phone bias の mid-block re-injection (= phone_reinject_blocks)
  - conditioning dropout (= forward の drop_phone_mask / drop_speaker_mask → 推論 CFG 用)

失敗した実装 (= phone_class head, latent phone classifier, speaker adapter) は一切含まない。
velocity 経路は全新規モジュール zero-init で、 step 0 で base と数値等価。

依存: kaburi_tts/two_stream/model.py
  (= Irodori1chTwoStreamModel, compute_interaction_state, SoftPhoneConditioner, TimingAuxHead)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from kaburi_tts.two_stream.model import (
    Irodori1chTwoStreamModel, compute_interaction_state,
)


class KaburiAcousticModel(Irodori1chTwoStreamModel):
    """2-stream wrapper + phone re-injection + conditioning dropout。

    新規引数:
      phone_reinject_blocks: list[int] — phone bias を再注入する block index 群 ([] で無効)
    forward 新規引数:
      drop_phone_mask  [B] bool — True の sample の phone bias を 0 化 (= CFG uncond 用)
      drop_speaker_mask [B] bool — True の sample の speaker_state を 0 化
    """

    def __init__(self, base, *, phone_reinject_blocks: list[int] | None = None, **kwargs):
        super().__init__(base, **kwargs)
        self.phone_reinject_blocks = sorted(set(phone_reinject_blocks)) if phone_reinject_blocks else []
        if self.phone_reinject_blocks:
            n_total = len(self.base.blocks)
            for b in self.phone_reinject_blocks:
                assert 0 <= b < n_total, f"reinject block {b} out of range [0,{n_total})"
            md = self.cfg.model_dim
            self.phone_reinject_proj = nn.ModuleDict({
                str(i): self._zero_linear(md, md) for i in self.phone_reinject_blocks
            })
        else:
            self.phone_reinject_proj = None

    @staticmethod
    def _zero_linear(i, o):
        m = nn.Linear(i, o)
        nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        return m

    def forward(
        self,
        latent_A, latent_B, t, *,
        phone_A, phone_B, activity_A, activity_B,
        ref_latent_A, ref_mask_A, ref_latent_B, ref_mask_B,
        text_A_input_ids, text_A_mask, text_B_input_ids, text_B_mask,
        latent_mask=None, soft_phone_A=None, soft_phone_B=None,
        return_timing_aux_logits=False,
        drop_phone_mask=None, drop_speaker_mask=None,
    ):
        from irodori_tts.model import get_timestep_embedding
        B = latent_A.shape[0]
        device = latent_A.device
        if drop_phone_mask is None:
            drop_phone_mask = torch.zeros(B, dtype=torch.bool, device=device)
        if drop_speaker_mask is None:
            drop_speaker_mask = torch.zeros(B, dtype=torch.bool, device=device)

        # ---- A/B stack ----
        latent_stk = torch.cat([latent_A, latent_B], dim=0)
        phone_stk = torch.cat([phone_A, phone_B], dim=0)
        activity_stk = torch.cat([activity_A, activity_B], dim=0).unsqueeze(-1)
        T_ref = max(ref_latent_A.shape[1], ref_latent_B.shape[1])
        def _pad_ref(rl, rm):
            if rl.shape[1] < T_ref:
                pad = T_ref - rl.shape[1]
                rl = torch.cat([rl, rl.new_zeros(B, pad, rl.shape[-1])], dim=1)
                rm = torch.cat([rm, rm.new_zeros(B, pad, dtype=torch.bool)], dim=1)
            return rl, rm
        ref_latent_A, ref_mask_A = _pad_ref(ref_latent_A, ref_mask_A)
        ref_latent_B, ref_mask_B = _pad_ref(ref_latent_B, ref_mask_B)
        ref_stk = torch.cat([ref_latent_A, ref_latent_B], dim=0)
        ref_mask_stk = torch.cat([ref_mask_A, ref_mask_B], dim=0)
        L = max(text_A_input_ids.shape[1], text_B_input_ids.shape[1])
        def _pad_txt(ti, tm):
            if ti.shape[1] < L:
                pad = L - ti.shape[1]
                ti = torch.cat([ti, ti.new_zeros(B, pad)], dim=1)
                tm = torch.cat([tm, tm.new_zeros(B, pad, dtype=torch.bool)], dim=1)
            return ti, tm
        text_A_input_ids, text_A_mask = _pad_txt(text_A_input_ids, text_A_mask)
        text_B_input_ids, text_B_mask = _pad_txt(text_B_input_ids, text_B_mask)
        text_stk = torch.cat([text_A_input_ids, text_B_input_ids], dim=0)
        text_mask_stk = torch.cat([text_A_mask, text_B_mask], dim=0)
        t_stk = torch.cat([t, t], dim=0)
        latent_mask_stk = torch.cat([latent_mask, latent_mask], dim=0) if latent_mask is not None else None
        drop_phone_stk = torch.cat([drop_phone_mask, drop_phone_mask], dim=0)
        drop_speaker_stk = torch.cat([drop_speaker_mask, drop_speaker_mask], dim=0)

        # ---- conditions ----
        (text_state, text_mask, speaker_state, speaker_mask,
         _cap_s, _cap_m) = self.base.encode_conditions(
            text_input_ids=text_stk, text_mask=text_mask_stk,
            ref_latent=ref_stk, ref_mask=ref_mask_stk)
        if self.text_scale != 1.0:
            text_state = text_state * self.text_scale
        # speaker drop
        if drop_speaker_stk.any() and speaker_state is not None:
            keep = (~drop_speaker_stk).to(dtype=speaker_state.dtype).view(-1, 1, 1)
            speaker_state = speaker_state * keep

        t_embed = get_timestep_embedding(t_stk, self.cfg.timestep_embed_dim).to(dtype=latent_stk.dtype)
        cond_embed = self.base.cond_module(t_embed)[:, None, :]
        x = self.base.in_proj(latent_stk)

        # activity bias (= drop しない)
        if self.use_activity and self.activity_proj is not None:
            x = x + self.activity_proj(activity_stk.to(dtype=self.activity_proj.weight.dtype)).to(dtype=x.dtype)

        # phone bias (= 再注入用に保持)
        phone_bias_stk = None
        if self.phone_condition_mode == "soft" and self.soft_phone_conditioner is not None:
            assert soft_phone_A is not None and soft_phone_B is not None
            def _sb(sp):
                sc = torch.stack([sp["phone_pos_frac"], sp["phone_dur_norm"],
                                  sp["dist_start_norm"], sp["dist_end_norm"], sp["is_phone_boundary"]], dim=-1)
                return self.soft_phone_conditioner(sp["phone_cur"], sp["phone_prev"], sp["phone_next"],
                                                   sp["phone_class"], sc)
            phone_bias_stk = torch.cat([_sb(soft_phone_A), _sb(soft_phone_B)], dim=0)
        elif self.use_phone and self.phone_emb is not None:
            pe = self.phone_emb(phone_stk)
            if self.phone_temporal_conv is not None:
                pe = self.phone_temporal_conv(pe.transpose(1, 2)).transpose(1, 2)
            td = next(self.phone_proj.parameters()).dtype
            phone_bias_stk = self.phone_proj(pe.to(dtype=td))
        # phone drop (= 再注入分も自動 0 になる)
        if phone_bias_stk is not None and drop_phone_stk.any():
            keep = (~drop_phone_stk).to(dtype=phone_bias_stk.dtype).view(-1, 1, 1)
            phone_bias_stk = phone_bias_stk * keep
        if phone_bias_stk is not None:
            x = x + phone_bias_stk.to(dtype=x.dtype)

        # interaction_state bias (= drop しない)
        if self.use_interaction_state and self.interaction_state_emb is not None:
            ist = compute_interaction_state(activity_A, activity_B)
            ie = self.interaction_state_emb(ist)
            x = x + torch.cat([ie, ie], dim=0).to(dtype=x.dtype)

        # ---- DiT blocks + mid-block phone re-injection ----
        freqs = self.base._rope_freqs(x.shape[1], x.device)
        for i, block in enumerate(self.base.blocks):
            if (self.phone_reinject_proj is not None and str(i) in self.phone_reinject_proj
                    and phone_bias_stk is not None):
                proj = self.phone_reinject_proj[str(i)]
                x = x + proj(phone_bias_stk.to(dtype=proj.weight.dtype)).to(dtype=x.dtype)
            x = block(x=x, cond_embed=cond_embed, text_state=text_state, text_mask=text_mask,
                      speaker_state=speaker_state, speaker_mask=speaker_mask,
                      caption_state=None, caption_mask=None, freqs_cis=freqs,
                      self_mask=latent_mask_stk, context_kv=None)
        x = self.base.out_norm(x)
        v_stk = self.base.out_proj(x)
        out_A = v_stk[:B].to(dtype=latent_A.dtype)
        out_B = v_stk[B:].to(dtype=latent_B.dtype)

        if return_timing_aux_logits:
            if self.timing_aux_head is None:
                raise RuntimeError("return_timing_aux_logits=True but timing_aux_head is None")
            aux = self.timing_aux_head(x[:B].float(), x[B:].float())
            return out_A, out_B, aux
        return out_A, out_B


__all__ = ["KaburiAcousticModel"]

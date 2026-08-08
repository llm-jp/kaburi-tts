"""dialog context dataset helpers。

acoustic dataset (= Normal2StreamDataset) は GT phone raster を出力する。
predictor では raster を input には使わず、 別途 collapsed phone sequence を作る。

利用元データ:
  - per-utt timing record format (= utterances list with phone_ids, speaker, gt_start/end)
  - utt_manifest (= text per utt、 必要なら)

collapsed phone sequence の作り方:
  1) chunk 内 utts を time 順 (= gt_start_frame ascending) に並べる
  2) 各 utt: [utt_start_token, phone_1, phone_2, ..., phone_k, utt_end_token]
  3) chunk 全体: concat、 max_collapsed_seq_len で truncate
  4) 各 token に: phone_id, phone_class, utt_boundary, speaker_id, channel_id を attach

utt_boundary values:
  0 = mid (= phone)
  1 = utt_start_token
  2 = utt_end_token
  3 = chunk_boundary  (= 予備)
"""
from __future__ import annotations

import torch


SIL_ID = 1   # = phone_vocab["<sil>"] = 1
UTT_START_ID = 0   # = phone_vocab["<pad>"] = 0 を流用 (= boundary は phone_id でなく boundary token で識別)
UTT_END_ID = 0


def build_collapsed_phone_sequence(
    utts: list[dict],
    *,
    max_collapsed_seq_len: int = 512,
    speaker_to_idx: dict[str, int] | None = None,
    phone_class_lookup: torch.Tensor | None = None,    # [n_phones] long
    fallback_phone_class: int = 0,
) -> dict[str, torch.Tensor]:
    """utts (= chunk 内の utts) を collapsed phone sequence に展開。

    Returns:
      phone_ids [N] long
      phone_class [N] long
      utt_boundary [N] long
      speaker_per_phone [N] long
      channel_per_phone [N] long (= 0=A, 1=B)
      attn_mask [N] bool (= valid)
      utt_token_ranges list[(utt_idx, start, end)]  (= 元 utt → collapsed seq の range)
    """
    # sort by gt_start_frame
    order = sorted(range(len(utts)), key=lambda i: utts[i].get("gt_start_frame", 0))
    tokens_ph = []; tokens_cls = []; tokens_bnd = []
    tokens_spk = []; tokens_ch = []
    utt_token_ranges = []
    for u_idx in order:
        u = utts[u_idx]
        spk_str = u.get("speaker", "A")   # = A/B
        speaker_name = u.get("utt_id", "")[:7] if u.get("utt_id") else "?"
        # speaker_id mapping: 簡易に speaker_str (= A/B) を使わず、 utt_id から W?? を抽出
        # = chunk-level speaker_A/B が必要なため、 dataset 側で予め渡される想定
        # ここでは utts dict 内に "speaker_idx" key があれば使う、 なければ channel と同じ
        spk_idx = int(u.get("speaker_idx", 0)) if "speaker_idx" in u else 0
        ch_idx = 0 if spk_str == "A" else 1

        start = len(tokens_ph)
        # utt_start token (= phone_id=0、 boundary=1)
        tokens_ph.append(0); tokens_cls.append(0); tokens_bnd.append(1)
        tokens_spk.append(spk_idx); tokens_ch.append(ch_idx)
        # phone tokens
        for ph in u.get("phone_ids", []):
            ph_int = int(ph)
            tokens_ph.append(ph_int)
            cls = int(phone_class_lookup[ph_int].item()) if phone_class_lookup is not None and ph_int < len(phone_class_lookup) else fallback_phone_class
            tokens_cls.append(cls)
            tokens_bnd.append(0)
            tokens_spk.append(spk_idx)
            tokens_ch.append(ch_idx)
        # utt_end token
        tokens_ph.append(0); tokens_cls.append(0); tokens_bnd.append(2)
        tokens_spk.append(spk_idx); tokens_ch.append(ch_idx)
        end = len(tokens_ph)
        utt_token_ranges.append((u_idx, start, end))
        if len(tokens_ph) >= max_collapsed_seq_len:
            break
    # truncate
    if len(tokens_ph) > max_collapsed_seq_len:
        tokens_ph = tokens_ph[:max_collapsed_seq_len]
        tokens_cls = tokens_cls[:max_collapsed_seq_len]
        tokens_bnd = tokens_bnd[:max_collapsed_seq_len]
        tokens_spk = tokens_spk[:max_collapsed_seq_len]
        tokens_ch = tokens_ch[:max_collapsed_seq_len]

    return {
        "phone_ids": torch.tensor(tokens_ph, dtype=torch.long),
        "phone_class": torch.tensor(tokens_cls, dtype=torch.long),
        "utt_boundary": torch.tensor(tokens_bnd, dtype=torch.long),
        "speaker_per_phone": torch.tensor(tokens_spk, dtype=torch.long),
        "channel_per_phone": torch.tensor(tokens_ch, dtype=torch.long),
        "attn_mask": torch.ones(len(tokens_ph), dtype=torch.bool),
        "utt_token_ranges": utt_token_ranges,
    }


def collate_collapsed_batch(batch: list[dict]) -> dict[str, torch.Tensor]:
    """list of collapsed dicts → padded batch [B, N_max]。"""
    B = len(batch)
    N_max = max(b["phone_ids"].shape[0] for b in batch)
    out = {}
    for key in ["phone_ids", "phone_class", "utt_boundary", "speaker_per_phone", "channel_per_phone"]:
        t = torch.zeros(B, N_max, dtype=torch.long)
        for i, b in enumerate(batch):
            n = b[key].shape[0]
            t[i, :n] = b[key]
        out[key] = t
    attn_mask = torch.zeros(B, N_max, dtype=torch.bool)
    for i, b in enumerate(batch):
        attn_mask[i, :b["attn_mask"].shape[0]] = b["attn_mask"]
    out["attn_mask"] = attn_mask
    out["utt_token_ranges"] = [b["utt_token_ranges"] for b in batch]
    return out


__all__ = ["build_collapsed_phone_sequence", "collate_collapsed_batch"]

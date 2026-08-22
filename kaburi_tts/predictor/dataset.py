"""predictor dataset.

Wraps normal dialogue chunks and attaches a per-channel PRE_SIL + PHONE token
sequence, utterance text context fields, duration targets, any-gap targets, and
frame-level activity-state labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from kaburi_tts.predictor.context_dataset import DialogChunkDataset, collate_chunks

from kaburi_tts.predictor import (
    PHONE_BIN_VALUES, SIL_BIN_VALUES, GAP_BIN_VALUES,
    N_PHONE_CLASSES, N_SIL_CLASSES,
    TOKEN_TYPE_PAD, TOKEN_TYPE_PHONE, TOKEN_TYPE_PRE_SIL, TOKEN_TYPE_FIRST_SIL,
    STATE_NONE, STATE_A_ONLY, STATE_B_ONLY, STATE_BOTH,
)
from kaburi_tts.predictor.frame_state import build_gt_frame_state_label


# -------- utterance metadata ids (= 0 is PAD/unknown) --------
TRANSITION_TYPE_TO_ID = {
    "chunk_first": 1,
    "A_to_B": 2,
    "B_to_A": 3,
    "A_to_A": 4,
    "B_to_B": 5,
}
LENGTH_BIN_TO_ID = {
    "short": 1,
    "mid": 2,
    "long": 3,
}


# -------- bin lookup helpers (= cached) --------
_SIL_BIN_TENSOR = torch.tensor(SIL_BIN_VALUES, dtype=torch.float32)
_GAP_BIN_TENSOR = torch.tensor(GAP_BIN_VALUES, dtype=torch.float32)


def sil_value_to_bin(value: float) -> int:
    """SIL frame 数 を 最近傍 bin index に変換。 overflow (= value > max) は last bin。"""
    if value >= SIL_BIN_VALUES[-1]:
        return len(SIL_BIN_VALUES) - 1
    if value <= 0:
        return 0
    diff = (_SIL_BIN_TENSOR - float(value)).abs()
    return int(diff.argmin().item())


def phone_value_to_bin(value: int) -> int:
    """PHONE frame 数 を bin index に変換 (= class c → c+1 frames、 1..30)。"""
    v = max(1, min(N_PHONE_CLASSES, int(value)))
    return v - 1


def gap_value_to_bin(value: float) -> int:
    """cross-channel gap frame 数を最近傍 bin index に変換。"""
    if value <= GAP_BIN_VALUES[0]:
        return 0
    if value >= GAP_BIN_VALUES[-1]:
        return len(GAP_BIN_VALUES) - 1
    diff = (_GAP_BIN_TENSOR - float(value)).abs()
    return int(diff.argmin().item())


class _TimingAttachMixin:
    """Attach timing-token fields to a normal dialogue chunk item."""

    T_frames: int = 750
    max_timing_seq_len: int = 1024
    max_utt_text_len: int = 64

    def _attach_timing(self, item: dict, rec: dict) -> dict:
        utts = rec["utterances"]
        spk_A_str = rec.get("speaker_A", "")
        spk_B_str = rec.get("speaker_B", "")
        spk_A_idx = self.spk_to_idx.get(spk_A_str, 0)
        spk_B_idx = self.spk_to_idx.get(spk_B_str, 0)

        # split + sort per channel. Keep original utterance ids for cross-channel gap targets.
        utts_with_id = []
        for orig_idx, u in enumerate(utts):
            uu = dict(u)
            uu["_timing_orig_utt_idx"] = orig_idx
            utts_with_id.append(uu)

        for u in utts_with_id:
            if not u.get("text"):
                u["text"] = getattr(self, "utt_text_by_id", {}).get(u.get("utt_id", ""), "")

        any_gap_by_orig = _compute_any_gap_by_orig_utt(utts_with_id)
        dialog_order_orig = [
            int(u["_timing_orig_utt_idx"])
            for u in sorted(
                utts_with_id,
                key=lambda u: (int(u.get("gt_start_frame", 0)), int(u.get("gt_end_frame", 0))),
            )
        ]
        dialog_rank_by_orig = {orig: rank for rank, orig in enumerate(dialog_order_orig)}

        utts_A = sorted([u for u in utts_with_id if u.get("speaker", "A") == "A"],
                        key=lambda u: int(u.get("gt_start_frame", 0)))
        utts_B = sorted([u for u in utts_with_id if u.get("speaker", "A") == "B"],
                        key=lambda u: int(u.get("gt_start_frame", 0)))

        ch_A = _build_channel_tokens(
            utts_A, channel_idx=0, speaker_idx=spk_A_idx, any_gap_by_orig=any_gap_by_orig,
            dialog_rank_by_orig=dialog_rank_by_orig,
            text_tokenizer=getattr(self, "timing_text_tokenizer", None),
            max_utt_text_len=getattr(self, "max_utt_text_len", 64),
        )
        ch_B = _build_channel_tokens(
            utts_B, channel_idx=1, speaker_idx=spk_B_idx, any_gap_by_orig=any_gap_by_orig,
            dialog_rank_by_orig=dialog_rank_by_orig,
            text_tokenizer=getattr(self, "timing_text_tokenizer", None),
            max_utt_text_len=getattr(self, "max_utt_text_len", 64),
        )

        merged = _merge_channels(ch_A, ch_B, self.max_timing_seq_len, dialog_order_orig)

        # GT 4-state frame label
        frame_state = build_gt_frame_state_label(utts, self.T_frames)

        # per-channel GT total length (= 末尾 utt の gt_end_frame、 channel 全活動長)
        gt_total_A = int(utts_A[-1].get("gt_end_frame", 0)) if utts_A else 0
        gt_total_B = int(utts_B[-1].get("gt_end_frame", 0)) if utts_B else 0

        # per-utt GT start/end per channel (= 評価用、 推論時 reconstruction との比較に使う)
        utt_starts_A = [int(u.get("gt_start_frame", 0)) for u in utts_A]
        utt_ends_A = [int(u.get("gt_end_frame", 0)) for u in utts_A]
        utt_starts_B = [int(u.get("gt_start_frame", 0)) for u in utts_B]
        utt_ends_B = [int(u.get("gt_end_frame", 0)) for u in utts_B]

        item.update({
            **merged,
            "timing_frame_state_label": frame_state,
            "timing_gt_total_len_A": gt_total_A,
            "timing_gt_total_len_B": gt_total_B,
            "timing_n_tokens_A": ch_A["n_tokens"],
            "timing_n_tokens_B": ch_B["n_tokens"],
            "timing_n_utts_A": ch_A["n_utts"],
            "timing_n_utts_B": ch_B["n_utts"],
            "timing_utt_starts_A": utt_starts_A,
            "timing_utt_ends_A": utt_ends_A,
            "timing_utt_starts_B": utt_starts_B,
            "timing_utt_ends_B": utt_ends_B,
            "timing_global_utt_order": merged["timing_global_utt_order"],
        })
        return item


def _compute_any_gap_by_orig_utt(utts: list) -> dict[int, float]:
    """dialog order での previous-any-end からの gap を計算。

    gap < 0 means cross-channel overlap. The first utterance uses its start from chunk head.
    """
    by_start = sorted(utts, key=lambda u: (int(u.get("gt_start_frame", 0)), int(u.get("gt_end_frame", 0))))
    last_any_end = 0
    out: dict[int, float] = {}
    for u in by_start:
        orig_idx = int(u.get("_timing_orig_utt_idx", -1))
        start = int(u.get("gt_start_frame", 0))
        end = int(u.get("gt_end_frame", 0))
        out[orig_idx] = float(start - last_any_end)
        last_any_end = max(last_any_end, end)
    return out


def _build_channel_tokens(
    utts_ch: list,
    *,
    channel_idx: int,
    speaker_idx: int,
    any_gap_by_orig: dict[int, float],
    dialog_rank_by_orig: dict[int, int],
    text_tokenizer=None,
    max_utt_text_len: int = 64,
) -> dict:
    """1 channel 分の [PRE_SIL, PHONE, PHONE, ..., PRE_SIL, PHONE, ...] sequence を構築。

    PRE_SIL は 前 utt 終端からの silence frame 数を target に持つ。
    最初の utt の PRE_SIL は FIRST_SIL としてマークし、chunk head からの start を学習する。
    """
    token_types: list[int] = []
    phone_ids: list[int] = []
    utt_indices: list[int] = []
    phone_dur_targets: list[int] = []   # = PHONE 用 class bin、 SIL token は -100
    sil_dur_targets: list[int] = []     # = SIL 用 class bin、 PHONE token は -100
    any_gap_targets: list[int] = []     # = utt first token 用 class bin、 other token は -100
    phone_dur_gt_frames: list[float] = []
    sil_dur_gt_frames: list[float] = []
    any_gap_gt_frames: list[float] = []
    is_utt_first_token: list[int] = []  # = 1 if this token is the PRE_SIL of an utt (= utt-level pool 入口)
    orig_utt_indices: list[int] = []
    dialog_ranks: list[int] = []
    transition_type_ids: list[int] = []
    length_bin_ids: list[int] = []
    utt_region_len_gt_frames: list[float] = []
    utt_text_ids: list[list[int]] = []
    utt_text_mask: list[list[int]] = []

    prev_end = -1  # = sentinel: 最初の utt
    for i, u in enumerate(utts_ch):
        start = int(u.get("gt_start_frame", 0))
        end = int(u.get("gt_end_frame", 0))
        ph_ids = [int(p) for p in u.get("phone_ids", [])]
        ph_durs = [float(d) for d in u.get("gt_phone_durations", [])]

        # --- PRE_SIL or FIRST_SIL token ---
        if prev_end < 0:
            sil_value = max(0, start)
            t_type = TOKEN_TYPE_FIRST_SIL  # = dur loss から除外
            sil_target = sil_value_to_bin(sil_value)
        else:
            sil_value = max(0, start - prev_end)
            t_type = TOKEN_TYPE_PRE_SIL
            sil_target = sil_value_to_bin(sil_value)
        orig_idx = int(u.get("_timing_orig_utt_idx", i))
        dialog_rank = int(dialog_rank_by_orig.get(orig_idx, i))
        transition_id = TRANSITION_TYPE_TO_ID.get(str(u.get("transition_type", "")), 0)
        length_id = LENGTH_BIN_TO_ID.get(str(u.get("length_bin", "")), 0)
        region_len = float(max(0, int(u.get("gt_region_len", max(0, end - start)))))
        text_ids, text_mask = _encode_utt_text(
            str(u.get("text", "")), text_tokenizer, max_utt_text_len,
        )
        any_gap_value = any_gap_by_orig.get(orig_idx, float(start))
        token_types.append(t_type)
        phone_ids.append(0)
        utt_indices.append(i)
        phone_dur_targets.append(-100)
        sil_dur_targets.append(sil_target)
        any_gap_targets.append(gap_value_to_bin(any_gap_value))
        phone_dur_gt_frames.append(0.0)
        sil_dur_gt_frames.append(float(sil_value))
        any_gap_gt_frames.append(float(any_gap_value))
        is_utt_first_token.append(1)
        orig_utt_indices.append(orig_idx)
        dialog_ranks.append(dialog_rank)
        transition_type_ids.append(transition_id)
        length_bin_ids.append(length_id)
        utt_region_len_gt_frames.append(region_len)
        utt_text_ids.append(text_ids)
        utt_text_mask.append(text_mask)

        # --- PHONE tokens ---
        for j, ph in enumerate(ph_ids):
            dur = ph_durs[j] if j < len(ph_durs) else 0.0
            ph_target = phone_value_to_bin(int(round(dur))) if dur > 0 else -100
            token_types.append(TOKEN_TYPE_PHONE)
            phone_ids.append(int(ph))
            utt_indices.append(i)
            phone_dur_targets.append(ph_target)
            sil_dur_targets.append(-100)
            any_gap_targets.append(-100)
            phone_dur_gt_frames.append(float(dur))
            sil_dur_gt_frames.append(0.0)
            any_gap_gt_frames.append(0.0)
            is_utt_first_token.append(0)
            orig_utt_indices.append(orig_idx)
            dialog_ranks.append(dialog_rank)
            transition_type_ids.append(transition_id)
            length_bin_ids.append(length_id)
            utt_region_len_gt_frames.append(region_len)
            utt_text_ids.append(text_ids)
            utt_text_mask.append(text_mask)

        prev_end = end

    return {
        "channel_idx": channel_idx,
        "speaker_idx": speaker_idx,
        "token_types": token_types,
        "phone_ids": phone_ids,
        "utt_indices": utt_indices,
        "phone_dur_targets": phone_dur_targets,
        "sil_dur_targets": sil_dur_targets,
        "any_gap_targets": any_gap_targets,
        "phone_dur_gt_frames": phone_dur_gt_frames,
        "sil_dur_gt_frames": sil_dur_gt_frames,
        "any_gap_gt_frames": any_gap_gt_frames,
        "is_utt_first_token": is_utt_first_token,
        "orig_utt_indices": orig_utt_indices,
        "dialog_ranks": dialog_ranks,
        "transition_type_ids": transition_type_ids,
        "length_bin_ids": length_bin_ids,
        "utt_region_len_gt_frames": utt_region_len_gt_frames,
        "utt_text_ids": utt_text_ids,
        "utt_text_mask": utt_text_mask,
        "n_tokens": len(token_types),
        "n_utts": len(utts_ch),
    }


def _encode_utt_text(text: str, text_tokenizer, max_len: int) -> tuple[list[int], list[int]]:
    """Encode utterance text for timing_text. Empty text keeps a single pad vector."""
    ids: list[int] = []
    if text and text_tokenizer is not None:
        try:
            enc = text_tokenizer.encode(text)
            if torch.is_tensor(enc):
                ids = [int(x) for x in enc[:max_len].tolist()]
            else:
                ids = [int(x) for x in list(enc)[:max_len]]
        except Exception:
            ids = []
    ids = [x for x in ids if x > 0][:max_len]
    mask = [1] * len(ids)
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
        mask = mask + [0] * (max_len - len(mask))
    return ids, mask


def _merge_channels(ch_A: dict, ch_B: dict, max_len: int, dialog_order_orig: list[int]) -> dict:
    """A の token 列 と B の token 列 を concat、 token ごとに channel_id / speaker_id 付与。

    位置 embedding は **channel 内 position** を 使う (= A と B が 独立に 0..N-1)。
    truncate は 末尾から (= 同 channel 内 整合性 維持)。
    """
    nA = ch_A["n_tokens"]
    nB = ch_B["n_tokens"]

    def _concat(field):
        return ch_A[field] + ch_B[field]

    all_types = _concat("token_types")
    all_phones = _concat("phone_ids")
    all_utts = _concat("utt_indices")
    all_phone_dur_t = _concat("phone_dur_targets")
    all_sil_dur_t = _concat("sil_dur_targets")
    all_any_gap_t = _concat("any_gap_targets")
    all_phone_dur_gt = _concat("phone_dur_gt_frames")
    all_sil_dur_gt = _concat("sil_dur_gt_frames")
    all_any_gap_gt = _concat("any_gap_gt_frames")
    all_first_tok = _concat("is_utt_first_token")
    all_orig_utts = _concat("orig_utt_indices")
    all_dialog_ranks = _concat("dialog_ranks")
    all_transition_ids = _concat("transition_type_ids")
    all_length_ids = _concat("length_bin_ids")
    all_utt_region_gt = _concat("utt_region_len_gt_frames")
    all_text_ids = _concat("utt_text_ids")
    all_text_mask = _concat("utt_text_mask")
    all_channels = [ch_A["channel_idx"]] * nA + [ch_B["channel_idx"]] * nB
    all_speakers = [ch_A["speaker_idx"]] * nA + [ch_B["speaker_idx"]] * nB
    pos_in_ch = list(range(nA)) + list(range(nB))

    # truncate (= channel 別に末尾切り、 ただし全体長で cap)
    if len(all_types) > max_len:
        # truncate B 後半 first、 then A 後半。 簡便のため後ろから切る。
        cut = max_len
        all_types = all_types[:cut]
        all_phones = all_phones[:cut]
        all_utts = all_utts[:cut]
        all_phone_dur_t = all_phone_dur_t[:cut]
        all_sil_dur_t = all_sil_dur_t[:cut]
        all_any_gap_t = all_any_gap_t[:cut]
        all_phone_dur_gt = all_phone_dur_gt[:cut]
        all_sil_dur_gt = all_sil_dur_gt[:cut]
        all_any_gap_gt = all_any_gap_gt[:cut]
        all_first_tok = all_first_tok[:cut]
        all_orig_utts = all_orig_utts[:cut]
        all_dialog_ranks = all_dialog_ranks[:cut]
        all_transition_ids = all_transition_ids[:cut]
        all_length_ids = all_length_ids[:cut]
        all_utt_region_gt = all_utt_region_gt[:cut]
        all_text_ids = all_text_ids[:cut]
        all_text_mask = all_text_mask[:cut]
        all_channels = all_channels[:cut]
        all_speakers = all_speakers[:cut]
        pos_in_ch = pos_in_ch[:cut]

    first_by_orig = {
        all_orig_utts[i]: i
        for i, is_first in enumerate(all_first_tok)
        if is_first and i < len(all_types)
    }
    first_token_positions = [first_by_orig[o] for o in dialog_order_orig if o in first_by_orig]

    return {
        "timing_token_type": torch.tensor(all_types, dtype=torch.long),
        "timing_phone_id": torch.tensor(all_phones, dtype=torch.long),
        "timing_utt_index": torch.tensor(all_utts, dtype=torch.long),
        "timing_channel_id": torch.tensor(all_channels, dtype=torch.long),
        "timing_speaker_id_per_token": torch.tensor(all_speakers, dtype=torch.long),
        "timing_phone_dur_target": torch.tensor(all_phone_dur_t, dtype=torch.long),
        "timing_sil_dur_target": torch.tensor(all_sil_dur_t, dtype=torch.long),
        "timing_any_gap_target": torch.tensor(all_any_gap_t, dtype=torch.long),
        "timing_phone_dur_gt": torch.tensor(all_phone_dur_gt, dtype=torch.float32),
        "timing_sil_dur_gt": torch.tensor(all_sil_dur_gt, dtype=torch.float32),
        "timing_any_gap_gt": torch.tensor(all_any_gap_gt, dtype=torch.float32),
        "timing_is_utt_first_token": torch.tensor(all_first_tok, dtype=torch.long),
        "timing_orig_utt_index": torch.tensor(all_orig_utts, dtype=torch.long),
        "timing_dialog_rank": torch.tensor(all_dialog_ranks, dtype=torch.long),
        "timing_transition_type_id": torch.tensor(all_transition_ids, dtype=torch.long),
        "timing_length_bin_id": torch.tensor(all_length_ids, dtype=torch.long),
        "timing_utt_region_len_gt": torch.tensor(all_utt_region_gt, dtype=torch.float32),
        "timing_utt_text_ids": torch.tensor(all_text_ids, dtype=torch.long),
        "timing_utt_text_mask": torch.tensor(all_text_mask, dtype=torch.bool),
        "timing_pos_in_channel": torch.tensor(pos_in_ch, dtype=torch.long),
        "timing_mask": torch.ones(len(all_types), dtype=torch.bool),
        "timing_split_A_end": nA,  # = A の token 数 (= 0..nA-1 が A、 nA..nA+nB-1 が B)
        "timing_global_utt_order": first_token_positions,
    }


# -------- Dataset classes (= dialog chunk dataset を base に timing fields attach) --------
class TimingPredictorDataset(_TimingAttachMixin, DialogChunkDataset):
    def __init__(
        self, *args, T_frames: int = 750, max_timing_seq_len: int = 1024,
        max_utt_text_len: int = 64,
        utt_manifest_path: str | None = None,
        **kwargs,
    ):
        self.timing_text_tokenizer = kwargs.get("text_tokenizer")
        self.max_utt_text_len = int(max_utt_text_len)
        super().__init__(*args, **kwargs)
        self.T_frames = T_frames
        self.max_timing_seq_len = max_timing_seq_len
        self.utt_text_by_id = _load_utt_text_map(utt_manifest_path)

    @property
    def chunk_ids(self) -> list[str]:
        """Chunk ids that are actually usable after utt-timing filtering."""
        return [
            str(self.inner.entries[i]["chunk_id"])
            for i in self.valid_indices
        ]

    def __getitem__(self, k: int) -> dict:
        item = super().__getitem__(k)
        rec = self.utt_timing[item["chunk_id"]]
        return self._attach_timing(item, rec)

# -------- collate --------
_TIMING_LONG_FIELDS = [
    "timing_token_type", "timing_phone_id", "timing_utt_index",
    "timing_channel_id", "timing_speaker_id_per_token",
    "timing_phone_dur_target", "timing_sil_dur_target", "timing_any_gap_target",
    "timing_is_utt_first_token", "timing_orig_utt_index", "timing_pos_in_channel",
    "timing_dialog_rank", "timing_transition_type_id", "timing_length_bin_id",
    "timing_utt_text_ids",
]
_TIMING_FLOAT_FIELDS = [
    "timing_phone_dur_gt", "timing_sil_dur_gt", "timing_any_gap_gt",
    "timing_utt_region_len_gt",
]
_TIMING_BOOL_FIELDS = ["timing_mask", "timing_utt_text_mask"]
_TIMING_PER_BATCH_FIELDS_LONG = [
    "timing_gt_total_len_A", "timing_gt_total_len_B",
    "timing_n_tokens_A", "timing_n_tokens_B",
    "timing_n_utts_A", "timing_n_utts_B",
    "timing_split_A_end",
]
_TIMING_LIST_FIELDS = [
    "timing_utt_starts_A", "timing_utt_ends_A",
    "timing_utt_starts_B", "timing_utt_ends_B",
    "timing_global_utt_order",
]
_TIMING_ALL_KEYS = (_TIMING_LONG_FIELDS + _TIMING_FLOAT_FIELDS + _TIMING_BOOL_FIELDS
                 + _TIMING_PER_BATCH_FIELDS_LONG + _TIMING_LIST_FIELDS
                 + ["timing_frame_state_label"])


def collate_timing(batch: list[dict]) -> dict:
    """chunk collate + timing fields。 timing fields は chunk collate に渡る前に stash。"""
    # 1) timing fields を 抜き取って 別 collate
    timing_stash = {k: [] for k in _TIMING_ALL_KEYS}
    for item in batch:
        for k in _TIMING_ALL_KEYS:
            if k in item:
                timing_stash[k].append(item[k])

    # 2) timing fields を 除いた batch を base collate に渡す
    cleaned_batch = []
    for item in batch:
        c = {k: v for k, v in item.items() if k not in _TIMING_ALL_KEYS}
        cleaned_batch.append(c)
    out = collate_chunks(cleaned_batch)

    # 3) timing fields padding + stack
    B = len(batch)
    N_max = max(t.shape[0] for t in timing_stash["timing_token_type"])

    for k in _TIMING_LONG_FIELDS:
        pad_val = -100 if k.endswith("target") else 0
        if k == "timing_utt_text_ids":
            L = timing_stash[k][0].shape[1]
            t = torch.full((B, N_max, L), pad_val, dtype=torch.long)
            for i, ten in enumerate(timing_stash[k]):
                t[i, : ten.shape[0], :] = ten
        else:
            t = torch.full((B, N_max), pad_val, dtype=torch.long)
            for i, ten in enumerate(timing_stash[k]):
                t[i, : ten.shape[0]] = ten
        out[k] = t

    for k in _TIMING_FLOAT_FIELDS:
        t = torch.zeros(B, N_max, dtype=torch.float32)
        for i, ten in enumerate(timing_stash[k]):
            t[i, : ten.shape[0]] = ten
        out[k] = t

    for k in _TIMING_BOOL_FIELDS:
        if k == "timing_utt_text_mask":
            L = timing_stash[k][0].shape[1]
            mask = torch.zeros(B, N_max, L, dtype=torch.bool)
            for i, m in enumerate(timing_stash[k]):
                mask[i, : m.shape[0], :] = m
        else:
            mask = torch.zeros(B, N_max, dtype=torch.bool)
            for i, m in enumerate(timing_stash[k]):
                mask[i, : m.shape[0]] = m
        out[k] = mask

    out["timing_frame_state_label"] = torch.stack(timing_stash["timing_frame_state_label"])

    for k in _TIMING_PER_BATCH_FIELDS_LONG:
        out[k] = torch.tensor(timing_stash[k], dtype=torch.long)

    for k in _TIMING_LIST_FIELDS:
        out[k] = timing_stash[k]  # = list of list、 batch ごと utt 数が異なるため tensor 化しない

    return out


def _load_utt_text_map(path: str | None) -> dict[str, str]:
    if not path or not Path(path).exists():
        return {}
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = r.get("utt_id")
            if uid:
                out[str(uid)] = str(r.get("text", ""))
    return out


__all__ = [
    "TimingPredictorDataset",
    "collate_timing",
    "TRANSITION_TYPE_TO_ID",
    "LENGTH_BIN_TO_ID",
    "sil_value_to_bin",
    "phone_value_to_bin",
    "gap_value_to_bin",
]

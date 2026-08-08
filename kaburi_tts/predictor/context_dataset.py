"""dialog chunk dataset = Normal2StreamDataset / Event2StreamDataset + per-utt timing record (collapsed phone sequence) の join。

各 item は standard fields + 以下を追加:
  - collapsed_phone_ids [N]
  - collapsed_phone_class [N]
  - collapsed_utt_boundary [N]
  - collapsed_speaker_per_phone [N]
  - collapsed_channel_per_phone [N]
  - collapsed_phone_mask [N] bool
  - utt_token_ranges list[(utt_idx, start, end)]
  - utt_gt_start_frames list[int]
  - utt_gt_end_frames list[int]
  - speaker_id_A, speaker_id_B (= int)

Normal と Event で同じ collapsed seq logic (= chunk_id 経由で utt-timing record を引く)。
Event sample は core_mask で loss が core 部分のみ重み付け、 collapsed seq は full chunk 分を使う。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from kaburi_tts.two_stream.dataset import (
    Normal2StreamDataset, Event2StreamDataset, collate_eventmix,
)

from kaburi_tts.predictor.context_helpers import build_collapsed_phone_sequence


def _load_utt_timing_records(jsonl_paths: list[str]) -> dict[str, dict]:
    """utt-timing raw jsonl から chunk_id → record の dict 構築。"""
    out = {}
    for p in jsonl_paths:
        if not Path(p).exists(): continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                out[r["chunk_id"]] = r
    return out


def _build_speaker_to_idx(utt_timing_recs: dict[str, dict]) -> dict[str, int]:
    """utt-timing records から speaker_str → idx mapping。"""
    spks = set()
    for r in utt_timing_recs.values():
        if r.get("speaker_A"): spks.add(r["speaker_A"])
        if r.get("speaker_B"): spks.add(r["speaker_B"])
    out = {"<pad>": 0}
    for sp in sorted(spks):
        out[sp] = len(out)
    return out


def load_default_speaker_to_idx() -> dict[str, int] | None:
    """学習時と同一の speaker→idx map (= assets/speaker_to_idx.json)。

    推論時に一部話者しか含まない utt-timing record を渡しても、 学習時の
    speaker embedding index と整合させるための既定 map。 未知話者は 0 に fallback。
    """
    from kaburi_tts import ASSETS_DIR
    p = ASSETS_DIR / "speaker_to_idx.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


class _DialogChunkDatasetBase(Dataset):
    """Normal2Stream / Event2Stream を wrap して collapsed phone seq を join する共通実装。"""
    def __init__(
        self,
        inner_ds: Dataset,
        utt_timing_recs: dict[str, dict],
        speaker_to_idx: dict[str, int],
        *,
        max_collapsed_seq_len: int = 512,
        phone_class_lookup: torch.Tensor | None = None,
    ):
        self.inner = inner_ds
        self.utt_timing = utt_timing_recs
        self.spk_to_idx = speaker_to_idx
        self.max_collapsed_seq_len = max_collapsed_seq_len
        self.phone_class_lookup = phone_class_lookup

        # filter: chunk_id が utt-timing records にあるもののみ使用
        # _index_map 互換 (= Normal2Stream / Event2Stream の内部 indexing)
        self._build_valid_indices()

    def _build_valid_indices(self):
        self.valid_indices = []
        for i, e in enumerate(self.inner.entries):
            if e.get("chunk_id", "") in self.utt_timing:
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, k: int) -> dict:
        inner_idx = self.valid_indices[k]
        # Inner __getitem__ は _index_map 経由を期待するため、 inner_idx を直接渡す
        # Normal2Stream / Event2Stream は __getitem__(k) で _index_map[k] を見るので
        # ここでは valid_indices[k] が _index_map の値ではなく、 raw entry index
        # → そのまま raw を渡せばよいが Normal2Stream の __getitem__ は _index_map[k] を見るので
        # _index_map を経由した k に変換する必要あり
        # 簡便のため、 inner.entries[inner_idx] から直接 item 構築は inner の機能依存
        # → inner_ds の __getitem__ を呼び出すが、 inner_ds の _index_map が k → entry_idx mapping
        # を持っているため、 そこに合わせて変換:
        if hasattr(self.inner, "_index_map"):
            # 元 dataset の _index_map のどこに inner_idx があるか
            # = inner_idx は entries 上の位置、 _index_map[k_inner] = inner_idx となる k_inner を見つける
            # 簡易: inner._index_map.index(inner_idx) (= O(n))
            try:
                k_inner = self.inner._index_map.index(inner_idx)
            except ValueError:
                # = balanced sampling で除外された場合、 skip 不能なので next を試す
                k_inner = inner_idx
        else:
            k_inner = inner_idx
        item = dict(self.inner[k_inner])
        cid = item["chunk_id"]
        rec = self.utt_timing[cid]
        return self._attach_collapsed(item, rec)

    def _attach_collapsed(self, item: dict, rec: dict) -> dict:
        utts = rec["utterances"]
        spk_A = rec.get("speaker_A", "")
        spk_B = rec.get("speaker_B", "")
        spk_A_idx = self.spk_to_idx.get(spk_A, 0)
        spk_B_idx = self.spk_to_idx.get(spk_B, 0)
        utts_aug = []
        for u in utts:
            uu = dict(u)
            ch = u.get("speaker", "A")
            uu["speaker_idx"] = spk_A_idx if ch == "A" else spk_B_idx
            utts_aug.append(uu)
        collapsed = build_collapsed_phone_sequence(
            utts_aug,
            max_collapsed_seq_len=self.max_collapsed_seq_len,
            phone_class_lookup=self.phone_class_lookup,
        )
        ranges = collapsed["utt_token_ranges"]
        utt_gt_starts = [int(utts[u_idx]["gt_start_frame"]) for (u_idx, _, _) in ranges]
        utt_gt_ends = [int(utts[u_idx]["gt_end_frame"]) for (u_idx, _, _) in ranges]
        # per-phone GT duration (= utt_timing "gt_phone_durations") per utt
        # 各 utt の phone_ids と長さが揃った list (= phone 単位 frame 数)
        utt_gt_phone_durations = [list(utts[u_idx].get("gt_phone_durations", [])) for (u_idx, _, _) in ranges]
        item.update({
            "collapsed_phone_ids": collapsed["phone_ids"],
            "collapsed_phone_class": collapsed["phone_class"],
            "collapsed_utt_boundary": collapsed["utt_boundary"],
            "collapsed_speaker_per_phone": collapsed["speaker_per_phone"],
            "collapsed_channel_per_phone": collapsed["channel_per_phone"],
            "collapsed_phone_mask": collapsed["attn_mask"],
            "utt_token_ranges": collapsed["utt_token_ranges"],
            "utt_gt_start_frames": utt_gt_starts,
            "utt_gt_end_frames": utt_gt_ends,
            "utt_gt_phone_durations": utt_gt_phone_durations,
            "speaker_id_A": spk_A_idx,
            "speaker_id_B": spk_B_idx,
        })
        return item


class DialogChunkDataset(_DialogChunkDatasetBase):
    """Normal2StreamDataset + per-utt timing collapsed seq。"""
    def __init__(
        self,
        acoustic_manifest_path: str,
        utt_timing_jsonl_paths: list[str],
        *,
        text_tokenizer,
        max_text_len: int = 64,
        latent_T: int = 750,
        soft_radius: int = 5,
        max_collapsed_seq_len: int = 512,
        phone_class_lookup: torch.Tensor | None = None,
        speaker_to_idx: dict[str, int] | None = None,
    ):
        inner = Normal2StreamDataset(
            acoustic_manifest_path,
            text_tokenizer=text_tokenizer, max_text_len=max_text_len,
            latent_T=latent_T, speaker_balanced=False,
            soft_phone=True, soft_boundary_radius=soft_radius,
        )
        utt_timing = _load_utt_timing_records(utt_timing_jsonl_paths)
        spk_idx = speaker_to_idx or load_default_speaker_to_idx() or _build_speaker_to_idx(utt_timing)
        super().__init__(
            inner, utt_timing, spk_idx,
            max_collapsed_seq_len=max_collapsed_seq_len,
            phone_class_lookup=phone_class_lookup,
        )


class DialogChunkEventDataset(_DialogChunkDatasetBase):
    """Event2StreamDataset + per-utt timing collapsed seq (= chunk_id 経由)。"""
    def __init__(
        self,
        event_manifest_path: str,
        utt_timing_jsonl_paths: list[str],
        *,
        text_tokenizer,
        max_text_len: int = 64,
        max_n_frames: int | None = None,
        soft_radius: int = 5,
        max_collapsed_seq_len: int = 512,
        phone_class_lookup: torch.Tensor | None = None,
        speaker_to_idx: dict[str, int] | None = None,
    ):
        inner = Event2StreamDataset(
            event_manifest_path,
            text_tokenizer=text_tokenizer, max_text_len=max_text_len,
            max_n_frames=max_n_frames, speaker_balanced=False,
            soft_phone=True, soft_boundary_radius=soft_radius,
        )
        utt_timing = _load_utt_timing_records(utt_timing_jsonl_paths)
        spk_idx = speaker_to_idx or load_default_speaker_to_idx() or _build_speaker_to_idx(utt_timing)
        super().__init__(
            inner, utt_timing, spk_idx,
            max_collapsed_seq_len=max_collapsed_seq_len,
            phone_class_lookup=phone_class_lookup,
        )




def collate_chunks(batch: list[dict]) -> dict:
    """collate_eventmix + collapsed seq padding。"""
    chunk_fields = ["collapsed_phone_ids", "collapsed_phone_class", "collapsed_utt_boundary",
                  "collapsed_speaker_per_phone", "collapsed_channel_per_phone",
                  "collapsed_phone_mask", "utt_token_ranges",
                  "utt_gt_start_frames", "utt_gt_end_frames",
                  "utt_gt_phone_durations",
                  "speaker_id_A", "speaker_id_B"]
    chunk_stash = {f: [] for f in chunk_fields}
    cleaned = []
    for item in batch:
        c = {k: v for k, v in item.items() if k not in chunk_fields}
        cleaned.append(c)
        for f in chunk_fields:
            chunk_stash[f].append(item.get(f))

    out = collate_eventmix(cleaned)

    B = len(batch)
    N_max = max(t.shape[0] for t in chunk_stash["collapsed_phone_ids"])
    for k in ["collapsed_phone_ids", "collapsed_phone_class", "collapsed_utt_boundary",
              "collapsed_speaker_per_phone", "collapsed_channel_per_phone"]:
        t = torch.zeros(B, N_max, dtype=torch.long)
        for i, ten in enumerate(chunk_stash[k]):
            t[i, :ten.shape[0]] = ten
        out[k] = t
    mask = torch.zeros(B, N_max, dtype=torch.bool)
    for i, m in enumerate(chunk_stash["collapsed_phone_mask"]):
        mask[i, :m.shape[0]] = m
    out["collapsed_phone_mask"] = mask
    out["utt_token_ranges"] = chunk_stash["utt_token_ranges"]
    out["utt_gt_start_frames"] = chunk_stash["utt_gt_start_frames"]
    out["utt_gt_end_frames"] = chunk_stash["utt_gt_end_frames"]
    out["utt_gt_phone_durations"] = chunk_stash["utt_gt_phone_durations"]
    out["speaker_id_A"] = torch.tensor(chunk_stash["speaker_id_A"], dtype=torch.long)
    out["speaker_id_B"] = torch.tensor(chunk_stash["speaker_id_B"], dtype=torch.long)
    return out


__all__ = [
    "DialogChunkDataset", "DialogChunkEventDataset",
    "collate_chunks", "_build_speaker_to_idx",
]

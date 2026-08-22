"""2-stream dialogue chunk dataset (+ optional phone-soft features)。

phone-soft feature は config (phone_condition.mode == "soft") で有効化、 __getitem__ で
runtime 生成 (= build_soft_phone_features を A/B それぞれに適用)。 hard mode では一切作らない。

設計:
  - Normal2StreamDataset と Event2StreamDataset を別 class とし、 batch dict 形式を統一
  - 混合は train スクリプト側で separate loader を交互/確率的に消費 (DDP 一貫性)
  - 統一 batch dict:
    {
      sample_kind: list[str]    # "normal" or "event"
      event_type:  list[str]    # event のみ、 normal は "normal"
      latent_A/B:  (B, T, 32)
      phone_A/B:   (B, T) long
      activity_A/B:(B, T) float
      ref_latent_A/B + ref_mask_A/B
      text_A_input_ids + text_A_mask, text_B_*
      core_mask:   (B, T) bool  # event は core 範囲のみ True、 normal は全 True
      latent_mask: (B, T) bool  # padding 除外
      chunk_id, event_id: list
      speech_ratio_A/B (normal のみ正値、 event は近似)
    }

normal 経路は legacy 2-stream dataset と論理同等、 core_mask を全 True で追加するのみ。
event 経路は event manifest の .pt をロード。
"""

from __future__ import annotations

import json
import random as _random
from bisect import bisect_left
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

# import soft phone feature builder
try:
    from kaburi_tts.two_stream.loss import build_soft_phone_features as _build_soft_phone_features
except ImportError:
    _build_soft_phone_features = None


# ---------------------------------------------------------------------------
# speaker-balanced resampling helpers
# ---------------------------------------------------------------------------

def _filter_excluded_dialogues(entries: list[dict], exclude: set[str] | None) -> list[dict]:
    if not exclude:
        return entries
    return [e for e in entries if str(e.get("dialogue_id", "")) not in exclude]


def _speaker_weights(entries: list[dict]) -> list[float]:
    """各 entry に対し (1/freq(spk_A) + 1/freq(spk_B))/2 を返す。 W03 等の頻出 spk を抑える。

    legacy compat: speaker_A/B (= new) も spk_A/B (= legacy) も accept。
    """
    def _get_spk(e: dict, key_new: str, key_old: str) -> str:
        return e.get(key_new) or e.get(key_old, "?")

    spk_count: Counter = Counter()
    for e in entries:
        spk_count[_get_spk(e, "speaker_A", "spk_A")] += 1
        spk_count[_get_spk(e, "speaker_B", "spk_B")] += 1
    out: list[float] = []
    for e in entries:
        a = spk_count[_get_spk(e, "speaker_A", "spk_A")] or 1
        b = spk_count[_get_spk(e, "speaker_B", "spk_B")] or 1
        out.append(0.5 * (1.0 / a + 1.0 / b))
    return out


def _resampled_index_map(weights: list[float], epoch_size: int, seed: int) -> list[int]:
    """weights に比例した確率で epoch_size 個の (with-replacement) index を deterministic に返す。"""
    if not weights:
        return []
    total = sum(weights)
    if total <= 0:
        return list(range(min(epoch_size, len(weights))))
    cum = []
    c = 0.0
    for w in weights:
        c += w / total
        cum.append(c)
    rng = _random.Random(int(seed))
    out: list[int] = []
    for _ in range(int(epoch_size)):
        r = rng.random()
        i = bisect_left(cum, r)
        if i >= len(cum):
            i = len(cum) - 1
        out.append(i)
    return out


class Normal2StreamDataset(Dataset):
    """dialogue chunk dataset (core_mask=all True + sample_kind="normal")。

    137h スケール用 オプション:
      exclude_dialogues: dialogue_id 集合 (ファイル取り違え等を除外)
      speaker_balanced : True で per-entry inverse-freq weight に基づき resample
      epoch_size       : speaker_balanced=True のとき 1 epoch の virtual sample 数 (default = len(entries))
      seed             : 再現性
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        text_tokenizer,
        max_text_len: int = 64,
        latent_T: int = 750,
        exclude_dialogues: set[str] | list[str] | None = None,
        speaker_balanced: bool = False,
        epoch_size: int | None = None,
        seed: int = 42,
        soft_phone: bool = False,                  # True で phone-soft feature 生成
        soft_boundary_radius: int = 5,
    ) -> None:
        super().__init__()
        raw = [json.loads(l) for l in Path(manifest_path).read_text().splitlines() if l.strip()]
        # 相対パスは manifest のあるディレクトリ基準で解決 (= 可搬な ref pack 用)
        mdir = Path(manifest_path).resolve().parent
        for e in raw:
            for key in ("chunk_path", "ref_A_path", "ref_B_path"):
                p = e.get(key)
                if p and not Path(p).is_absolute():
                    e[key] = str(mdir / p)
        if exclude_dialogues:
            exclude_set = set(exclude_dialogues)
            n_before = len(raw)
            raw = _filter_excluded_dialogues(raw, exclude_set)
            print(f"[Normal2StreamDataset] excluded dialogues: {n_before - len(raw)} chunks dropped "
                  f"(from {n_before} → {len(raw)})", flush=True)
        self.entries = raw
        self.text_tokenizer = text_tokenizer
        self.max_text_len = int(max_text_len)
        self.latent_T = int(latent_T)
        self.soft_phone = bool(soft_phone)
        self.soft_boundary_radius = int(soft_boundary_radius)
        self._ref_cache: dict[str, torch.Tensor] = {}
        if speaker_balanced and self.entries:
            weights = _speaker_weights(self.entries)
            self._index_map = _resampled_index_map(
                weights, epoch_size or len(self.entries), seed,
            )
            uniq = len(set(self._index_map))
            print(f"[Normal2StreamDataset] speaker-balanced: epoch_size={len(self._index_map)} "
                  f"unique_entries_per_epoch={uniq} (raw={len(self.entries)})", flush=True)
        else:
            self._index_map = list(range(len(self.entries)))

    def __len__(self) -> int:
        return len(self._index_map)

    def _load_ref(self, ref_path: str) -> torch.Tensor:
        cached = self._ref_cache.get(ref_path)
        if cached is None:
            d = torch.load(ref_path, map_location="cpu", weights_only=True)
            # legacy compat: ref .pt の field 名は "latent"、 legacy era は "ref_latent"
            cached = d.get("ref_latent", d.get("latent"))
            assert cached is not None, f"ref latent not found in {ref_path}"
            self._ref_cache[ref_path] = cached
        return cached

    def _pad_to_T(self, x: torch.Tensor, T: int, pad_value: float = 0.0) -> torch.Tensor:
        if x.shape[0] >= T:
            return x[:T]
        pad_len = T - x.shape[0]
        if x.dim() == 1:
            pad = x.new_full((pad_len,), pad_value)
        else:
            pad = x.new_full((pad_len,) + x.shape[1:], pad_value)
        return torch.cat([x, pad], dim=0)

    def __getitem__(self, idx: int) -> dict:
        e = self.entries[self._index_map[idx]]
        c = torch.load(e["chunk_path"], map_location="cpu", weights_only=True)
        T = self.latent_T
        latent_A = self._pad_to_T(c["latent_A"].float(), T)
        latent_B = self._pad_to_T(c["latent_B"].float(), T)
        phone_A = self._pad_to_T(c["phone_A"].long(), T, pad_value=1)
        phone_B = self._pad_to_T(c["phone_B"].long(), T, pad_value=1)
        activity_A = self._pad_to_T(c["activity_A"].float(), T)
        activity_B = self._pad_to_T(c["activity_B"].float(), T)
        # legacy compat: manifest field 名は ref_A_path、 legacy era は ref_A
        ref_A = self._load_ref(e.get("ref_A_path", e.get("ref_A"))).float()
        ref_B = self._load_ref(e.get("ref_B_path", e.get("ref_B"))).float()
        # text: text_scale=0.0 想定なので marker のみ (model が無視する)
        # legacy compat: speaker_A (new) → spk_A (legacy) fallback
        spk_A = c.get("speaker_A") or c.get("spk_A") or e.get("speaker_A") or e.get("spk_A", "A")
        spk_B = c.get("speaker_B") or c.get("spk_B") or e.get("speaker_B") or e.get("spk_B", "B")
        text_A_ids = self.text_tokenizer.encode(f"[{spk_A}]")[: self.max_text_len].long()
        text_B_ids = self.text_tokenizer.encode(f"[{spk_B}]")[: self.max_text_len].long()
        item = {
            "sample_kind": "normal",
            "event_type": "normal",
            "chunk_id": c.get("chunk_id", e.get("chunk_id", "")),
            "event_id": "",
            "latent_A": latent_A, "latent_B": latent_B,
            "phone_A": phone_A, "phone_B": phone_B,
            "activity_A": activity_A, "activity_B": activity_B,
            "ref_latent_A": ref_A, "ref_latent_B": ref_B,
            "text_A_input_ids": text_A_ids, "text_B_input_ids": text_B_ids,
            "core_mask": torch.ones(T, dtype=torch.bool),
            "speech_ratio_A": float(c.get("speech_ratio_A", 0.0)),
            "speech_ratio_B": float(c.get("speech_ratio_B", 0.0)),
        }
        if self.soft_phone and _build_soft_phone_features is not None:
            for ch, ph in [("A", phone_A), ("B", phone_B)]:
                feats = _build_soft_phone_features(ph, boundary_radius=self.soft_boundary_radius)
                for k, v in feats.items():
                    item[f"{k}_{ch}"] = v
        return item


class Event2StreamDataset(Dataset):
    """event manifest の .pt を read し、 normal と同じ batch dict 形式で返す。

    137h スケール用 オプション (Normal2StreamDataset と同じ意味):
      exclude_dialogues, speaker_balanced, epoch_size, seed
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        text_tokenizer,
        max_text_len: int = 64,
        max_n_frames: int | None = None,
        exclude_dialogues: set[str] | list[str] | None = None,
        speaker_balanced: bool = False,
        epoch_size: int | None = None,
        seed: int = 42,
        soft_phone: bool = False,
        soft_boundary_radius: int = 5,
    ) -> None:
        super().__init__()
        raw = [json.loads(l) for l in Path(manifest_path).read_text().splitlines() if l.strip()]
        if exclude_dialogues:
            exclude_set = set(exclude_dialogues)
            n_before = len(raw)
            raw = _filter_excluded_dialogues(raw, exclude_set)
            print(f"[Event2StreamDataset] excluded dialogues: {n_before - len(raw)} events dropped "
                  f"(from {n_before} → {len(raw)})", flush=True)
        self.entries = raw
        self.text_tokenizer = text_tokenizer
        self.max_text_len = int(max_text_len)
        self.max_n_frames = max_n_frames
        self.soft_phone = bool(soft_phone)
        self.soft_boundary_radius = int(soft_boundary_radius)
        self._ref_cache: dict[str, torch.Tensor] = {}
        if speaker_balanced and self.entries:
            weights = _speaker_weights(self.entries)
            self._index_map = _resampled_index_map(
                weights, epoch_size or len(self.entries), seed,
            )
            uniq = len(set(self._index_map))
            print(f"[Event2StreamDataset] speaker-balanced: epoch_size={len(self._index_map)} "
                  f"unique_entries_per_epoch={uniq} (raw={len(self.entries)})", flush=True)
        else:
            self._index_map = list(range(len(self.entries)))

    def __len__(self) -> int:
        return len(self._index_map)

    def _load_ref(self, ref_path: str) -> torch.Tensor:
        cached = self._ref_cache.get(ref_path)
        if cached is None:
            d = torch.load(ref_path, map_location="cpu", weights_only=True)
            # legacy compat: ref .pt の field 名は "latent"、 legacy era は "ref_latent"
            cached = d.get("ref_latent", d.get("latent"))
            assert cached is not None, f"ref latent not found in {ref_path}"
            self._ref_cache[ref_path] = cached
        return cached

    def __getitem__(self, idx: int) -> dict:
        e = self.entries[self._index_map[idx]]
        ev = torch.load(e["event_path"], map_location="cpu", weights_only=True)
        latent_A = ev["latent_A"].float()
        latent_B = ev["latent_B"].float()
        phone_A = ev["phone_A"].long()
        phone_B = ev["phone_B"].long()
        activity_A = ev["activity_A"].float()
        activity_B = ev["activity_B"].float()
        core_mask = ev["core_mask"].bool()
        if self.max_n_frames is not None and latent_A.shape[0] > self.max_n_frames:
            latent_A = latent_A[: self.max_n_frames]; latent_B = latent_B[: self.max_n_frames]
            phone_A = phone_A[: self.max_n_frames]; phone_B = phone_B[: self.max_n_frames]
            activity_A = activity_A[: self.max_n_frames]; activity_B = activity_B[: self.max_n_frames]
            core_mask = core_mask[: self.max_n_frames]
        # legacy compat: manifest field 名は ref_A_path、 legacy era は ref_A
        ref_A = self._load_ref(e.get("ref_A_path", e.get("ref_A"))).float()
        ref_B = self._load_ref(e.get("ref_B_path", e.get("ref_B"))).float()
        # legacy compat: speaker_A (new) → spk_A (legacy) fallback
        spk_A = ev.get("speaker_A") or ev.get("spk_A") or e.get("speaker_A") or e.get("spk_A", "A")
        spk_B = ev.get("speaker_B") or ev.get("spk_B") or e.get("speaker_B") or e.get("spk_B", "B")
        text_A_ids = self.text_tokenizer.encode(f"[{spk_A}]")[: self.max_text_len].long()
        text_B_ids = self.text_tokenizer.encode(f"[{spk_B}]")[: self.max_text_len].long()
        item = {
            "sample_kind": "event",
            "event_type": ev.get("event_type", e.get("event_type", "")),
            "chunk_id": ev.get("chunk_id", e.get("chunk_id", "")),
            "event_id": ev.get("event_id", e.get("event_id", "")),
            "latent_A": latent_A, "latent_B": latent_B,
            "phone_A": phone_A, "phone_B": phone_B,
            "activity_A": activity_A, "activity_B": activity_B,
            "ref_latent_A": ref_A, "ref_latent_B": ref_B,
            "text_A_input_ids": text_A_ids, "text_B_input_ids": text_B_ids,
            "core_mask": core_mask,
            "speech_ratio_A": float(activity_A.mean()),
            "speech_ratio_B": float(activity_B.mean()),
        }
        if self.soft_phone and _build_soft_phone_features is not None:
            for ch, ph in [("A", phone_A), ("B", phone_B)]:
                feats = _build_soft_phone_features(ph, boundary_radius=self.soft_boundary_radius)
                for k, v in feats.items():
                    item[f"{k}_{ch}"] = v
        return item


def _pad_refs(refs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    B = len(refs); lens = [r.shape[0] for r in refs]
    T_max = max(lens) if lens else 1; D = refs[0].shape[-1]
    out = torch.zeros(B, T_max, D, dtype=torch.float32)
    msk = torch.zeros(B, T_max, dtype=torch.bool)
    for i, r in enumerate(refs):
        L = r.shape[0]; out[i, :L] = r; msk[i, :L] = True
    return out, msk


def _pad_1d_long(ids_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    B = len(ids_list); lens = [t.shape[0] for t in ids_list]
    L_max = max(lens) if lens else 1
    out = torch.zeros(B, L_max, dtype=torch.long)
    msk = torch.zeros(B, L_max, dtype=torch.bool)
    for i, t in enumerate(ids_list):
        L = t.shape[0]; out[i, :L] = t; msk[i, :L] = True
    return out, msk


def _pad_time(seqs: list[torch.Tensor], pad_value=0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """batch 内 T 最大に右 pad。 (B, T_max, ...) と (B, T_max) bool mask を返す。"""
    B = len(seqs); lens = [s.shape[0] for s in seqs]
    T_max = max(lens) if lens else 1
    if seqs[0].dim() == 1:
        out = seqs[0].new_full((B, T_max), pad_value)
    else:
        out = seqs[0].new_full((B, T_max) + seqs[0].shape[1:], pad_value)
    msk = torch.zeros(B, T_max, dtype=torch.bool)
    for i, s in enumerate(seqs):
        L = s.shape[0]; out[i, :L] = s; msk[i, :L] = True
    return out, msk


def collate_eventmix(batch: list[dict]) -> dict:
    """batch 内 T を統一 (pad-to-max)、 normal/event 混在対応。"""
    latent_A, latent_mask = _pad_time([b["latent_A"] for b in batch], pad_value=0.0)
    latent_B, _ = _pad_time([b["latent_B"] for b in batch], pad_value=0.0)
    phone_A, _ = _pad_time([b["phone_A"] for b in batch], pad_value=1)   # 1 = <sil>
    phone_B, _ = _pad_time([b["phone_B"] for b in batch], pad_value=1)
    activity_A, _ = _pad_time([b["activity_A"] for b in batch], pad_value=0.0)
    activity_B, _ = _pad_time([b["activity_B"] for b in batch], pad_value=0.0)
    core_mask, _ = _pad_time([b["core_mask"] for b in batch], pad_value=False)
    ref_latent_A, ref_mask_A = _pad_refs([b["ref_latent_A"] for b in batch])
    ref_latent_B, ref_mask_B = _pad_refs([b["ref_latent_B"] for b in batch])
    text_A_input_ids, text_A_mask = _pad_1d_long([b["text_A_input_ids"] for b in batch])
    text_B_input_ids, text_B_mask = _pad_1d_long([b["text_B_input_ids"] for b in batch])

    # soft phone features (もし存在すれば)
    soft_phone_keys: list[tuple[str, object]] = []
    if "phone_cur_A" in batch[0]:
        # long type keys
        for k_long in ["phone_cur", "phone_prev", "phone_next"]:
            soft_phone_keys.append((k_long, "long"))
        for k_f in ["phone_pos_frac", "phone_dur_norm", "dist_start_norm", "dist_end_norm", "is_phone_boundary"]:
            soft_phone_keys.append((k_f, "float"))
    soft_out: dict[str, torch.Tensor] = {}
    for key, dtype_kind in soft_phone_keys:
        for ch in ("A", "B"):
            field = f"{key}_{ch}"
            pad_v = 1 if dtype_kind == "long" else 0.0
            t, _ = _pad_time([b[field] for b in batch], pad_value=pad_v)
            soft_out[field] = t

    return {
        "sample_kind": [b["sample_kind"] for b in batch],
        "event_type": [b["event_type"] for b in batch],
        "chunk_id": [b["chunk_id"] for b in batch],
        "event_id": [b["event_id"] for b in batch],
        "latent_A": latent_A, "latent_B": latent_B,
        "phone_A": phone_A, "phone_B": phone_B,
        "activity_A": activity_A, "activity_B": activity_B,
        "ref_latent_A": ref_latent_A, "ref_mask_A": ref_mask_A,
        "ref_latent_B": ref_latent_B, "ref_mask_B": ref_mask_B,
        "text_A_input_ids": text_A_input_ids, "text_A_mask": text_A_mask,
        "text_B_input_ids": text_B_input_ids, "text_B_mask": text_B_mask,
        "core_mask": core_mask, "latent_mask": latent_mask,
        "speech_ratio_A": torch.tensor([b["speech_ratio_A"] for b in batch]),
        "speech_ratio_B": torch.tensor([b["speech_ratio_B"] for b in batch]),
        **soft_out,    # phone-soft features (空 dict なら hard mode)
    }

"""GT 4-state frame label builder + training set からの class weight 計算。

各 frame の A/B 活動状態を 4 class に分類:
  0: none   (= A silent, B silent)
  1: A_only (= A active, B silent)
  2: B_only (= A silent, B active)
  3: both   (= A active, B active)

class weight は inverse square-root frequency + mean normalize + clip(0.5, 5.0)。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from kaburi_tts.predictor import (
    STATE_NONE, STATE_A_ONLY, STATE_B_ONLY, STATE_BOTH, N_STATES,
)


def build_gt_frame_state_label(utts: list, T: int) -> torch.Tensor:
    """utts (= utt-timing record の utterances) から frame-level 4-state label を構築。

    Returns:
      [T] long、 各 frame の class (= 0:none, 1:A_only, 2:B_only, 3:both)
    """
    A_active = torch.zeros(T, dtype=torch.bool)
    B_active = torch.zeros(T, dtype=torch.bool)
    for u in utts:
        ch = u.get("speaker", "A")
        s = max(0, int(u.get("gt_start_frame", 0)))
        e = min(T, int(u.get("gt_end_frame", 0)))
        if e <= s:
            continue
        if ch == "A":
            A_active[s:e] = True
        else:
            B_active[s:e] = True

    state = torch.zeros(T, dtype=torch.long)
    state[A_active & ~B_active] = STATE_A_ONLY
    state[~A_active & B_active] = STATE_B_ONLY
    state[A_active & B_active] = STATE_BOTH
    return state


def count_class_frames_in_jsonl(
    jsonl_paths: list[str], T: int = 750, max_chunks: int | None = None
) -> torch.Tensor:
    """utt-timing jsonl 群から、 各 4-state class の累積 frame 数を集計。

    Returns:
      [4] long、 各 class の総 frame 数 (= 全 chunks 合算)
    """
    counts = torch.zeros(N_STATES, dtype=torch.long)
    seen = 0
    for jp in jsonl_paths:
        p = Path(jp)
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                utts = r.get("utterances", [])
                state = build_gt_frame_state_label(utts, T)
                for c in range(N_STATES):
                    counts[c] += int((state == c).sum().item())
                seen += 1
                if max_chunks is not None and seen >= max_chunks:
                    return counts
    return counts


def compute_class_weight(
    counts: torch.Tensor,
    *,
    eps: float = 1e-6,
    clip_min: float = 0.5,
    clip_max: float = 5.0,
) -> torch.Tensor:
    """inverse square-root frequency + mean normalize + clip。

    Args:
      counts: [N_STATES] long、 各 class の frame 数
    Returns:
      [N_STATES] float、 frame-level CE に渡す weight
    """
    counts_f = counts.float()
    total = counts_f.sum().clamp_min(1.0)
    freq = counts_f / total
    raw = 1.0 / torch.sqrt(freq + eps)
    w = raw / raw.mean().clamp_min(eps)
    w = w.clamp(clip_min, clip_max)
    return w


def load_or_compute_class_weight(
    weight_path: str | None,
    jsonl_paths: list[str],
    *,
    T: int = 750,
    max_chunks: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """precomputed JSON file があれば load、 なければ jsonl から compute して保存。

    Returns:
      (weight [4] float, counts [4] long)
    """
    if weight_path is not None and Path(weight_path).exists():
        d = json.loads(Path(weight_path).read_text())
        counts = torch.tensor(d["counts"], dtype=torch.long)
        weight = torch.tensor(d["weight"], dtype=torch.float32)
        return weight, counts

    counts = count_class_frames_in_jsonl(jsonl_paths, T=T, max_chunks=max_chunks)
    weight = compute_class_weight(counts)

    if weight_path is not None:
        Path(weight_path).parent.mkdir(parents=True, exist_ok=True)
        Path(weight_path).write_text(json.dumps({
            "counts": counts.tolist(),
            "freq": (counts.float() / counts.float().sum().clamp_min(1.0)).tolist(),
            "weight": weight.tolist(),
            "T": T,
            "max_chunks": max_chunks,
        }, indent=2))
    return weight, counts


def load_or_compute_class_weight_for_chunk_ids(
    weight_path: str | None,
    *,
    utt_timing: dict,
    chunk_ids: list[str],
    T: int = 750,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load or compute 4-state class weights for a specific split.

    The public timing predictor is trained on dialogue-level split manifests, so
    class weights should be derived from the same train split rather than from
    all available utt_timing records.
    """
    if weight_path is not None:
        from kaburi_tts.two_stream.loss import _resolve_repo_path
        weight_path = str(_resolve_repo_path(weight_path))
    if weight_path is not None and Path(weight_path).exists():
        d = json.loads(Path(weight_path).read_text())
        counts = torch.tensor(d["counts"], dtype=torch.long)
        weight = torch.tensor(d["weight"], dtype=torch.float32)
        return weight, counts

    counts = torch.zeros(N_STATES, dtype=torch.long)
    for cid in chunk_ids:
        rec = utt_timing.get(str(cid))
        if rec is None:
            continue
        state = build_gt_frame_state_label(rec.get("utterances", []), T)
        for c in range(N_STATES):
            counts[c] += int((state == c).sum().item())
    weight = compute_class_weight(counts)

    if weight_path is not None:
        Path(weight_path).parent.mkdir(parents=True, exist_ok=True)
        Path(weight_path).write_text(json.dumps({
            "counts": counts.tolist(),
            "freq": (counts.float() / counts.float().sum().clamp_min(1.0)).tolist(),
            "weight": weight.tolist(),
            "T": T,
            "source": "split_chunk_ids",
            "n_chunk_ids": len(chunk_ids),
        }, indent=2))
    return weight, counts


__all__ = [
    "build_gt_frame_state_label",
    "count_class_frames_in_jsonl",
    "compute_class_weight",
    "load_or_compute_class_weight",
    "load_or_compute_class_weight_for_chunk_ids",
]

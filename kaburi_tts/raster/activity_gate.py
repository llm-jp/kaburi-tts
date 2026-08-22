"""完全無発話区間へ漏れた音声断片を抑制する raster 由来の出力側ゲート。

acoustic checkpoint は変更しない。union された活動区間とその保護 padding 内の
波形は bit 単位で不変で、外側だけを線形 fade で落とす。
"""

from __future__ import annotations

import numpy as np


def union_activity_envelope(
    activity: np.ndarray,
    num_samples: int,
    sample_rate: int,
    *,
    fps: int = 25,
    pad_frames: int = 2,
    fade_ms: float = 40.0,
) -> np.ndarray:
    """Return a sample-rate envelope that is one throughout all active speech.

    Activity is unioned across speakers.  Active runs are protected by
    ``pad_frames`` on both sides and only the exterior receives a linear fade.
    Consequently samples aligned to any active raster frame are bit-preserved.
    """
    activity = np.asarray(activity)
    if activity.ndim == 3 and activity.shape[0] == 1:
        activity = activity[0]
    if activity.ndim != 2:
        raise ValueError(f"activity must have shape [C,T] or [1,C,T], got {activity.shape}")
    if sample_rate % fps:
        raise ValueError(f"sample_rate {sample_rate} must be divisible by fps {fps}")
    union = np.any(activity > 0.5, axis=0)
    samples_per_frame = sample_rate // fps
    envelope = np.zeros(num_samples, dtype=np.float32)
    padded = np.r_[False, union, False].astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    fade = max(0, int(round(sample_rate * fade_ms / 1000.0)))
    pad = max(0, int(pad_frames)) * samples_per_frame
    for frame_start, frame_end in zip(edges[::2], edges[1::2]):
        core_start = frame_start * samples_per_frame
        core_end = min(num_samples, frame_end * samples_per_frame)
        protected_start = max(0, core_start - pad)
        protected_end = min(num_samples, core_end + pad)
        envelope[protected_start:protected_end] = 1.0
        if fade:
            left = max(0, protected_start - fade)
            if protected_start > left:
                ramp = np.linspace(0.0, 1.0, protected_start - left, endpoint=False, dtype=np.float32)
                envelope[left:protected_start] = np.maximum(envelope[left:protected_start], ramp)
            right = min(num_samples, protected_end + fade)
            if right > protected_end:
                ramp = np.linspace(1.0, 0.0, right - protected_end, endpoint=False, dtype=np.float32)
                envelope[protected_end:right] = np.maximum(envelope[protected_end:right], ramp)
    return envelope


def suppress_fully_inactive(
    audio: np.ndarray,
    activity: np.ndarray,
    sample_rate: int,
    *,
    fps: int = 25,
    pad_frames: int = 2,
    fade_ms: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the union-activity envelope to mono or channels-last audio."""
    audio = np.asarray(audio)
    if audio.ndim not in (1, 2):
        raise ValueError(f"audio must be mono or [N,C], got {audio.shape}")
    envelope = union_activity_envelope(
        activity, audio.shape[0], sample_rate, fps=fps,
        pad_frames=pad_frames, fade_ms=fade_ms,
    )
    gated = audio * (envelope if audio.ndim == 1 else envelope[:, None])
    return gated.astype(audio.dtype, copy=False), envelope


__all__ = ["suppress_fully_inactive", "union_activity_envelope"]


def gate_channels_first(wav, activity, sample_rate, *, fps: int = 25,
                        pad_frames: int = 2, fade_ms: float = 40.0):
    """torch の channels-first 波形 [C,N] に union activity gate を適用する。

    activity は [C,T] (25 fps の raster)。戻り値は (gated wav [C,N], envelope [N])。
    numpy 側は channels-last 契約なので、ここで転置を吸収する。
    """
    import torch
    if wav.ndim != 2:
        raise ValueError(f"wav must be [C,N] channels-first, got {tuple(wav.shape)}")
    audio = wav.transpose(0, 1).cpu().numpy()          # [N,C]
    act = np.asarray([a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
                      for a in activity])
    gated, envelope = suppress_fully_inactive(
        audio, act, sample_rate, fps=fps, pad_frames=pad_frames, fade_ms=fade_ms)
    out = torch.from_numpy(np.ascontiguousarray(gated)).transpose(0, 1).to(wav.dtype)
    return out, torch.from_numpy(envelope)


__all__ = ["suppress_fully_inactive", "union_activity_envelope", "gate_channels_first"]

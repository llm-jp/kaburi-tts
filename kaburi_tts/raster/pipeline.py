"""音素ラスタ生成の統合パイプライン: 対話テキスト → ラスタ → 音声。

realizer (実現形 + duration) → gap model (first start + any-gap) → 配置
(start_i = prev_any_end + gap_i) → 30 秒キャンバス fit (正比例 start 圧縮、
発話長は不変) → ラスタ (activity ≡ phone != sil。学習データ規約) →
既存 acoustic (無変更) で合成。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .models import MAX_UTTS, load_gap_model
from .structured import load_realizer

REPO = Path(__file__).resolve().parents[2]
T = 750
FIT_MARGIN = 5
SIL = 1

# release ごとの asset 束。既定は最新 release。
RELEASES = {
    "global-opt-20260820": {
        "decode": REPO / "assets/raster/decode_config.json",
        "realizer": REPO / "hf_upload/raster/global-opt-20260820/realizer.pt",
        "edit_prior": REPO / "hf_upload/raster/global-opt-20260820/edit_prior.pt",
        "gap": REPO / "hf_upload/raster/global-opt-20260820/gap.pt",
        "hf_realizer": "raster/global-opt-20260820/realizer.pt",
        "hf_edit_prior": "raster/global-opt-20260820/edit_prior.pt",
        "hf_gap": "raster/global-opt-20260820/gap.pt",
    },
}
DEFAULT_RELEASE = "global-opt-20260820"
DEFAULT_DECODE = RELEASES[DEFAULT_RELEASE]["decode"]
DEFAULT_REALIZER = RELEASES[DEFAULT_RELEASE]["realizer"]
DEFAULT_GAP = RELEASES[DEFAULT_RELEASE]["gap"]
HF_REALIZER = RELEASES[DEFAULT_RELEASE]["hf_realizer"]
HF_GAP = RELEASES[DEFAULT_RELEASE]["hf_gap"]


def _resolve_ckpt(local_path, hf_filename):
    p = Path(local_path)
    if p.exists():
        return str(p)
    from huggingface_hub import hf_hub_download
    from kaburi_tts import HF_REPO
    return hf_hub_download(repo_id=HF_REPO, filename=hf_filename)


class RasterGenerator:
    """CLI から使う高レベル API。モデルは初回のみロード。"""

    def __init__(self, device: str = "cpu", realizer_ckpt=None, gap_ckpt=None,
                 decode_json=None, release: str | None = None):
        rel = RELEASES[release or DEFAULT_RELEASE]
        self.release = release or DEFAULT_RELEASE
        decode_cfg = json.loads(Path(decode_json or rel["decode"]).read_text())
        self.decode_cfg = decode_cfg
        prior_path = None
        if decode_cfg.get("edit_verifier", {}).get("enabled") and rel.get("edit_prior"):
            # edit prior も realizer と同じく local -> HF の順で解決する
            prior_path = _resolve_ckpt(rel["edit_prior"], rel["hf_edit_prior"])
        self.realizer = load_realizer(
            _resolve_ckpt(realizer_ckpt or rel["realizer"], rel["hf_realizer"]),
            decode_cfg, device=device, prior_path=prior_path)
        self.gap_model = load_gap_model(
            _resolve_ckpt(gap_ckpt or rel["gap"], rel["hf_gap"]), device=device)
        self.device = device

    @torch.no_grad()
    def _predict_gaps(self, realized_utts, speakers):
        U = len(realized_utts)
        P = max(len(p) for p, _ in realized_utts)
        b = {
            "phones": torch.zeros(1, U, P, dtype=torch.long),
            "phone_mask": torch.zeros(1, U, P, dtype=torch.bool),
            "spk": torch.zeros(1, U, dtype=torch.long),
            "dur_total": torch.zeros(1, U),
            "utt_mask": torch.ones(1, U, dtype=torch.bool),
        }
        for j, ((phones, durs), spk) in enumerate(zip(realized_utts, speakers)):
            L = len(phones)
            b["phones"][0, j, :L] = torch.tensor(phones)
            b["phone_mask"][0, j, :L] = True
            b["spk"][0, j] = 0 if spk == "A" else 1
            b["dur_total"][0, j] = float(sum(durs))
        b = {k: v.to(self.device) for k, v in b.items()}
        return self.gap_model(b, teacher_gaps=None)["gap"][0].tolist()

    def timeline(self, utterances, chunk_id="dialog"):
        """utterances=[{'speaker','text'},...] → 配置済み発話リスト + メタ。"""
        pairs = [(u["speaker"], u["text"]) for u in utterances]
        realized = self.realizer.realize_chunk(pairs, chunk_id=chunk_id)
        keep = [(s, r) for (s, _t), r in zip(pairs, realized) if r is not None]
        if not keep:
            raise ValueError("有効な発話がありません (G2P 失敗 or 音素長超過)")
        if len(keep) > MAX_UTTS:
            raise ValueError(f"発話数が上限 {MAX_UTTS} を超えています: {len(keep)}")
        speakers = [s for s, _ in keep]
        realized_utts = [r for _, r in keep]
        gaps = self._predict_gaps(realized_utts, speakers)
        placed = []
        prev_any_end = None
        for (phones, durs), spk, gap in zip(realized_utts, speakers, gaps):
            start = max(0.0, gap) if prev_any_end is None else max(0.0, prev_any_end + gap)
            end = start + sum(durs)
            prev_any_end = end if prev_any_end is None else max(prev_any_end, end)
            placed.append(dict(speaker=spk, start=start, phones=phones, durs=durs))
        # canvas fit: start の正比例圧縮 (発話長・順序は不変)
        target = T - FIT_MARGIN
        max_end = max(u["start"] + sum(u["durs"]) for u in placed)
        scale = 1.0
        if max_end > target:
            for s in np.linspace(1.0, 0.2, 81):
                if all(u["start"] * s + sum(u["durs"]) <= target for u in placed):
                    scale = s
                    break
            for u in placed:
                u["start"] *= scale
        return placed, {"fit_scale": scale, "n_utts": len(placed)}

    @staticmethod
    def rasterize(placed):
        """activity ≡ (phone != sil)。発話内 sil を active に塗ると分布外 (学習データ検証済)。"""
        phone = {ch: np.ones(T, dtype=np.int64) for ch in "AB"}
        act = {ch: np.zeros(T, dtype=np.float32) for ch in "AB"}
        for u in placed:
            ch = u["speaker"]
            cursor = float(u["start"])
            for p, d in zip(u["phones"], u["durs"]):
                s = int(round(cursor))
                e = int(round(cursor + d))
                cursor += d
                if s >= T:
                    break
                phone[ch][s:min(e, T)] = p
                if p != SIL:
                    act[ch][s:min(e, T)] = 1.0
        return (torch.from_numpy(phone["A"]).long(), torch.from_numpy(phone["B"]).long(),
                torch.from_numpy(act["A"]).float(), torch.from_numpy(act["B"]).float())

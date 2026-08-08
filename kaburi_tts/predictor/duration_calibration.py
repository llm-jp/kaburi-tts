"""MFA 制約付き posterior calibration decoder (duration 較正)。

timing predictor の phone duration 分布 (logits) を破棄せず、境界文脈ごとの scalar
exponential tilt を加えてから期待値デコードする:

    q_i(d) = softmax(z_i(d) + lambda(c_i) * v_d),   v_d = 1..30
    duration_i = sum_d q_i(d) * v_d

lambda>0 は長い bin へ、lambda<0 は短い bin へ確率質量を移す。元分布への KL を
最小に保ちながら MFA duration moment に合わせる I-projection。ordinary 文脈は
lambda=0 固定で境界外を変えない。係数は正式 split の凸最適化で推定し、デモ台本
固有情報は使わない (CLAUDE_CODE_MFA_DURATION_CALIBRATION.md, 2026-07-19)。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

SIL_ID = 1
N_BINS = 30
BIN_VALUES = torch.arange(1, N_BINS + 1, dtype=torch.float32)

# 優先順位順 (先勝ち)。仕様 §6: sil 系 3 文脈を発話末条件より優先する。
CONTEXTS = [
    "internal_sil",
    "pre_internal_sil_1",
    "post_internal_sil_1",
    "pre_internal_sil_2",
    "utterance_final_1",
    "utterance_final_2",
    "ordinary",
]


def classify_contexts(phone_ids: list[int]) -> list[str]:
    """1 発話の音素 id 列 (内部 sil 含む) に対する PHONE トークン文脈ラベル。

    internal sil = 端以外の SIL_ID。優先順位は CONTEXTS の順で決定的に分類する。
    """
    n = len(phone_ids)
    is_int_sil = [pid == SIL_ID and 0 < i < n - 1 for i, pid in enumerate(phone_ids)]
    out = []
    for i in range(n):
        if is_int_sil[i]:
            out.append("internal_sil")
        elif i + 1 < n and is_int_sil[i + 1]:
            out.append("pre_internal_sil_1")
        elif i - 1 >= 0 and is_int_sil[i - 1]:
            out.append("post_internal_sil_1")
        elif i + 2 < n and is_int_sil[i + 2]:
            out.append("pre_internal_sil_2")
        elif i == n - 1:
            out.append("utterance_final_1")
        elif i == n - 2:
            out.append("utterance_final_2")
        else:
            out.append("ordinary")
    return out


def file_sha1(path: str | Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


# 発話長ビン (発話レベルのテンポ系統バイアス補正、全 phone への加法 λ)。
# 実測 (valid_seen 400ch): MFA/pred 速度比 = 1-8ph: 1.21 / 9-15: 1.07 / 16-25: 0.99 /
# 26-40: 0.92 / 41+: 0.92 — 短発話は速すぎ・長発話は遅すぎの平均回帰。
# 基準ビン 16-25 は lambda=0 固定。
LENGTH_BINS = [(1, 8), (9, 15), (16, 25), (26, 40), (41, 10_000)]
LENGTH_REF_BIN = "16-25"


def length_bin_name(n_phones: int) -> str:
    for a, b in LENGTH_BINS:
        if a <= n_phones <= b:
            return f"{a}-{b}" if b < 10_000 else f"{a}+"
    return LENGTH_REF_BIN


def load_calibration(path: str | Path, *, predictor_id: str | None = None,
                     phone_vocab_hash: str | None = None) -> dict:
    """較正資産の load + 厳格 validate。識別子不一致は黙って適用せずエラー。"""
    obj = json.loads(Path(path).read_text())
    if obj.get("format_version") != 1:
        raise ValueError(f"calibration format_version 不一致: {obj.get('format_version')}")
    if predictor_id is not None and obj.get("predictor_id") not in (None, predictor_id):
        raise ValueError(f"calibration の predictor_id 不一致: {obj.get('predictor_id')} != {predictor_id}")
    if phone_vocab_hash is not None and obj.get("phone_vocab_hash") not in (None, phone_vocab_hash):
        raise ValueError("calibration の phone_vocab_hash 不一致")
    for c in obj.get("contexts", {}):
        if c not in CONTEXTS:
            raise ValueError(f"未知の context: {c}")
    if float(obj.get("contexts", {}).get("ordinary", {}).get("lambda", 0.0)) != 0.0:
        raise ValueError("ordinary の lambda は 0 固定")
    lb = obj.get("length_bins", {})
    if float(lb.get(LENGTH_REF_BIN, {}).get("lambda", 0.0)) != 0.0:
        raise ValueError(f"length bin {LENGTH_REF_BIN} の lambda は 0 固定 (基準ビン)")
    obj["_lambda_by_context"] = {c: float(v.get("lambda", 0.0))
                                 for c, v in obj.get("contexts", {}).items()}
    obj["_lambda_by_lenbin"] = {k: float(v.get("lambda", 0.0)) for k, v in lb.items()}
    return obj


def tilt_expected(phone_logits: torch.Tensor, contexts: list[str], calib: dict,
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """logits (N, 30) + 文脈 → (較正後期待値 (N,), 較正後 posterior 分散 (N,))。

    λ = λ_boundary(c_i) + λ_lenbin(発話長ビン)。lenbin は系列全体に加法適用。"""
    lam_len = calib.get("_lambda_by_lenbin", {}).get(length_bin_name(len(contexts)), 0.0)
    lam = torch.tensor([calib["_lambda_by_context"].get(c, 0.0) + lam_len for c in contexts],
                       dtype=torch.float32, device=phone_logits.device)
    v = BIN_VALUES.to(phone_logits.device)
    q = torch.softmax(phone_logits.float() + lam[:, None] * v[None, :], dim=-1)
    exp = (q * v[None, :]).sum(-1)
    var = (q * v[None, :] ** 2).sum(-1) - exp ** 2
    return exp, var


def project_total(durations: torch.Tensor, contexts: list[str], variances: torch.Tensor,
                  target_total: float) -> tuple[torch.Tensor, float]:
    """発話 total を target に合わせる projection。ordinary は動かさず、境界 token を
    posterior 分散重みで動かす。最低 1 frame。返り値 (新 durations, 未達 frame 数)。"""
    idx = [i for i, c in enumerate(contexts) if c != "ordinary"]
    cur = float(durations.sum())
    delta = target_total - cur
    if not idx or abs(delta) < 1e-6:
        return durations, (0.0 if idx else abs(delta))
    out = durations.clone()
    w = variances[idx].clamp(min=1e-6)
    w = w / w.sum()
    remaining = delta
    # 減算方向は min 1 frame 制約で複数回に分けて配分する
    for _ in range(4):
        if abs(remaining) < 1e-6:
            break
        movable = torch.tensor(idx)
        alloc = w * remaining
        new = out[movable] + alloc
        clipped = new.clamp(min=1.0)
        applied = clipped - out[movable]
        out[movable] = clipped
        remaining -= float(applied.sum())
        headroom = (out[movable] > 1.0).float()
        if headroom.sum() == 0:
            break
        w = (variances[idx].clamp(min=1e-6) * headroom)
        if float(w.sum()) <= 0:
            break
        w = w / w.sum()
    return out, abs(remaining)

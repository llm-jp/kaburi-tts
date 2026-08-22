"""データ由来の局所操作事前分布 (one-sided edit verifier 用、推論のみ)。

学習データの canonical token phone pattern から「その位置でどの操作が実際に
起きたか」の頻度を数えた asset。表層文字列・POS・人手の語彙リストは使わない
(asset の feature_contract がそれを明示する)。

Tower1 の提案 (non-KEEP) ごとに
    combined = model log(P(proposal)/P(KEEP)) + w * prior log-odds
を計算し、threshold 未満なら KEEP へ戻す。新しい編集は決して作らない。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

FORMAT_INNER = "empirical_edit_prior_v1"
FORMAT_OUTER = "tower1_empirical_edit_prior_v1"


def _position_category(position: int, length: int) -> int:
    if length == 1:
        return 0
    if position == 0:
        return 1
    if position == length - 1:
        return 3
    return 2


@dataclass(frozen=True)
class EmpiricalEditPrior:
    operation_classes: int
    exact: dict
    window: dict
    shape: dict
    source: dict
    smoothing: float = 0.5
    minimum_exact_support: int = 20
    minimum_window_support: int = 100
    minimum_shape_support: int = 100

    def __post_init__(self) -> None:
        if self.operation_classes < 2:
            raise ValueError("operation_classes must include KEEP and DEL")
        if self.smoothing <= 0.0:
            raise ValueError("smoothing must be positive")

    @staticmethod
    def _support(counts) -> int:
        return sum(counts.values()) if counts else 0

    def counts_for(self, phones: Sequence[int], position: int):
        """exact -> window -> shape -> source の順に back off する。"""
        values = tuple(int(p) for p in phones)
        length = len(values)
        if not 0 <= position < length:
            raise ValueError("position is outside token phone sequence")
        exact = self.exact.get((values, position))
        support = self._support(exact)
        if support >= self.minimum_exact_support:
            return exact, support, "exact"
        previous = values[position - 1] if position else 0
        following = values[position + 1] if position + 1 < length else 0
        category = _position_category(position, length)
        window = self.window.get((previous, values[position], following, category))
        support = self._support(window)
        if support >= self.minimum_window_support:
            return window, support, "window"
        shape = self.shape.get((values[position], category, min(length, 8)))
        support = self._support(shape)
        if support >= self.minimum_shape_support:
            return shape, support, "shape"
        source = self.source.get(values[position], {0: 1})
        return source, self._support(source), "source"

    def proposal_log_odds(self, phones: Sequence[int], position: int, proposal: int):
        if not 1 <= proposal < self.operation_classes:
            raise ValueError("proposal must be a supported non-KEEP class")
        counts, support, level = self.counts_for(phones, position)
        numerator = counts.get(int(proposal), 0) + self.smoothing
        denominator = counts.get(0, 0) + self.smoothing
        return math.log(numerator / denominator), support, level

    @classmethod
    def from_dict(cls, payload: dict) -> "EmpiricalEditPrior":
        if payload.get("format") != FORMAT_INNER:
            raise ValueError(f"not an {FORMAT_INNER} payload")
        return cls(
            operation_classes=int(payload["operation_classes"]),
            exact=payload["exact"], window=payload["window"],
            shape=payload["shape"], source=payload["source"],
            smoothing=float(payload["smoothing"]),
            minimum_exact_support=int(payload["minimum_exact_support"]),
            minimum_window_support=int(payload["minimum_window_support"]),
            minimum_shape_support=int(payload["minimum_shape_support"]))


def load_edit_prior(path: str):
    """edit_prior checkpoint を読み、(prior, meta) を返す。format 不一致は fail fast。"""
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT_OUTER:
        raise ValueError(f"{FORMAT_OUTER} checkpoint ではありません: {path}")
    meta = {k: v for k, v in payload.items() if k != "prior"}
    return EmpiricalEditPrior.from_dict(payload["prior"]), meta


__all__ = ["EmpiricalEditPrior", "load_edit_prior", "FORMAT_INNER", "FORMAT_OUTER"]

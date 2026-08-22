"""structured deletion 版 realizer (global-opt-20260820 以降) の推論実装。

公開 realizer は各音素の編集操作を独立に予測する。本 module はトークン内の
KEEP/DEL の並びを線形鎖 CRF で捉え直し、**one-sided** に decode する:
公開 decoder が既に DEL と判定した位置を KEEP へ戻すことはできるが、
新しい DEL を増やすことは決してない。

学習コードは研究リポジトリ側にあり、ここには推論に必要な最小実装のみを置く。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

from kaburi_tts.g2p.dict_g2p import load_existing_dict
from .edit_prior import load_edit_prior
from .models import MAXLEN, build_joint_tagger
from .realize import REPO, JointRealizer, apply_edits

FORMAT = "tower1_structured_deletion_v1"


class LinearChainCRF(nn.Module):
    """KEEP/DEL の 2 状態線形鎖 CRF (推論のみ)。"""

    def __init__(self, states: int = 2) -> None:
        super().__init__()
        self.states = states
        self.start = nn.Parameter(torch.zeros(states))
        self.transition = nn.Parameter(torch.zeros(states, states))
        self.end = nn.Parameter(torch.zeros(states))

    @torch.no_grad()
    def viterbi(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        score = self.start + emissions[:, 0]
        history = []
        for index in range(1, emissions.shape[1]):
            candidate = score.unsqueeze(2) + self.transition.unsqueeze(0)
            best_score, best_state = candidate.max(1)
            next_score = best_score + emissions[:, index]
            score = torch.where(mask[:, index, None], next_score, score)
            history.append(best_state)
        state = (score + self.end).argmax(1)
        result = torch.zeros(mask.shape, dtype=torch.long, device=mask.device)
        lengths = mask.long().sum(1)
        result.scatter_(1, (lengths - 1)[:, None], state[:, None])
        for index in range(emissions.shape[1] - 1, 0, -1):
            active = lengths > index
            previous = history[index - 1].gather(1, state[:, None]).squeeze(1)
            result[:, index - 1] = torch.where(active, previous, result[:, index - 1])
            state = torch.where(active, previous, state)
        return result


class TokenPatternHead(nn.Module):
    """トークン局所 CRF の emission (公開 DEL 閾値を step-0 方策とする残差)。"""

    def __init__(self, hidden_dim: int, inner_dim: int = 128,
                 delete_threshold: float = 0.6) -> None:
        super().__init__()
        if not 0.0 < delete_threshold < 1.0:
            raise ValueError("delete_threshold must be in (0, 1)")
        self.hidden_dim = int(hidden_dim)
        self.inner_dim = int(inner_dim)
        self.delete_threshold = float(delete_threshold)
        self.network = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, inner_dim),
            nn.SiLU(),
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, 2),
        )

    def forward(self, phone_hidden, token_hidden, delete_log_odds):
        if phone_hidden.shape != token_hidden.shape:
            raise ValueError("phone and token hidden tensors must have equal shape")
        if delete_log_odds.shape != phone_hidden.shape[:2]:
            raise ValueError("delete_log_odds must have shape [tokens, phones]")
        residual = self.network(
            torch.cat([phone_hidden, token_hidden,
                       delete_log_odds.clamp(-12.0, 12.0).unsqueeze(-1) / 4.0], dim=-1))
        threshold_log_odds = math.log(self.delete_threshold / (1.0 - self.delete_threshold))
        base = torch.stack([torch.zeros_like(delete_log_odds),
                            delete_log_odds - threshold_log_odds], dim=-1)
        return base + residual


def _token_positions(ph_tok, n_tokens):
    result = [[] for _ in range(n_tokens)]
    for position, token in enumerate(ph_tok):
        if 0 <= int(token) < n_tokens:
            result[int(token)].append(position)
    return result


def pack_token_patterns(hidden, delete_log_odds, ph_tok, n_tokens,
                        max_token_phones: int = 32):
    """トークンごとの音素列を padding して束ねる (推論用)。"""
    positions_by_token = _token_positions(ph_tok, n_tokens)
    rows = [(token, positions) for token, positions in enumerate(positions_by_token)
            if positions and len(positions) <= max_token_phones]
    if not rows:
        return None
    width = max(len(r[1]) for r in rows)
    count = len(rows)
    dimension = hidden.shape[-1]
    phone = hidden.new_zeros(count, width, dimension)
    pooled = hidden.new_zeros(count, width, dimension)
    odds = delete_log_odds.new_zeros(count, width)
    mask = torch.zeros(count, width, dtype=torch.bool, device=hidden.device)
    keys = []
    for index, (token, positions) in enumerate(rows):
        position = torch.tensor(positions, dtype=torch.long, device=hidden.device)
        length = len(positions)
        values = hidden[0].index_select(0, position)
        phone[index, :length] = values
        pooled[index, :length] = values.mean(0, keepdim=True)
        odds[index, :length] = delete_log_odds[0].index_select(0, position)
        mask[index, :length] = True
        keys.append(token)
    return phone, pooled, odds, mask, keys, positions_by_token


def token_sequence_score(crf, emissions, tags, mask):
    """packed トークンごとの CRF 経路スコア (正規化なし)。"""
    score = crf.start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0, None]).squeeze(1)
    for index in range(1, emissions.shape[1]):
        active = mask[:, index]
        step = (crf.transition[tags[:, index - 1], tags[:, index]]
                + emissions[:, index].gather(1, tags[:, index, None]).squeeze(1))
        score = score + step * active
    lengths = mask.long().sum(1)
    last = tags.gather(1, (lengths - 1)[:, None]).squeeze(1)
    return score + crf.end[last]


@torch.no_grad()
def decode_one_sided(crf, emissions, base_tags, mask, *,
                     minimum_margin_per_phone: float = 0.0):
    """base_tags に無い DEL を決して追加しない制約付き decode。"""
    allowed_delete = base_tags.bool() & mask
    constrained = emissions.clone()
    constrained[..., 1] = constrained[..., 1].masked_fill(~allowed_delete, -1e9)
    best = crf.viterbi(constrained, mask)
    best &= mask.long()
    base = base_tags.long() & mask.long()
    length = mask.long().sum(1).clamp_min(1)
    margin = (token_sequence_score(crf, constrained, best, mask)
              - token_sequence_score(crf, constrained, base, mask)) / length
    changed = best.ne(base).any(1) & margin.ge(minimum_margin_per_phone)
    decoded = torch.where(changed[:, None], best, base)
    return decoded, margin


class StructuredDeletionJointRealizer(JointRealizer):
    """公開 realizer の DEL 系列のみを one-sided に精緻化する realizer。"""

    def __init__(self, ckpt_path: str, decode_cfg: dict, device: str = "cpu",
                 dictionary=None, overrides=None, payload=None,
                 edit_prior=None, prior_weight: float = 0.0,
                 prior_threshold: float = 0.0):
        payload = payload if payload is not None else torch.load(
            ckpt_path, map_location="cpu", weights_only=True)
        if payload.get("format") != FORMAT:
            raise ValueError(f"{FORMAT} checkpoint ではありません: {ckpt_path}")
        self.model, self.maps = build_joint_tagger(payload["base"], device)
        self.sub_targets = self.maps["sub_targets"]
        self.decode = decode_cfg
        self.device = device
        self.vocab = json.load(open(REPO / "assets/phone_vocab.json"))["phone_vocab"]
        self.inv = {v: k for k, v in self.vocab.items()}
        self.d = dictionary if dictionary is not None else \
            load_existing_dict(str(REPO / "assets/g2p/japanese_mfa.dict"))
        self.ov = overrides if overrides is not None else {}
        self._cache = {}

        pattern = payload["pattern"]
        cfg = pattern["config"]
        self.pattern_head = TokenPatternHead(
            cfg["hidden_dim"], cfg["inner_dim"], cfg["delete_threshold"]).to(device)
        self.pattern_head.load_state_dict(pattern["head_state_dict"], strict=True)
        self.pattern_head.eval()
        self.pattern_crf = LinearChainCRF(2).to(device)
        self.pattern_crf.load_state_dict(pattern["crf_state_dict"], strict=True)
        self.pattern_crf.eval()
        self.minimum_margin = float(cfg["minimum_margin_per_phone"])
        # decode metadata と checkpoint の margin 不一致は fail fast (公開契約)
        meta_margin = decode_cfg.get("structured_margin_per_phone")
        if meta_margin is not None and abs(float(meta_margin) - self.minimum_margin) > 1e-9:
            raise ValueError(
                f"structured margin 不一致: ckpt {self.minimum_margin} != decode.json {meta_margin}")
        # データ由来 empirical edit verifier (one-sided: non-KEEP を KEEP へ戻すのみ)
        self.edit_prior = edit_prior
        self.prior_weight = float(prior_weight)
        self.prior_threshold = float(prior_threshold)
        self.audit_rows: list[dict] = []
        self._encoded_hidden = None
        self.model.enc.register_forward_hook(self._capture_hidden)

    def _capture_hidden(self, _module, _inputs, output) -> None:
        self._encoded_hidden = output.detach()

    def _base_ops(self, op_logits, ph_tok, c):
        """公開 decoder と同一のゲート + protect_onset を適用した ops。"""
        probability = torch.softmax(op_logits, -1)
        ops = op_logits.argmax(-1).tolist()
        sub_threshold = float(self.decode.get("sub_threshold", self.decode["op_threshold"]))
        for i in range(len(ops)):
            threshold = sub_threshold if ops[i] >= 2 else float(self.decode["op_threshold"])
            if ops[i] != 0 and float(probability[i, ops[i]]) < threshold:
                ops[i] = 0
            if ops[i] == 1 and float(probability[i, 1]) < float(self.decode["del_threshold"]):
                ops[i] = 0
        if ph_tok:  # protect_onset: 先頭トークンは編集しない
            for i, token in enumerate(ph_tok):
                if token == ph_tok[0]:
                    ops[i] = 0
        if self.decode.get("safe_sub_only", True):  # 異音系 SUB のみ許可
            for i in range(len(ops)):
                if ops[i] >= 2:
                    src = self.inv.get(c[i], "")
                    tgt = self.inv.get(self.sub_targets[ops[i] - 2], "")
                    if not (tgt.startswith(src) and len(tgt) > len(src)):
                        ops[i] = 0
        return ops

    @torch.no_grad()
    def _decode_utt(self, text, ctx):
        cc = self.canon(text)
        if cc is None:
            return None
        c, tf, tokens_meta, ph_tok, ph_mora = cc
        if len(c) > MAXLEN:
            return None
        op_l, sil_l, dur, sil_dur = self._forward(c, tf, tokens_meta, ph_tok, ph_mora, ctx)
        hidden = self._encoded_hidden
        if hidden is None:
            raise RuntimeError("realizer encoder hook が hidden state を捕捉していない")
        ops = self._base_ops(op_l, ph_tok, c)

        # structured refinement: DEL の並びを CRF で見直し、KEEP へ戻す方向のみ許可
        log_probability = torch.log_softmax(op_l, -1)
        delete_log_odds = (log_probability[:, 1] - log_probability[:, 0]).unsqueeze(0)
        packed = pack_token_patterns(hidden[:, :len(c)], delete_log_odds,
                                     ph_tok, len(tokens_meta))
        if packed is not None:
            phone_h, token_h, odds, mask, keys, positions_by_token = packed
            emissions = self.pattern_head(phone_h, token_h, odds)
            base_tags = torch.zeros_like(mask, dtype=torch.long)
            for index, token in enumerate(keys):
                for local, position in enumerate(positions_by_token[token]):
                    base_tags[index, local] = int(ops[position] == 1)
            decoded, _margin = decode_one_sided(
                self.pattern_crf, emissions, base_tags, mask,
                minimum_margin_per_phone=self.minimum_margin)
            for index, token in enumerate(keys):
                for local, position in enumerate(positions_by_token[token]):
                    if ops[position] == 1 and int(decoded[index, local]) == 0:
                        ops[position] = 0

        sils = torch.sigmoid(sil_l).tolist()
        durations = dur.tolist()
        sil_durations = sil_dur.tolist()
        sil_after = [sils[i] >= float(self.decode["sil_threshold"])
                     and i < len(c) - 1 and tf[i] == 1 for i in range(len(c))]
        for i in range(len(ops)):  # sil 越境融合の禁止 (公開契約)
            if ops[i] != 1:
                continue
            same_prev = i > 0 and c[i - 1] == c[i] and not sil_after[i - 1]
            same_next = i + 1 < len(c) and c[i + 1] == c[i] and not sil_after[i]
            cross_prev = i > 0 and c[i - 1] == c[i] and sil_after[i - 1]
            cross_next = i + 1 < len(c) and c[i + 1] == c[i] and sil_after[i]
            if (cross_prev or cross_next) and not (same_prev or same_next):
                ops[i] = 0

        if self.edit_prior is not None:   # 移植元と同じ順序 (sil 越境レールの直後)
            log_probability = torch.log_softmax(op_l, -1)
            positions_by_token = [[] for _ in tokens_meta]
            for position, token in enumerate(ph_tok):
                positions_by_token[token].append(position)
            scores, rejected = [], []
            for token, positions in enumerate(positions_by_token):
                token_phones = [int(c[position]) for position in positions]
                for local, position in enumerate(positions):
                    proposal = int(ops[position])
                    if proposal == 0:
                        continue
                    prior_odds, support, level = self.edit_prior.proposal_log_odds(
                        token_phones, local, proposal)
                    model_odds = float(log_probability[position, proposal]
                                       - log_probability[position, 0])
                    score = model_odds + self.prior_weight * prior_odds
                    is_rejected = score < self.prior_threshold
                    scores.append({"token": tokens_meta[token][0],
                                   "phone_position": position,
                                   "operation": "DEL" if proposal == 1 else "SUB",
                                   "model_log_odds": model_odds,
                                   "prior_log_odds": prior_odds,
                                   "combined_score": score,
                                   "support": support, "backoff_level": level,
                                   "rejected": is_rejected})
                    if is_rejected:
                        ops[position] = 0      # KEEP へ戻すだけ (新規編集は作らない)
                        rejected.append(position)
            self.audit_rows.append({"text": text, "rejected_positions": rejected,
                                    "scores": scores})

        out_phones, out_durs = apply_edits(
            c, ops, durations, sil_after, sil_durations, self.sub_targets, self.inv)
        if c and not out_phones:  # 全削除 fallback (公開契約)
            out_phones = list(c)
            out_durs = list(durations)
        return out_phones, out_durs


def _resolve_edit_prior(decode_cfg: dict, ckpt_path: str, prior_path=None):
    """decode 設定の edit_verifier に従って prior を読み込む (SHA 照合つき)。"""
    cfg = decode_cfg.get("edit_verifier") or {}
    if not cfg.get("enabled"):
        return None, 0.0, 0.0
    path = Path(prior_path) if prior_path else Path(ckpt_path).parent / "edit_prior.pt"
    if not path.exists():
        raise FileNotFoundError(f"edit verifier が有効ですが prior がありません: {path}")
    expected = cfg.get("sha256")
    if expected:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError(f"edit prior の SHA256 が decode 設定と不一致: {path}")
    prior, meta = load_edit_prior(str(path))
    smoothing = cfg.get("smoothing")
    if smoothing is not None and abs(float(smoothing) - prior.smoothing) > 1e-12:
        raise ValueError(
            f"smoothing 不一致: asset {prior.smoothing} != decode.json {smoothing}")
    return prior, float(cfg["prior_weight"]), float(cfg["threshold"])


def load_realizer(ckpt_path: str, decode_cfg: dict, device: str = "cpu",
                  dictionary=None, overrides=None, prior_path=None):
    """checkpoint の format を見て realizer を選ぶ factory。"""
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    fmt = payload.get("format")
    if fmt == FORMAT:
        prior, weight, threshold = _resolve_edit_prior(decode_cfg, ckpt_path, prior_path)
        return StructuredDeletionJointRealizer(
            ckpt_path, decode_cfg, device=device, dictionary=dictionary,
            overrides=overrides, payload=payload, edit_prior=prior,
            prior_weight=weight, prior_threshold=threshold)
    if fmt == "joint_tagger_v1":
        return JointRealizer(ckpt_path, decode_cfg, device=device,
                             dictionary=dictionary, overrides=overrides,
                             payload=payload)
    raise ValueError(f"未知の realizer checkpoint format: {fmt}")

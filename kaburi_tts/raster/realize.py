"""realizer 推論: テキスト → duration つき実現形音素列。

規範形化は kaburi_tts.g2p.canonical の canon() (Sudachi 分割 + 辞書 + overrides +
語末促音) を使う。decode は MAP + 確信度ゲート (op / DEL 別) +
既存と同系の安全レール (protect_onset / safe_sub_only / sil 越境整合 / 融合)。
ゲート値は assets/raster/ の decode 設定 (held-out 較正値) から与える。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from kaburi_tts.g2p.canonical import FUSE, SIL_ID, canon, build_dialog_ctx
from kaburi_tts.g2p.dict_g2p import load_existing_dict

from .models import MAXLEN, build_joint_tagger, load_joint_tagger

REPO = Path(__file__).resolve().parents[2]


def apply_edits(c, ops, durations, sil_after, sil_durations, sub_targets, inv):
    """編集 op 列を教師契約どおりに適用する。

    教師契約 (build_joint_data.py): DEL 位置の duration は -1 で学習から除外される。
    したがって推論でも DEL 位置は音素・duration とも一切出力に使わない
    (同一母音隣接の D+K/K+D も、残る短母音とその duration のみを出力する。
    暗黙の長母音化・duration 加算は行わない)。長母音が出力されるのは
    明示的な SUB (長母音 target) の場合のみ。
    """
    out_phones: list[int] = []
    out_durs: list[float] = []
    skip = False
    for i, pid in enumerate(c):
        if skip:
            skip = False
            continue
        op = ops[i]
        if op == 0:
            out_phones.append(pid)
            out_durs.append(durations[i])
        elif op == 1:
            pass  # DEL: 出力なし (duration は未学習値のため使用しない)
        else:
            sub_id = sub_targets[op - 2]
            out_phones.append(sub_id)
            out_durs.append(durations[i])
            if i + 1 < len(c):
                sub_sym = inv.get(sub_id, "")
                cur_sym = inv.get(pid, "")
                nxt_sym = inv.get(c[i + 1], "")
                if sub_sym.endswith("ː") and FUSE.get((cur_sym, nxt_sym)) == sub_sym:
                    skip = True
        if sil_after[i] and out_phones and out_phones[-1] != SIL_ID:
            out_phones.append(SIL_ID)
            out_durs.append(max(2.0, sil_durations[i]))
    return out_phones, out_durs


class JointRealizer:
    def __init__(self, ckpt_path: str, decode_cfg: dict, device: str = "cpu",
                 dictionary=None, overrides=None, payload=None):
        if payload is not None:   # factory が読み込み済みの payload を渡す場合
            if payload.get("format") != "joint_tagger_v1":
                raise ValueError(f"joint_tagger_v1 checkpoint ではありません: {ckpt_path}")
            self.model, self.maps = build_joint_tagger(payload, device)
        else:
            self.model, self.maps = load_joint_tagger(ckpt_path, device)
        self.sub_targets = self.maps["sub_targets"]
        self.decode = decode_cfg
        self.device = device
        self.vocab = json.load(open(REPO / "assets/phone_vocab.json"))["phone_vocab"]
        self.inv = {v: k for k, v in self.vocab.items()}
        self.d = dictionary if dictionary is not None else \
            load_existing_dict(str(REPO / "assets/g2p/japanese_mfa.dict"))
        self.ov = overrides if overrides is not None else {}
        self._cache: dict = {}

    def canon(self, text):
        return canon(text, self.d, self.ov, self.vocab, self._cache)

    @torch.no_grad()
    def _forward(self, c, tf, tokens_meta, ph_tok, ph_mora, ctx):
        L = min(len(c), MAXLEN)
        dev = self.device
        b = {
            "c": torch.tensor([c[:L]], dtype=torch.long, device=dev),
            "tf": torch.tensor([tf[:L]], dtype=torch.long, device=dev),
            "mask": torch.ones(1, L, dtype=torch.bool, device=dev),
            "lens": torch.tensor([L], dtype=torch.long, device=dev),
        }
        pos = torch.zeros(1, L, dtype=torch.long, device=dev)
        tokpos = torch.zeros(1, L, dtype=torch.long, device=dev)
        for j in range(L):
            pos[0, j] = self.maps["pos"].get(tokens_meta[ph_tok[j]][1], 0)
            tokpos[0, j] = min(ph_mora[j], 63)
        b["pos"], b["tokpos"] = pos, tokpos
        b["trans"] = torch.tensor([self.maps["trans"].get(ctx.get("trans"), 0)],
                                  dtype=torch.long, device=dev)
        b["pspk"] = torch.tensor([self.maps["pspk"].get(ctx.get("prev_spk"), 0)],
                                 dtype=torch.long, device=dev)
        b["nspk"] = torch.tensor([self.maps["pspk"].get(ctx.get("next_spk"), 0)],
                                 dtype=torch.long, device=dev)
        b["pic"] = torch.tensor([[float(ctx.get("pos_in_chunk") or 0.0)]],
                                dtype=torch.float, device=dev)
        op_l, sil_l, dur, sil_dur = self.model(b)
        return op_l[0], sil_l[0], dur[0], sil_dur[0]

    def _decode_utt(self, text, ctx):
        cc = self.canon(text)
        if cc is None:
            return None
        c, tf, tokens_meta, ph_tok, ph_mora = cc
        if len(c) > MAXLEN:
            return None
        op_l, sil_l, dur, sil_dur = self._forward(c, tf, tokens_meta, ph_tok, ph_mora, ctx)
        op_threshold = float(self.decode["op_threshold"])
        del_threshold = float(self.decode["del_threshold"])
        sil_threshold = float(self.decode["sil_threshold"])
        p = torch.softmax(op_l, -1)
        ops = op_l.argmax(-1).tolist()
        for i in range(len(ops)):
            if ops[i] != 0 and float(p[i, ops[i]]) < op_threshold:
                ops[i] = 0
            if ops[i] == 1 and float(p[i, 1]) < del_threshold:
                ops[i] = 0
        if ph_tok:  # protect_onset: 先頭トークンは編集しない (語頭脱落抑止)
            for i in range(len(ops)):
                if ph_tok[i] == ph_tok[0]:
                    ops[i] = 0
        for i in range(len(ops)):  # safe_sub_only: 異音系 SUB のみ許可
            if ops[i] >= 2:
                src = self.inv.get(c[i], "")
                tgt = self.inv.get(self.sub_targets[ops[i] - 2], "")
                if not (tgt.startswith(src) and len(tgt) > len(src)):
                    ops[i] = 0
        sils = torch.sigmoid(sil_l).tolist()
        durations = dur.tolist()
        sil_durations = sil_dur.tolist()

        allow_sil = [tf[i] == 1 for i in range(len(c))]
        sil_after = [
            sils[i] >= sil_threshold and i < len(c) - 1 and allow_sil[i]
            for i in range(len(c))
        ]
        for i in range(len(ops)):  # sil 越境融合の禁止
            if ops[i] != 1:
                continue
            same_prev = i > 0 and c[i - 1] == c[i] and not sil_after[i - 1]
            same_next = i + 1 < len(c) and c[i + 1] == c[i] and not sil_after[i]
            cross_prev = i > 0 and c[i - 1] == c[i] and sil_after[i - 1]
            cross_next = i + 1 < len(c) and c[i + 1] == c[i] and sil_after[i]
            if (cross_prev or cross_next) and not (same_prev or same_next):
                ops[i] = 0

        out_phones, out_durs = apply_edits(
            c, ops, durations, sil_after, sil_durations,
            self.sub_targets, self.inv)
        if c and not out_phones:  # 全削除 fallback
            out_phones = list(c)
            out_durs = [durations[i] for i in range(len(c))]
        return out_phones, out_durs

    def realize_chunk(self, utterances, chunk_id="dialog"):
        """utterances=[(speaker,text),...] → [(phones, durations[frames]) or None]。"""
        out = []
        for i, (spk, text) in enumerate(utterances):
            ctx = build_dialog_ctx(utterances, i)
            out.append(self._decode_utt(text, ctx))
        return out

"""音素ラスタ生成のモデル定義 (推論用)。

realizer (realizer.pt) = JointTagger: 発話テキスト(規範形音素列)から
  編集op(KEEP/DEL/SUB48)・発話内sil挿入・各音素duration・sil duration を同時出力。
gap model (gap.pt) = GapModel: 対話全体の発話列から first start と符号つき any-gap を AR 生成。

いずれも実対話コーパスの MFA 整列実測 (実現形・実 duration・実 gap) を教師に
教師あり学習したもの。学習コードは本リポジトリには含めていない。
本パッケージは checkpoint の読み込みと forward のみを持つ。

checkpoint 内部の format 識別子 ("joint_tagger_v1" / "tower2_gap_v1") は学習時に
payload へ焼き込まれた来歴 ID であり、公開ファイル名 (realizer.pt / gap.pt) とは
独立。ID を変えると既存 ckpt がロードできなくなるため変更しない。
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_SUB = 48
MAXLEN = 176          # realizer の音素長上限 (超過発話は canonical fallback)
MAX_UTTS = 64         # gap model の発話数上限
GAP_SCALE = 20.0


class JointTagger(nn.Module):
    """realizer 本体。位置ごとの分類/回帰 head を持つ Transformer encoder。"""

    def __init__(self, vsz, maps, dim, layers, heads, ff):
        super().__init__()
        self.emb = nn.Embedding(vsz, dim, padding_idx=0)
        self.pos = nn.Embedding(MAXLEN, dim)
        self.pos_end = nn.Embedding(MAXLEN, dim)
        self.tokf = nn.Embedding(2, dim)
        self.pos_e = nn.Embedding(len(maps["pos"]), dim)
        self.tokpos_e = nn.Embedding(64, dim)
        self.trans_e = nn.Embedding(len(maps["trans"]), dim)
        self.pspk_e = nn.Embedding(len(maps["pspk"]), dim)
        self.nspk_e = nn.Embedding(len(maps["pspk"]), dim)
        self.pic_lin = nn.Linear(1, dim)
        enc = nn.TransformerEncoderLayer(dim, heads, ff, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.op_head = nn.Linear(dim, 2 + N_SUB)
        self.sil_head = nn.Linear(dim, 1)
        self.dur_head = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.sil_dur_head = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))

    def forward(self, b):
        x, tf = b["c"], b["tf"]
        B, T = x.shape
        idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        dist = (b["lens"].unsqueeze(1) - 1 - idx).clamp(0, MAXLEN - 1)
        h = (self.emb(x) + self.pos(idx.clamp(max=MAXLEN - 1)) + self.pos_end(dist)
             + self.tokf(tf) + self.pos_e(b["pos"]) + self.tokpos_e(b["tokpos"].clamp(max=63)))
        u = (self.trans_e(b["trans"]) + self.pspk_e(b["pspk"]) + self.nspk_e(b["nspk"])
             + self.pic_lin(b["pic"]))
        h = h + u.unsqueeze(1)
        h = self.enc(h, src_key_padding_mask=~b["mask"])
        durations = 1.0 + nn.functional.softplus(self.dur_head(h).squeeze(-1))
        sil_durations = 1.0 + nn.functional.softplus(self.sil_dur_head(h).squeeze(-1))
        return (self.op_head(h), self.sil_head(h).squeeze(-1), durations, sil_durations)


class GapModel(nn.Module):
    """gap model 本体。双方向文脈 + AR-GRU、gap は overlap 符号 x 正負 magnitude に因子化。

    endpoint_k>0 の checkpoint は、各発話の先頭/末尾 k 音素の embedding を
    残差 adapter として encode に加える (global-opt-20260820 以降)。
    endpoint_k=0 は従来 checkpoint と数値等価。
    """

    def __init__(self, vsz, dim=256, layers=4, heads=4, ff=1024, endpoint_k=0):
        super().__init__()
        self.endpoint_k = int(endpoint_k)
        if self.endpoint_k < 0:
            raise ValueError("endpoint_k must be non-negative")
        self.phone_emb = nn.Embedding(vsz, 96, padding_idx=0)
        self.spk_emb = nn.Embedding(2, 16)
        self.pos_emb = nn.Embedding(MAX_UTTS, dim)
        self.input = nn.Sequential(
            nn.Linear(96 + 16 + 1, dim), nn.SiLU(), nn.LayerNorm(dim))
        if self.endpoint_k:
            self.endpoint_adapter = nn.Sequential(
                nn.Linear(2 * self.endpoint_k * 96, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            nn.init.zeros_(self.endpoint_adapter[-1].weight)
            nn.init.zeros_(self.endpoint_adapter[-1].bias)
        enc = nn.TransformerEncoderLayer(dim, heads, ff, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.ar = nn.GRUCell(dim + 1, dim)
        state = 2 * dim
        self.first_head = nn.Sequential(nn.Linear(state, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.sign_head = nn.Sequential(nn.Linear(state, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.pos_head = nn.Sequential(nn.Linear(state, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.neg_head = nn.Sequential(nn.Linear(state, dim), nn.SiLU(), nn.Linear(dim, 1))

    def encode(self, b):
        emb = self.phone_emb(b["phones"])
        m = b["phone_mask"].unsqueeze(-1).to(emb.dtype)
        pooled = (emb * m).sum(-2) / m.sum(-2).clamp_min(1.0)
        x = torch.cat([pooled, self.spk_emb(b["spk"]),
                       torch.log1p(b["dur_total"]).unsqueeze(-1)], dim=-1)
        h = self.input(x)
        if self.endpoint_k:
            B, U_, P, E = emb.shape
            k = self.endpoint_k
            if P < k:
                pad = emb.new_zeros(B, U_, k - P, E)
                first = torch.cat([emb, pad], dim=2)
                first_mask = torch.cat([
                    b["phone_mask"],
                    torch.zeros(B, U_, k - P, dtype=torch.bool, device=emb.device),
                ], dim=2)
            else:
                first = emb[:, :, :k]
                first_mask = b["phone_mask"][:, :, :k]
            lengths = b["phone_mask"].sum(-1)
            slots = torch.arange(k, device=emb.device).view(1, 1, k)
            last_index = (lengths.unsqueeze(-1) - k + slots).clamp(0, P - 1)
            last = emb.gather(2, last_index.unsqueeze(-1).expand(B, U_, k, E))
            last_mask = slots >= (k - lengths.unsqueeze(-1)).clamp_min(0)
            endpoints = torch.cat([
                first * first_mask.unsqueeze(-1).to(emb.dtype),
                last * last_mask.unsqueeze(-1).to(emb.dtype),
            ], dim=2).flatten(-2)
            h = h + self.endpoint_adapter(endpoints)
        U = h.shape[1]
        h = h + self.pos_emb(torch.arange(U, device=h.device)).unsqueeze(0)
        return self.enc(h, src_key_padding_mask=~b["utt_mask"])

    def forward(self, b, teacher_gaps=None):
        enc = self.encode(b)
        B, U, D = enc.shape
        hidden = torch.zeros(B, D, device=enc.device)
        prev = torch.zeros(B, 1, device=enc.device)
        gaps = []
        for u in range(U):
            proposed = self.ar(torch.cat([enc[:, u], prev], -1), hidden)
            valid = b["utt_mask"][:, u].unsqueeze(-1)
            hidden = torch.where(valid, proposed, hidden)
            state = torch.cat([enc[:, u], hidden], -1)
            sign_l = self.sign_head(state).squeeze(-1)
            pos_m = nn.functional.softplus(self.pos_head(state)).squeeze(-1) * GAP_SCALE
            neg_m = nn.functional.softplus(self.neg_head(state)).squeeze(-1) * GAP_SCALE
            first = nn.functional.softplus(self.first_head(state)).squeeze(-1)
            hard = (torch.sigmoid(sign_l) >= 0.5).float()
            decoded = torch.where(
                torch.tensor(u == 0, device=enc.device),
                first, (1 - hard) * pos_m - hard * neg_m)
            gaps.append(decoded)
            nxt = (teacher_gaps[:, u] if teacher_gaps is not None else decoded)
            prev = (nxt / GAP_SCALE).unsqueeze(-1)
        return dict(gap=torch.stack(gaps, dim=1))


def build_joint_tagger(payload: dict, device="cpu"):
    """joint_tagger_v1 payload (単体 or structured ckpt の base) から本体を構築。"""
    cfg = payload["config"]
    if cfg.get("use_txt"):
        raise ValueError("realizer は no-txt checkpoint のみ対応")
    model = JointTagger(cfg["vsz"], payload["maps"], cfg["dim"], cfg["layers"],
                        cfg["heads"], cfg["ff"]).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload["maps"]


def load_joint_tagger(ckpt_path: str, device="cpu"):
    payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    if payload.get("format") != "joint_tagger_v1":
        raise ValueError(f"joint_tagger_v1 checkpoint ではありません: {ckpt_path}")
    cfg = payload["config"]
    if cfg.get("use_txt"):
        raise ValueError("realizer は no-txt checkpoint のみ対応")
    model = JointTagger(cfg["vsz"], payload["maps"], cfg["dim"], cfg["layers"],
                        cfg["heads"], cfg["ff"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["maps"]


def load_gap_model(ckpt_path: str, device="cpu"):
    payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    if payload.get("format") != "tower2_gap_v1":
        raise ValueError(f"tower2_gap_v1 checkpoint ではありません: {ckpt_path}")
    cfg = payload["config"]
    model = GapModel(cfg["vsz"], cfg["dim"], cfg["layers"],
                     endpoint_k=int(cfg.get("endpoint_k", 0))).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model

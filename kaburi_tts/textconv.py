"""テキストコンバータ: 整った書き言葉の対話を「話し言葉」に自動変換する。

KABURI-TTS は自発対話音声で学習されているため、入力テキストが実際の話し言葉の
分布 (短いターン・相槌・フィラー・砕けた終止形) に近いほど自然に合成される。
本モジュールは、学習コーパスの実書き起こしを LLM で「整った文」に逆翻訳し、
その逆方向を学習した小型モデル (llm-jp-3-440m ベース) で、任意の対話テキストを
学習分布に一致した話し言葉へ変換する。相槌の挿入・発話の分割も行う。

使い方:
    from kaburi_tts.textconv import KaburiTextConverter
    conv = KaburiTextConverter.from_hf()          # HF から取得
    conv = KaburiTextConverter("path/to/ckpt")    # ローカル ckpt
    utts = conv.convert_utts([{"speaker": "A", "text": "..."}, ...])
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_REPO = "llm-jp/kaburi-tts"
HF_SUBFOLDER = "textconverter"
BASE_TOKENIZER = "llm-jp/llm-jp-3-440m"

# 学習時のプロンプト形式 (変更すると学習済みモデルと不整合になるため固定)
PROMPT_HEADER = "### 入力対話\n"
COMPLETION_HEADER = "\n### 話し言葉\n"

_LINE_RE = re.compile(r"^[ＡＢAB]\s*[:：]\s*(.+)$")
_SPEAKER_NORM = {"Ａ": "A", "Ｂ": "B", "A": "A", "B": "B"}

# 縮退検出: 1〜4 字のユニットが 5 回以上連続 (「ねえねえねえ…」)。
# 実対話の「うんうんうん」程度 (3 回前後) は許容する。
_DEGENERATE_RE = re.compile(r"(.{1,4})\1{4,}")


def _parse_lines(raw: str) -> tuple[list[tuple[str, str]], int]:
    parsed, n_bad = [], 0
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        m = _LINE_RE.match(ln)
        if m:
            parsed.append((_SPEAKER_NORM[ln[0]], m.group(1).strip()))
        else:
            n_bad += 1
    return parsed, n_bad


def _is_degenerate(text: str) -> bool:
    return bool(_DEGENERATE_RE.search(text)) or len(text) > 100


class KaburiTextConverter:
    def __init__(self, ckpt_dir: str, device: str = "cuda",
                 temperature: float = 0.55, top_p: float = 0.9,
                 max_new_tokens: int = 600, repetition_penalty: float = 1.15,
                 hf_subfolder: str | None = None):
        kw = {"subfolder": hf_subfolder} if hf_subfolder else {}
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, **kw)
        except (ValueError, OSError):
            # ckpt の tokenizer 形式が手元の transformers で読めない場合は
            # ベースモデルの tokenizer で代替 (fine-tune では tokenizer 不変)
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, torch_dtype=torch.float16, **kw).to(device).eval()
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty

    @classmethod
    def from_hf(cls, repo_id: str = HF_REPO, subfolder: str = HF_SUBFOLDER, **kw):
        return cls(repo_id, hf_subfolder=subfolder, **kw)

    def _encode_prompt(self, clean_block: str) -> list[int]:
        ids = self.tokenizer(PROMPT_HEADER + clean_block + COMPLETION_HEADER,
                             add_special_tokens=False)["input_ids"]
        if self.tokenizer.bos_token_id is not None:
            ids = [self.tokenizer.bos_token_id] + ids
        return ids

    @torch.no_grad()
    def convert_window(self, utts: list[dict], n_retry: int = 3) -> list[dict]:
        """utts: [{"speaker": "A", "text": ...}, ...] の 1 窓分 (最大 16 発話目安)。

        パース失敗・話者不整合・縮退は引き直し、n_retry 回失敗したら
        その窓は入力をそのまま返す (無変換 fallback)。
        """
        block = "\n".join(f"{u['speaker']}: {u['text']}" for u in utts)
        prompt_ids = torch.tensor([self._encode_prompt(block)], device=self.device)
        attn = torch.ones_like(prompt_ids)
        src_speakers = {u["speaker"] for u in utts}
        for _ in range(n_retry):
            out = self.model.generate(
                input_ids=prompt_ids, attention_mask=attn,
                do_sample=self.temperature > 0, temperature=self.temperature,
                top_p=self.top_p, max_new_tokens=self.max_new_tokens,
                repetition_penalty=self.repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id)
            text = self.tokenizer.decode(out[0][prompt_ids.shape[1]:],
                                         skip_special_tokens=True)
            parsed, n_bad = _parse_lines(text)
            if (parsed and n_bad == 0
                    and {s for s, _ in parsed} <= (src_speakers or {"A", "B"})
                    and not any(_is_degenerate(t) for _, t in parsed)):
                return [{"speaker": s, "text": t} for s, t in parsed]
        return list(utts)

    def convert_utts(self, utts: list[dict], max_utts: int = 16,
                     backtrack: int = 3) -> list[dict]:
        """対話全体を窓に区切って逐次変換 (窓境界は話者交替点を優先)。"""
        windows: list[list[dict]] = []
        cur: list[dict] = []
        for u in utts:
            if len(cur) >= max_utts:
                cut = len(cur)
                for k in range(1, min(backtrack, len(cur) - 1) + 1):
                    if cur[-k]["speaker"] != cur[-k - 1]["speaker"]:
                        cut = len(cur) - k
                        break
                windows.append(cur[:cut])
                cur = cur[cut:]
            cur.append(u)
        if cur:
            windows.append(cur)
        out: list[dict] = []
        for w in windows:
            out.extend(self.convert_window(w))
        return out

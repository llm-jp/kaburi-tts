"""音素ラスタ生成 runtime の契約・回帰テスト。

実行:
  .venv/bin/python -m unittest tests.test_global_opt -v

CPU のみで動く。checkpoint が無い環境では該当 test を skip する。
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

NEW_REALIZER = REPO / "hf_upload/raster/global-opt-20260820/realizer.pt"
NEW_GAP = REPO / "hf_upload/raster/global-opt-20260820/gap.pt"
NEW_DECODE = REPO / "assets/raster/decode_config.json"
NEW_PRIOR = REPO / "hf_upload/raster/global-opt-20260820/edit_prior.pt"

# 公開デモ text 入力の台本 (発音回帰の検査に使う)
SAMPLE02 = [
    {
        "speaker": "A",
        "text": "あ、えっとね"
    },
    {
        "speaker": "A",
        "text": "私実は夏とかも裸足で過ごすのが苦手なんだよね"
    },
    {
        "speaker": "B",
        "text": "うーん"
    },
    {
        "speaker": "B",
        "text": "あーーー"
    },
    {
        "speaker": "A",
        "text": "もうなんかあのなんていうのかな"
    },
    {
        "speaker": "A",
        "text": "靴とか"
    },
    {
        "speaker": "B",
        "text": "床のぺたぺた"
    },
    {
        "speaker": "A",
        "text": "そうそうそうそうサンダルも苦手だし家の中でも靴下履くしみたいな"
    },
    {
        "speaker": "B",
        "text": "一年中ずっと?"
    },
    {
        "speaker": "A",
        "text": "よほどの暑い日以外の時はもうずーっと"
    },
    {
        "speaker": "B",
        "text": "逆に私とかは裸足好きだから"
    }
]

SAMPLE04 = [
    {
        "speaker": "A",
        "text": "なんか夏のそのご予定で楽しみにしていることってある?"
    },
    {
        "speaker": "B",
        "text": "えっとね"
    },
    {
        "speaker": "B",
        "text": "夏はね"
    },
    {
        "speaker": "A",
        "text": "うん"
    },
    {
        "speaker": "B",
        "text": "特にないのよ"
    },
    {
        "speaker": "B",
        "text": "秋に旅行行きたいなと思ってて"
    },
    {
        "speaker": "A",
        "text": "いいよね、なんか涼しい方がね。"
    },
    {
        "speaker": "B",
        "text": "そうそうそう"
    },
    {
        "speaker": "B",
        "text": "でも本当は海がいいんだけど"
    },
    {
        "speaker": "B",
        "text": "真夏は"
    },
    {
        "speaker": "A",
        "text": "ううん"
    },
    {
        "speaker": "B",
        "text": "暑すぎるから遊べなくなっちゃって"
    },
    {
        "speaker": "A",
        "text": "あーもう日焼けもするしねすぐ"
    }
]

UTTS = [
    {"speaker": "A", "text": "最近さ、朝がほんと起きられなくて"},
    {"speaker": "B", "text": "あー、分かる"},
    {"speaker": "A", "text": "目覚まし三個かけてるんだけどね"},
]

# 公開デモ sample06 と同一の発話列。realizer は対話文脈 (話者遷移・chunk 内位置)
# を条件に取るため、発音回帰は同じ文脈で検査する必要がある。
SAMPLE06 = [
    {"speaker": "A", "text": "ちょっと最近したちょっとお高いのとか"},
    {"speaker": "B", "text": "なんか実はちょっといいお化粧品を思い切って買っちゃって"},
    {"speaker": "A", "text": "いいですね"},
    {"speaker": "A", "text": "たまには自分のご褒美とかも大事だと思いますね"},
    {"speaker": "B", "text": "使うたんびに"},
    {"speaker": "B", "text": "気分が上がる"},
    {"speaker": "B", "text": "あなたは何買われたんですか?"},
    {"speaker": "A", "text": "えっと私なんかその軽めの掃除機を買って"},
    {"speaker": "B", "text": "はい"},
    {"speaker": "A", "text": "掃除が楽になりましたね"},
    {"speaker": "B", "text": "いいお買い物"},
]


def _need(path):
    if not Path(path).exists():
        raise unittest.SkipTest(f"asset not available: {path}")


class CheckpointCompatTest(unittest.TestCase):
    """新旧 checkpoint がどちらも読め、期待の実装が選ばれること。"""

    def test_new_realizer_strict_load_and_decodes(self):
        _need(NEW_REALIZER)
        from kaburi_tts.raster.structured import StructuredDeletionJointRealizer, load_realizer
        r = load_realizer(str(NEW_REALIZER), json.loads(NEW_DECODE.read_text()), device="cpu")
        self.assertIsInstance(r, StructuredDeletionJointRealizer)
        self.assertAlmostEqual(r.minimum_margin, 0.25, places=9)
        out = r.realize_chunk([(u["speaker"], u["text"]) for u in UTTS], chunk_id="t")
        self.assertEqual(len(out), len(UTTS))
        for item in out:
            self.assertIsNotNone(item)
            self.assertGreater(len(item[0]), 0)

    def test_new_gap_model_endpoint_k_three(self):
        _need(NEW_GAP)
        from kaburi_tts.raster.models import load_gap_model
        m = load_gap_model(str(NEW_GAP), device="cpu")
        self.assertEqual(m.endpoint_k, 3)
        self.assertTrue(hasattr(m, "endpoint_adapter"))

    def test_margin_mismatch_fails_fast(self):
        _need(NEW_REALIZER)
        from kaburi_tts.raster.structured import StructuredDeletionJointRealizer
        cfg = json.loads(NEW_DECODE.read_text())
        cfg["structured_margin_per_phone"] = 0.5   # ckpt は 0.25
        with self.assertRaises(ValueError):
            StructuredDeletionJointRealizer(str(NEW_REALIZER), cfg, device="cpu")


class EditVerifierTest(unittest.TestCase):
    """empirical edit verifier: asset 契約と one-sided 動作。"""

    def test_prior_asset_format_and_contract(self):
        _need(NEW_PRIOR)
        import torch as _t
        from kaburi_tts.raster.edit_prior import load_edit_prior
        prior, meta = load_edit_prior(str(NEW_PRIOR))
        self.assertEqual(meta["format"], "tower1_empirical_edit_prior_v1")
        self.assertGreaterEqual(prior.operation_classes, 2)
        self.assertAlmostEqual(prior.smoothing, 0.5, places=9)
        # 契約は metadata の宣言ではなく asset の中身で検査する。
        # 条件付けキーは音素 ID とトークン内位置のみで、表層文字列・POS を含まない。
        import re as _re
        kana = _re.compile(r"[぀-ヿ一-鿿]")
        for table in (prior.exact, prior.window, prior.shape, prior.source):
            for key in list(table)[:2000]:
                self.assertFalse(kana.search(str(key)),
                                 f"表層文字列を含むキーがある: {key!r}")

    def test_prior_sha_mismatch_fails_fast(self):
        _need(NEW_PRIOR)
        from kaburi_tts.raster.structured import _resolve_edit_prior
        cfg = json.loads(NEW_DECODE.read_text())
        cfg["edit_verifier"] = dict(cfg["edit_verifier"], sha256="0" * 64)
        with self.assertRaises(ValueError):
            _resolve_edit_prior(cfg, str(NEW_REALIZER))

    def test_verifier_only_reverts_to_keep(self):
        """verifier は non-KEEP を KEEP へ戻すだけで、新しい編集を作らない。"""
        _need(NEW_REALIZER)
        from kaburi_tts.raster.structured import load_realizer
        cfg = json.loads(NEW_DECODE.read_text())
        off = dict(cfg, edit_verifier=dict(cfg["edit_verifier"], enabled=False))
        r_on = load_realizer(str(NEW_REALIZER), cfg, device="cpu")
        r_off = load_realizer(str(NEW_REALIZER), off, device="cpu")
        pairs = [(u["speaker"], u["text"]) for u in SAMPLE06]
        on = r_on.realize_chunk(pairs, chunk_id="sample06")
        off_out = r_off.realize_chunk(pairs, chunk_id="sample06")
        for a, b in zip(on, off_out):
            if a is None or b is None:
                continue
            # KEEP へ戻す方向のみ = 出力音素数は減らない
            self.assertGreaterEqual(len(a[0]), len(b[0]),
                                    "verifier が音素を減らしている (新しい編集を作った)")

    def test_rejects_the_two_reported_defects(self):
        """sample02「家」の SUB と sample04「すぐ」の DEL が reject される。"""
        _need(NEW_REALIZER)
        from kaburi_tts.raster import RasterGenerator
        g = RasterGenerator(device="cpu")
        inv = {v: k for k, v in g.realizer.vocab.items()}
        for sid, utts, idx, expect in (("sample02", SAMPLE02, 7, "ɕ i e n o"),
                                       ("sample04", SAMPLE04, 12, "s ɨ ɡ ɯ")):
            placed, _ = g.timeline(utts, chunk_id=sid)
            ph = " ".join(inv.get(p, "?") for p in placed[idx]["phones"])
            self.assertIn(expect, ph, f"{sid} utt{idx}: 期待する実現形になっていない")
            if sid == "sample02":
                self.assertNotIn("ɕ i oː", ph, "「家」の e->oː SUB が残っている")


class StructuredDecodeContractTest(unittest.TestCase):
    """one-sided 制約: base に無い DEL は決して増えない。"""

    def test_one_sided_never_adds_deletion(self):
        from kaburi_tts.raster.structured import LinearChainCRF, decode_one_sided
        crf = LinearChainCRF(2)
        with torch.no_grad():   # DEL を強く選好する emission
            emissions = torch.tensor([[[0.0, 5.0], [0.0, 5.0], [0.0, 5.0]]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        base = torch.zeros(1, 3, dtype=torch.long)          # base に DEL なし
        decoded, _ = decode_one_sided(crf, emissions, base, mask)
        self.assertEqual(int(decoded.sum()), 0, "base に無い DEL が追加された")

    def test_one_sided_can_restore_but_not_extend(self):
        from kaburi_tts.raster.structured import LinearChainCRF, decode_one_sided
        crf = LinearChainCRF(2)
        emissions = torch.tensor([[[5.0, 0.0], [5.0, 0.0], [0.0, 5.0]]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        base = torch.tensor([[1, 1, 0]])                     # 位置 0,1 が DEL
        decoded, _ = decode_one_sided(crf, emissions, base, mask)
        self.assertLessEqual(int(decoded.sum()), int(base.sum()), "DEL が増えている")
        self.assertEqual(int(decoded[0, 2]), 0, "base=KEEP の位置が DEL になった")


class ActivityGateTest(unittest.TestCase):
    SR, FPS = 48000, 25

    def _act(self):
        a = torch.zeros(2, 750)
        a[0, 100:200] = 1.0
        a[1, 400:450] = 1.0
        return a

    def test_shape_and_active_preserved(self):
        from kaburi_tts.raster.activity_gate import gate_channels_first
        spf = self.SR // self.FPS
        wav = torch.ones(2, 750 * spf)
        gated, env = gate_channels_first(wav, self._act(), self.SR)
        self.assertEqual(tuple(gated.shape), tuple(wav.shape))
        self.assertEqual(env.shape[0], wav.shape[1])
        self.assertTrue(torch.all(gated[:, 100 * spf:200 * spf] == 1.0),
                        "union active 区間の波形が変化した")
        self.assertTrue(torch.all(gated[:, 400 * spf:450 * spf] == 1.0))

    def test_padding_protected_and_exterior_muted(self):
        from kaburi_tts.raster.activity_gate import gate_channels_first
        spf = self.SR // self.FPS
        wav = torch.ones(2, 750 * spf)
        gated, _ = gate_channels_first(wav, self._act(), self.SR, pad_frames=2)
        self.assertTrue(torch.all(gated[:, 98 * spf:100 * spf] == 1.0), "pad 区間が保護されていない")
        self.assertEqual(float(gated[:, 600 * spf:700 * spf].abs().max()), 0.0,
                         "完全無発話区間が無音化されていない")

    def test_sample_rate_must_divide_fps(self):
        from kaburi_tts.raster.activity_gate import union_activity_envelope
        with self.assertRaises(ValueError):
            union_activity_envelope(self._act().numpy(), 1000, 44101, fps=25)

    def test_channels_last_contract(self):
        """numpy 側は [N,C] 契約 (channels-first を渡すと形が合わない)。"""
        import numpy as np
        from kaburi_tts.raster.activity_gate import suppress_fully_inactive
        spf = self.SR // self.FPS
        audio = np.ones((750 * spf, 2), dtype=np.float32)
        gated, env = suppress_fully_inactive(audio, self._act().numpy(), self.SR)
        self.assertEqual(gated.shape, audio.shape)
        self.assertEqual(env.shape[0], audio.shape[0])


class PronunciationRegressionTest(unittest.TestCase):
    """§8.4 発音回帰。runtime 規則は一般モデルのまま、出力 phone 列のみ検査する。"""

    @classmethod
    def setUpClass(cls):
        _need(NEW_REALIZER)
        from kaburi_tts.raster import RasterGenerator
        cls.g = RasterGenerator(device="cpu")
        cls.inv = {v: k for k, v in cls.g.realizer.vocab.items()}

    def _phones(self, utts, idx, chunk_id="fixture"):
        placed, _ = self.g.timeline(utts, chunk_id=chunk_id)
        return [self.inv.get(p, "?") for p in placed[idx]["phones"]]

    def test_natsu_survives(self):
        utts = [{"speaker": "A", "text": "なんか夏のそのご予定で楽しみにしていることってある?"}]
        ph = " ".join(self._phones(utts, 0))
        self.assertIn("n a ts ɨ", ph, "「夏」が消えている")
        self.assertIn("s o n o", ph, "「その」が消えている")

    def test_atashi_not_lengthened(self):
        ph = self._phones(SAMPLE06, 7, chunk_id="sample06")
        self.assertNotIn("aː", ph, "「私」が長母音化 (あーし) している")
        self.assertIn("w a t a ɕ", " ".join(ph), "「私」が失われている")

    def test_agaru_onset_kept(self):
        ph = " ".join(self._phones(SAMPLE06, 5, chunk_id="sample06"))
        self.assertIn("ɡ a ɾ ɯ", ph, "「あがる」が失われている")

    def test_nani_reading_matches_adopted_release(self):
        ph = " ".join(self._phones(SAMPLE06, 6, chunk_id="sample06"))
        self.assertIn("n a ɲ k a w a ɾ e t a", ph, "「何」の読みが採用 B と異なる")

    def test_no_empty_realization(self):
        utts = [{"speaker": "A", "text": "うんうん"},
                {"speaker": "B", "text": "あはは"},
                {"speaker": "A", "text": "そうなんだよ、無意識に止めてるっぽくて"}]
        placed, _ = self.g.timeline(utts, chunk_id="fixture")
        for u in placed:
            self.assertGreater(len(u["phones"]), 0, "空の phone 列が生成された")


if __name__ == "__main__":
    unittest.main()

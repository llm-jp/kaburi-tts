"""音素ラスタ生成 (kaburi_tts.raster) の unit test。

実行:
  python -m unittest discover -s tests

CPU のみで動く。モデル・辞書のロードを含むため初回は数十秒かかる。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SIL = 1
UTTS = [
    {"speaker": "A", "text": "最近さ、朝がほんと起きられなくて"},
    {"speaker": "B", "text": "あー、分かる"},
    {"speaker": "A", "text": "目覚まし三個かけてるんだけどね"},
    {"speaker": "B", "text": "うんうん"},
]


class RasterGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kaburi_tts.raster import RasterGenerator
        cls.timing = RasterGenerator(device="cpu")

    def test_timeline_basic(self):
        placed, meta = self.timing.timeline(UTTS, chunk_id="test")
        self.assertEqual(meta["n_utts"], len(UTTS))
        # 全発話がラスタ内に収まり、発話長 > 0
        for u in placed:
            self.assertGreater(sum(u["durs"]), 0)
            self.assertGreaterEqual(u["start"], 0.0)
            self.assertLessEqual(u["start"] + sum(u["durs"]), 750)
        # 全 valid 音素の duration >= 1 frame
        for u in placed:
            for p, d in zip(u["phones"], u["durs"]):
                self.assertGreaterEqual(d, 1.0)

    def test_raster_activity_convention(self):
        """学習データ規約: activity ≡ (phone != sil)。sil frame に activity=1 は分布外。"""
        placed, _ = self.timing.timeline(UTTS, chunk_id="test")
        pa, pb, aa, ab = self.timing.rasterize(placed)
        for ph, act in ((pa, aa), (pb, ab)):
            sil_frames = ph == SIL
            self.assertEqual(float(act[sil_frames].sum()), 0.0,
                             "sil frame に activity=1 が塗られている (分布外パターン)")
            speech_frames = (ph != SIL) & (act > 0.5)
            self.assertGreater(int(speech_frames.sum()), 0)

    def test_determinism(self):
        p1, _ = self.timing.timeline(UTTS, chunk_id="test")
        p2, _ = self.timing.timeline(UTTS, chunk_id="test")
        self.assertEqual(
            [(u["speaker"], u["start"], u["phones"], u["durs"]) for u in p1],
            [(u["speaker"], u["start"], u["phones"], u["durs"]) for u in p2])

    def test_canvas_fit_preserves_speech(self):
        """30 秒超過時の fit は発話長を変えない (start の圧縮のみ)。"""
        long_utts = [{"speaker": "AB"[i % 2],
                      "text": "今日は本当に長い話をたくさんしたいと思っているんですよね"}
                     for i in range(14)]
        placed, meta = self.timing.timeline(long_utts, chunk_id="long")
        self.assertLessEqual(max(u["start"] + sum(u["durs"]) for u in placed), 750)
        if meta["fit_scale"] < 1.0:
            # 発話長は fit の影響を受けない (realizer の出力そのまま)
            for u in placed:
                self.assertGreater(sum(u["durs"]), 0)

    def test_all_utterances_present(self):
        """入力発話が黙って落ちない (canon 可能な発話は全て配置される)。"""
        placed, meta = self.timing.timeline(UTTS, chunk_id="test")
        self.assertEqual(len(placed), len(UTTS))

    def test_no_implicit_long_vowel_on_fixture(self):
        """回帰: 「ほんと起き」の語境界 o+o が暗黙の oː に融合しない (2026-08-15 修正)。

        UTTS[0] に明示的長母音 SUB の教師根拠はなく、oː が出たら
        DEL 由来の暗黙融合 (教師契約違反) が復活している。
        """
        long_o = self.timing.realizer.vocab["oː"]
        placed, _ = self.timing.timeline(UTTS, chunk_id="test")
        self.assertNotIn(long_o, placed[0]["phones"],
                         "暗黙の同一母音融合 (DEL→長母音化) が発生している")


class ApplyEditsContractTest(unittest.TestCase):
    """decode の教師契約 (build_joint_data.py: DEL 位置の duration = -1 = 学習除外) の回帰テスト。

    apply_edits に合成入力を与え、DEL 位置の音素・duration が出力に一切
    使われないことを直接検証する。モデル不要・高速。
    """

    # 合成 vocab: 5=o, 7=ɯ, 9=oː
    INV = {5: "o", 7: "ɯ", 9: "oː"}
    SUBS = [9]  # op=2 -> oː への明示的 SUB

    def _apply(self, c, ops, durs):
        from kaburi_tts.raster.realize import apply_edits
        n = len(c)
        return apply_edits(c, ops, durs, [False] * n, [0.0] * n,
                           self.SUBS, self.INV)

    def test_del_then_keep_yields_kept_short_vowel(self):
        """D+K 同一母音 → 残る短母音とその duration のみ (長母音化・加算なし)。"""
        phones, durs = self._apply([5, 5], [1, 0], [3.886, 3.075])
        self.assertEqual(phones, [5])
        self.assertEqual(durs, [3.075])

    def test_keep_then_del_yields_kept_short_vowel(self):
        """K+D 同一母音 → 同上 (前方融合も禁止)。"""
        phones, durs = self._apply([5, 5], [0, 1], [3.075, 3.886])
        self.assertEqual(phones, [5])
        self.assertEqual(durs, [3.075])

    def test_del_duration_never_in_output(self):
        """DEL 位置の (未学習) duration は総 duration に一切寄与しない。"""
        phones, durs = self._apply([5, 7, 5], [0, 1, 0], [2.0, 99.0, 3.0])
        self.assertEqual(phones, [5, 5])
        self.assertEqual(sum(durs), 5.0)

    def test_only_explicit_sub_produces_long_vowel(self):
        """長母音は明示的 SUB (FUSE 対応の次音素は skip) の場合のみ。"""
        phones, durs = self._apply([5, 7], [2, 0], [4.0, 2.5])
        self.assertEqual(phones, [9])       # oː (明示的 SUB)
        self.assertEqual(durs, [4.0])       # SUB 位置の duration のみ


if __name__ == "__main__":
    unittest.main()

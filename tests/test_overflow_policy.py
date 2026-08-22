import sys
import unittest
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import synth_with_predictor as pred  # noqa: E402
import synth_from_text as synth_text  # noqa: E402
from kaburi_tts.placement import baselines as stat  # noqa: E402


class PredictorOverflowPolicyTest(unittest.TestCase):
    def _timeline_inputs(self):
        # Two same-speaker utterances.  Each PHONE is 30 frames and the
        # predicted pre-utterance silences are 10 and 20 frames.
        durations = torch.tensor([10.0, 30.0, 20.0, 30.0])
        gaps = torch.tensor([0.0, 0.0, 0.0, 0.0])
        token_type = torch.tensor([
            pred.TOKEN_TYPE_FIRST_SIL,
            pred.TOKEN_TYPE_PHONE,
            pred.TOKEN_TYPE_PRE_SIL,
            pred.TOKEN_TYPE_PHONE,
        ])
        channel = torch.tensor([0, 0, 0, 0])
        utt = torch.tensor([0, 0, 1, 1])
        mask = torch.ones(4, dtype=torch.bool)
        return durations, gaps, token_type, channel, utt, mask, [0, 2]

    def test_gap_only_preserves_phone_spans(self):
        args = self._timeline_inputs()
        timeline, info = pred._fit_timeline_by_gap_scale(
            *args, T=80, gap_blend=0.0, fit_margin=5,
            overflow_policy="gap-only",
        )
        spans = timeline["token_ends"] - timeline["token_starts"]
        self.assertAlmostEqual(float(spans[1]), 30.0)
        self.assertAlmostEqual(float(spans[3]), 30.0)
        self.assertTrue(info["gap_fit_used"])
        self.assertTrue(info["gap_fit_success"])
        self.assertLess(info["gap_scale"], 1.0)

    def test_error_rejects_timeline_fit(self):
        args = self._timeline_inputs()
        with self.assertRaises(pred.RasterOverflowError):
            pred._fit_timeline_by_gap_scale(
                *args, T=80, gap_blend=0.0, fit_margin=5,
                overflow_policy="error",
            )

    def test_duration_audit_never_retimes_tokens(self):
        durations = torch.tensor([10.0, 80.0])
        token_type = torch.tensor([pred.TOKEN_TYPE_FIRST_SIL, pred.TOKEN_TYPE_PHONE])
        channel = torch.tensor([0, 0])
        mask = torch.ones(2, dtype=torch.bool)
        fitted, info = pred._fit_durations_to_T(
            durations, token_type, channel, mask, 50,
            overflow_policy="gap-only",
        )
        self.assertEqual(float(fitted[0]), 10.0)
        self.assertEqual(float(fitted[1]), 80.0)
        self.assertEqual(info["ch0_sil_reduced"], 0.0)
        self.assertEqual(info["ch0_clamped"], 0.0)
        fitted_error, _ = pred._fit_durations_to_T(
            durations, token_type, channel, mask, 50,
            overflow_policy="error",
        )
        self.assertTrue(torch.equal(fitted_error, durations))

    def test_internal_sil_is_inactive(self):
        timeline = {
            "token_starts": torch.tensor([0.0, 3.0]),
            "token_ends": torch.tensor([3.0, 7.0]),
        }
        p0, _, a0, _ = pred._timeline_to_raster(
            timeline,
            torch.tensor([pred.TOKEN_TYPE_PHONE, pred.TOKEN_TYPE_PHONE]),
            torch.tensor([1, 2]),
            torch.tensor([0, 0]),
            torch.ones(2, dtype=torch.bool),
            T=20,
            phone_vocab_size=80,
        )
        self.assertTrue(torch.equal(p0[:3], torch.ones(3, dtype=torch.long)))
        self.assertEqual(float(a0[:3].sum()), 0.0)
        self.assertEqual(float(a0[3:7].sum()), 4.0)

    def test_dialog_id_seed_is_stable_across_subsets(self):
        self.assertEqual(synth_text._resolve_dialog_seed(0, 61, "c062", "order"), 61)
        self.assertEqual(synth_text._resolve_dialog_seed(0, 0, "c062", "dialog-id"), 61)
        self.assertEqual(synth_text._resolve_dialog_seed(10, 0, "c083", "dialog-id"), 92)

    def test_duration_override_targets_phone_tokens_by_utterance(self):
        durations = torch.tensor([8.0, 2.0, 3.0, 9.0, 4.0])
        token_type = torch.tensor([
            pred.TOKEN_TYPE_FIRST_SIL,
            pred.TOKEN_TYPE_PHONE,
            pred.TOKEN_TYPE_PHONE,
            pred.TOKEN_TYPE_PRE_SIL,
            pred.TOKEN_TYPE_PHONE,
        ])
        utt = torch.tensor([0, 0, 0, 1, 1])
        fitted = pred._apply_phone_duration_override(
            durations, token_type, utt, torch.ones(5, dtype=torch.bool),
            [[5, 6], [7]],
        )
        self.assertTrue(torch.equal(fitted, torch.tensor([8.0, 5.0, 6.0, 9.0, 7.0])))

    def test_duration_override_uses_global_dialog_rank(self):
        durations = torch.tensor([8.0, 2.0, 9.0, 3.0])
        token_type = torch.tensor([
            pred.TOKEN_TYPE_FIRST_SIL,
            pred.TOKEN_TYPE_PHONE,
            pred.TOKEN_TYPE_FIRST_SIL,
            pred.TOKEN_TYPE_PHONE,
        ])
        # Both channels use local utt_index=0, while dialog ranks are distinct.
        utt = torch.tensor([0, 0, 0, 0])
        rank = torch.tensor([0, 0, 1, 1])
        fitted = pred._apply_phone_duration_override(
            durations, token_type, utt, torch.ones(4, dtype=torch.bool),
            [[5], [7]], dialog_rank=rank,
        )
        self.assertTrue(torch.equal(fitted, torch.tensor([8.0, 5.0, 9.0, 7.0])))


class StatisticalOverflowPolicyTest(unittest.TestCase):
    def _inputs(self):
        utts = [
            {"speaker": "A", "text": "a", "phone_ids": [2] * 10},
            {"speaker": "A", "text": "b", "phone_ids": [2] * 10},
        ]
        q = {"p05": 10, "p10": 10, "median": 10, "p90": 10, "p95": 10}
        stats = {
            "fpp_by_nphones": {"10": 3.0},
            "fpp_tail": 3.0,
            "gap_by_transition": {"A_to_A": q},
        }
        return utts, stats

    def test_gap_only_preserves_utterance_durations(self):
        utts, stats = self._inputs()
        placements, info = stat.make_statistical_placements(
            utts, T=80, seed=7, start_offset=10, end_margin=10,
            overlap_prob=0.0, duration_scale=1.0, stats=stats,
            overflow_policy="gap-only", return_fit_info=True,
        )
        self.assertEqual([p["dur"] for p in placements], [30, 30])
        self.assertTrue(info["gap_fit_used"])
        self.assertFalse(info["global_scale_used"])

    def test_error_rejects_statistical_fit(self):
        utts, stats = self._inputs()
        with self.assertRaises(stat.PlacementOverflowError):
            stat.make_statistical_placements(
                utts, T=80, seed=7, start_offset=10, end_margin=10,
                overlap_prob=0.0, duration_scale=1.0, stats=stats,
                overflow_policy="error",
            )


if __name__ == "__main__":
    unittest.main()

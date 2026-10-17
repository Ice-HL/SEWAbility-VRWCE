"""Tests for the public spatial estimator's data-driven bandwidth rule."""

import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import pandas as pd


SCRIPT = (Path(__file__).resolve().parents[1] / "analysis-code"
          / "02_spatial_analysis" / "cycle_balanced_wrist_regions.py")
SPEC = importlib.util.spec_from_file_location("spatial_analysis", SCRIPT)
spatial = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(spatial)


def scores_with_winners(*winners):
    return pd.DataFrame([
        {
            "phase": phase,
            "hand": hand,
            "bandwidth_sw": bandwidth,
            "contributing_cycles": 100 + group_index,
            "mean_heldout_cycle_log_score": 0.0 if bandwidth in winners else -1.0,
        }
        for group_index, (phase, hand) in enumerate(
            (phase, hand) for phase in spatial.PHASES for hand in spatial.HANDS
        )
        for bandwidth in spatial.BANDWIDTH_CANDIDATES_SW
    ])


class BandwidthSelectionTests(unittest.TestCase):
    def test_exactly_thirty_regular_candidates(self):
        expected = tuple(round(index / 100, 2) for index in range(1, 31))
        self.assertEqual(spatial.BANDWIDTH_CANDIDATES_SW, expected)
        self.assertEqual(len(set(expected)), 30)

    def test_argmax_is_not_a_fixed_winner(self):
        for winner in (0.01, 0.02, 0.04, 0.17, 0.30):
            with self.subTest(winner=winner):
                aggregate, selected = spatial.aggregate_bandwidth_scores(
                    scores_with_winners(winner)
                )
                self.assertEqual(selected, winner)
                self.assertEqual(len(aggregate), 30)
                self.assertEqual(aggregate.selected_shared_bandwidth.sum(), 1)
                self.assertEqual(
                    aggregate.loc[aggregate.score_rank.eq(1), "bandwidth_sw"].tolist(),
                    [winner],
                )

    def test_public_selector_uses_the_scored_winner(self):
        data = pd.DataFrame([
            {"phase": phase, "hand": hand, "cycle_key": "v:1"}
            for phase in spatial.PHASES for hand in spatial.HANDS
        ])
        eligible = {phase: pd.Index(["v:1"]) for phase in spatial.PHASES}
        with mock.patch.object(
            spatial, "cycle_histograms", return_value=(np.ones((1, 1, 1)), ["v:1"])
        ), mock.patch.object(
            spatial, "smooth_stack", side_effect=lambda stack, bandwidth: bandwidth
        ), mock.patch.object(
            spatial, "leave_one_video_out_score",
            side_effect=lambda stack, keys, bandwidth: (-(bandwidth - 0.21) ** 2, 0.0),
        ):
            selected, scores = spatial.select_shared_bandwidth(
                data, eligible, np.array([0, 1]), np.array([0, 1])
            )
        self.assertEqual(selected, 0.21)
        self.assertEqual(len(scores), 180)
        self.assertEqual(scores.selected_shared_bandwidth.sum(), 6)

    def test_smallest_exact_tie_independent_of_input_order(self):
        scores = scores_with_winners(0.02, 0.21).sample(frac=1, random_state=123)
        aggregate, selected = spatial.aggregate_bandwidth_scores(scores)
        self.assertEqual(selected, 0.02)
        self.assertEqual(aggregate.score_rank.eq(1).sum(), 2)

    def test_near_tie_is_not_an_exact_tie(self):
        scores = scores_with_winners(0.02)
        scores.loc[scores.bandwidth_sw.eq(0.21), "mean_heldout_cycle_log_score"] = 1e-14
        _, selected = spatial.aggregate_bandwidth_scores(scores)
        self.assertEqual(selected, 0.21)

    def test_group_means_have_equal_not_cycle_count_weights(self):
        scores = scores_with_winners()
        first_group = scores.phase.eq("P1") & scores.hand.eq("left")
        scores.loc[scores.bandwidth_sw.eq(0.02), "mean_heldout_cycle_log_score"] = 0.0
        scores.loc[scores.bandwidth_sw.eq(0.21), "mean_heldout_cycle_log_score"] = 1.0
        scores.loc[first_group & scores.bandwidth_sw.eq(0.21),
                   "mean_heldout_cycle_log_score"] = -10.0
        scores.loc[first_group, "contributing_cycles"] = 1
        scores.loc[~first_group, "contributing_cycles"] = 1000
        aggregate, selected = spatial.aggregate_bandwidth_scores(scores)
        self.assertEqual(selected, 0.02)
        self.assertAlmostEqual(
            aggregate.loc[aggregate.bandwidth_sw.eq(0.21),
                          "equal_group_mean_loo_log_score"].item(), -5 / 6,
        )

    def test_reject_incomplete_candidates_or_group_scores(self):
        scores = scores_with_winners(0.02)
        invalid = [
            scores[~scores.bandwidth_sw.eq(0.30)],
            scores.iloc[1:],
            pd.concat([scores, scores.iloc[[0]]], ignore_index=True),
        ]
        for frame in invalid:
            with self.subTest(rows=len(frame)), self.assertRaises(ValueError):
                spatial.aggregate_bandwidth_scores(frame)
        scores.loc[0, "mean_heldout_cycle_log_score"] = np.nan
        with self.assertRaises(ValueError):
            spatial.aggregate_bandwidth_scores(scores)

    def test_historical_edges_remain_exact_inside_whole_cell_padding(self):
        data = pd.DataFrame({
            "x_from_needle_sw": [-0.3, 1.1],
            "y_from_needle_sw": [-0.8, 0.4],
        })
        x_edges, y_edges = spatial.make_edges(data)
        cell = spatial.CELL_SIZE_SW
        margin = spatial.HISTORICAL_GRID_MARGIN_SW
        pad = int(np.ceil((4 * max(spatial.BANDWIDTH_CANDIDATES_SW) - margin) / cell))
        for name, actual in (("x_from_needle_sw", x_edges),
                             ("y_from_needle_sw", y_edges)):
            lower = np.floor((data[name].min() - margin) / cell) * cell
            upper = np.ceil((data[name].max() + margin) / cell) * cell
            historical = np.arange(lower, upper + 0.5 * cell, cell)
            np.testing.assert_array_equal(actual[pad:-pad], historical)
            np.testing.assert_allclose(np.diff(actual), cell, rtol=0, atol=1e-14)

    def test_histograms_remain_cycle_equal_after_smoothing(self):
        data = pd.DataFrame({
            "cycle_key": ["a:1", "b:1", "b:1", "b:1"],
            "x_from_needle_sw": [0.0, 0.1, 0.1, 0.1],
            "y_from_needle_sw": [0.0, 0.1, 0.1, 0.1],
        })
        edges = spatial.make_edges(data)
        stack, keys = spatial.cycle_histograms(data, *edges)
        self.assertEqual(keys, ["a:1", "b:1"])
        np.testing.assert_allclose(stack.sum(axis=(1, 2)), 1.0)
        np.testing.assert_allclose(
            spatial.smooth_stack(stack, 0.30).sum(axis=(1, 2)), 1.0, atol=2e-7,
        )


if __name__ == "__main__":
    unittest.main()

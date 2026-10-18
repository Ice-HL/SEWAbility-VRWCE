"""Regression tests for midpoint assignment to half-open work-cycle intervals."""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd


SCRIPT = (Path(__file__).resolve().parents[1] / "analysis-code"
          / "01_video_analysis" / "06_map_segment_clusters_to_work_cycles.py")
SPEC = importlib.util.spec_from_file_location("cycle_mapping", SCRIPT)
mapping = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mapping)


def candidate_cycles():
    return pd.DataFrame({
        "Video_ID": ["1", "1"],
        "Cycle_Start_Frame": [0, 10],
        "Cycle_End_Frame": [10, 20],
        "FPS": [20.0, 20.0],
        "Cycle_ID": ["first", "second"],
        "Cycle_Duration_sec": [0.5, 0.5],
    })


def motion_windows(centers, clusters=None):
    if clusters is None:
        clusters = [1] * len(centers)
    return pd.DataFrame({
        "Video_ID": ["1"] * len(centers),
        "Segment_ID": [f"segment_{index + 1}" for index in range(len(centers))],
        "Segment_Center_Frame": centers,
        "Segment_Start_Frame": [center - 0.5 for center in centers],
        "Segment_End_Frame": [center + 0.5 for center in centers],
        "Cluster": clusters,
    })


class CycleMappingTests(unittest.TestCase):
    def map_windows(self, windows):
        with tempfile.TemporaryDirectory(prefix="cycle_mapping_test_") as folder:
            with mock.patch.object(mapping, "OUTPUT_DIR", folder), redirect_stdout(io.StringIO()):
                result = mapping.map_segments_to_cycles(windows, candidate_cycles())
            saved = pd.read_csv(Path(folder) / "segment_to_workcycle_mapping.csv")
            self.assertEqual(len(saved), len(result))
            self.assertEqual(saved["Cluster"].tolist(), result["Cluster"].tolist())
            return result

    def test_shared_boundary_belongs_only_to_following_cycle(self):
        result = self.map_windows(motion_windows([10.0]))
        self.assertEqual(len(result), 1)
        self.assertEqual(result["Cycle_ID"].tolist(), ["second"])
        self.assertEqual(result["Relative_Center_Percent"].tolist(), [0.0])

    def test_cycle_start_is_included(self):
        result = self.map_windows(motion_windows([0.0, 10.0]))
        self.assertEqual(result["Cycle_ID"].tolist(), ["first", "second"])
        self.assertEqual(result["Segment_ID"].tolist(), ["segment_1", "segment_2"])

    def test_final_cycle_end_is_excluded(self):
        result = self.map_windows(motion_windows([19.5, 20.0]))
        self.assertEqual(len(result), 1)
        self.assertEqual(result["Segment_ID"].tolist(), ["segment_1"])
        self.assertEqual(result["Cycle_ID"].tolist(), ["second"])

    def test_regular_windows_retain_counts_and_cluster_assignments(self):
        result = self.map_windows(motion_windows(
            [0.5, 5.5, 9.5, 10.5, 15.5, 19.5],
            [1, 2, 1, 1, 3, 2],
        ))
        self.assertEqual(len(result), 6)
        self.assertEqual(result.groupby("Cycle_ID").size().to_dict(),
                         {"first": 3, "second": 3})
        self.assertEqual(result["Cluster"].value_counts().to_dict(), {1: 3, 2: 2, 3: 1})
        self.assertFalse(result.duplicated(["Video_ID", "Segment_ID"]).any())


if __name__ == "__main__":
    unittest.main()

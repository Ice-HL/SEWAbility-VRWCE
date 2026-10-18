"""Tests for the numerical-only VR entry point and CLI compatibility.

The synthetic JSON records used here are generated in a temporary directory;
no study data, original video, or participant record is included in this test.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'analysis-code/04_vr_outcome_analysis/analyze_vrwce_sessions.py'
SPEC = importlib.util.spec_from_file_location('vr_analysis_test_module', CORE)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def write_synthetic_sessions(root: Path) -> list[str]:
    mappings = []
    for label in ('S1', 'S2', 'U1', 'U2'):
        folder = root / label
        folder.mkdir()
        samples = []
        transfer_samples = []
        for i in range(11):
            timestamp = f'2000-01-01T00:00:0{i // 10}.{(i % 10) * 100:03d}'
            left = {'x': -0.137, 'y': 0.0, 'z': i * 0.005}
            right = {'x': 0.137, 'y': 0.0, 'z': i * 0.005}
            samples.append({'time': timestamp, 'hasLeft': True, 'hasRight': True,
                            'leftTableLocalPosition': left,
                            'rightTableLocalPosition': right,
                            'leftInLeftZone': True, 'rightInRightZone': True})
            transfer_samples.append({'time': timestamp, 'tableLocalPosition': left})
        episode = {'startTime': samples[0]['time'], 'endTime': samples[-1]['time'],
                   'hand': 'Left', 'samples': transfer_samples}
        transfer = {'sessionId': label, 'grabSessions': [episode]}
        zones = {}
        for side, x in [('Left', -0.137), ('Right', 0.137)]:
            zones[f'push{side}Zone'] = {
                'width': 0.028, 'length': 0.056,
                'tableLocalCenter': {'x': x, 'y': 0.0, 'z': 0.0},
                'tableLocalForward': {'x': 0.0, 'y': 0.0, 'z': 1.0},
                'tableLocalRight': {'x': 1.0, 'y': 0.0, 'z': 0.0},
            }
        push = {'sessionId': label, 'pushCycleCount': 1, 'needleSafetyEvents': [],
                'pushCycles': [{'startTime': samples[0]['time'],
                                'endTime': samples[-1]['time'],
                                'usedLeft': True, 'usedRight': True,
                                'isSingleHand': False, 'samples': samples}], **zones}
        metadata = {'sessionId': label, 'endReason': 'completed',
                    'isPartialSession': False, 'isSimulated': label.startswith('S')}
        for name, data in [('phase1_grab.json', transfer), ('phase2_push.json', push),
                           ('phase3_return.json', transfer), ('session_meta.json', metadata)]:
            (folder / name).write_text(json.dumps(data), encoding='utf-8')
        mappings.extend(['--session', f'{label}={folder}'])
    return mappings


class VRAnalysisTests(unittest.TestCase):
    def test_core_import_does_not_import_matplotlib(self):
        program = '''
import builtins, importlib.util, sys
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "matplotlib" or name.startswith("matplotlib."):
        raise AssertionError("Matplotlib must not load during numerical import")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
spec = importlib.util.spec_from_file_location("vr_core_without_plotting", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not any(name == "matplotlib" or name.startswith("matplotlib.") for name in sys.modules)
'''
        result = subprocess.run([sys.executable, '-B', '-c', program, str(CORE)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_existing_cli_options_and_defaults(self):
        arguments = ANALYSIS.parse_arguments([
            '--session', 'S1=a', '--session', 'S2=b', '--session', 'U1=c',
            '--session', 'U2=d', '--profile', 'U1=Researcher-operated',
            '--output-dir', 'out', '--figure-stem', 'main',
            '--supplementary-stem', 'stability', '--skip-figures',
        ])
        self.assertTrue(arguments.skip_figures)
        self.assertEqual(arguments.figure_stem, 'main')
        self.assertEqual(arguments.supplementary_stem, 'stability')
        self.assertEqual(arguments.profile, ['U1=Researcher-operated'])
        defaults = ANALYSIS.parse_arguments(['--session', 'S1=a'])
        self.assertFalse(defaults.skip_figures)
        self.assertEqual(defaults.figure_stem, 'Fig5_VRWCE_technical_sessions')
        self.assertEqual(defaults.supplementary_stem, 'VRWCE_higher_order_stability')

    def test_core_numeric_helpers(self):
        points = ANALYSIS.np.asarray([[0., 0., 0.], [0., 0., 0.5], [0., 0., 1.]])
        times = ANALYSIS.np.asarray([0., 0.5, 1.])
        result = ANALYSIS.basic_trajectory_metrics(times, points)
        self.assertTrue(result['basic_eligible'])
        self.assertEqual(result['path_m'], 1.)
        self.assertEqual(result['path_efficiency'], 1.)
        self.assertEqual(ANALYSIS.BASIC_MIN_SAMPLES, 3)
        self.assertEqual(ANALYSIS.HIGHER_MIN_SAMPLES, 10)

    def test_skip_figures_completes_with_matplotlib_blocked(self):
        program = '''
import builtins, runpy, sys
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "matplotlib" or name.startswith("matplotlib.") or name == "plot_vrwce_outputs":
        raise AssertionError("Numerical-only execution must not import plotting")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
core = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(core, run_name="__main__")
assert not any(name == "matplotlib" or name.startswith("matplotlib.") for name in sys.modules)
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mappings = write_synthetic_sessions(root)
            output = root / 'out'
            result = subprocess.run([sys.executable, '-B', '-c', program, str(CORE), *mappings,
                                     '--output-dir', str(output), '--skip-figures'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(list(output.glob('*.csv'))), 6)
            self.assertTrue((output / 'analysis_qa.json').is_file())
            self.assertFalse(list(output.glob('*.png')))
            with (output / 'session_summary.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row['p2_recorded_action_count'] for row in rows], ['1']*4)
            self.assertEqual([row['p2_basic_bilateral_action_count'] for row in rows], ['1']*4)


if __name__ == '__main__':
    unittest.main()

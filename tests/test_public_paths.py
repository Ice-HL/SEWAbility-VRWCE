"""Check portable artifact references without executing the clustering CLI."""
import ast
import os
from pathlib import Path
import re
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_SCRIPT = (
    REPOSITORY_ROOT
    / "analysis-code"
    / "01_video_analysis"
    / "02_cluster_video_level_features.py"
)


def load_path_helper():
    tree = ast.parse(CLUSTER_SCRIPT.read_text(encoding="utf-8"))
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "relative_output_path"
    )
    module = ast.Module(body=[helper], type_ignores=[])
    namespace = {"os": os}
    exec(compile(module, str(CLUSTER_SCRIPT), "exec"), namespace)
    return namespace["relative_output_path"]


class PublicPathTests(unittest.TestCase):
    def test_absolute_output_root_preserves_subdirectory(self):
        helper = load_path_helper()
        root = Path.cwd() / "example_results"
        artifact = root / "PCA90" / "cluster_assignment_k3.csv"
        self.assertEqual(helper(artifact, root), "PCA90/cluster_assignment_k3.csv")

    def test_relative_output_root_preserves_subdirectory(self):
        helper = load_path_helper()
        root = Path("example_results")
        artifact = root / "RawStandardized" / "cluster_assignment_k4.csv"
        self.assertEqual(
            helper(artifact, root), "RawStandardized/cluster_assignment_k4.csv"
        )

    def test_output_root_name_is_not_exported(self):
        helper = load_path_helper()
        root = Path.cwd() / "private machine folder" / "results"
        result = helper(root / "PCA95" / "cluster_assignment_k13.csv", root)
        self.assertEqual(result, "PCA95/cluster_assignment_k13.csv")
        self.assertFalse(os.path.isabs(result))
        self.assertNotIn("private machine folder", result)

    def test_summary_uses_portable_helper(self):
        tree = ast.parse(CLUSTER_SCRIPT.read_text(encoding="utf-8"))
        expressions = [
            value
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == "Assignment_File"
        ]
        self.assertEqual(len(expressions), 1)
        value = expressions[0]
        self.assertIsInstance(value, ast.Call)
        self.assertIsInstance(value.func, ast.Name)
        self.assertEqual(value.func.id, "relative_output_path")
        self.assertEqual([argument.id for argument in value.args], ["out_csv", "OUTPUT_DIR"])

    def test_no_hardcoded_personal_home_paths_in_public_text(self):
        # Generic public examples are allowed; user-specific local home roots are not.
        home_root = re.compile(
            r"/" + r"(?:Users|home)/[^/\s\"'`]+/|[A-Za-z]:\\" + r"Users\\"
        )
        bad = []
        for path in REPOSITORY_ROOT.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".md", ".csv", ".txt"}:
                if home_root.search(path.read_text(encoding="utf-8")):
                    bad.append(str(path.relative_to(REPOSITORY_ROOT)))
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()

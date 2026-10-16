import tempfile
import unittest
from pathlib import Path

from common.config import PROJECT_ROOT, get_device, load_yaml, resolve_path, set_seed


class ConfigTests(unittest.TestCase):
    def test_resolve_relative_path_is_under_project_root(self):
        self.assertEqual(resolve_path("data/x"), PROJECT_ROOT / "data" / "x")

    def test_resolve_absolute_path_is_unchanged(self):
        self.assertEqual(resolve_path(Path(tempfile.gettempdir())), Path(tempfile.gettempdir()))

    def test_shipped_configs_parse(self):
        for name in ("detector", "classifier", "datasets"):
            self.assertIsInstance(load_yaml(f"configs/{name}.yaml"), dict)

    def test_load_yaml_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_yaml("configs/does_not_exist.yaml")

    def test_load_yaml_rejects_non_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.yaml"
            p.write_text("- a\n- b\n")
            with self.assertRaises(ValueError):
                load_yaml(p)

    def test_set_seed_is_reproducible(self):
        import random

        set_seed(123)
        a = [random.random() for _ in range(3)]
        set_seed(123)
        b = [random.random() for _ in range(3)]
        self.assertEqual(a, b)

    def test_get_device_returns_known_value(self):
        self.assertTrue(get_device("auto").split(":")[0] in {"cpu", "cuda", "mps"})
        self.assertEqual(get_device("cpu"), "cpu")

    def test_get_device_rejects_garbage(self):
        # Without torch installed this falls back to cpu; with torch it raises.
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        with self.assertRaises(ValueError):
            get_device("tpu")


if __name__ == "__main__":
    unittest.main()

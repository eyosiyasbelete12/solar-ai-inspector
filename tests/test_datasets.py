import tempfile
import unittest
from pathlib import Path

from common.datasets import get_dataset, load_dataset_specs, validate_spec

CLASSIFICATION_YAML = """
datasets:
  soil:
    task: classification
    name: Test soiling set
    source_url: https://example.invalid/soil
    license: TODO
    image_dir: {root}/soil
    class_names: [Clean, Dirty]
    class_map: {{Clean: clean, Dirty: dusty}}
    label_semantics: dust only
    group_by: filename_regex
    group_arg: '^(flight\\d+)_'
  det:
    task: detection
    name: Test detection set
    source_url: https://example.invalid/det
    license: TODO
    image_dir: {root}/det/images
    annotation_dir: {root}/det/labels
    class_names: [solar_panel]
"""


class DatasetRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "soil" / "Clean").mkdir(parents=True)
        (self.root / "soil" / "Dirty").mkdir(parents=True)
        (self.root / "soil" / "Clean" / "flight1_a.jpg").write_bytes(b"x")
        (self.root / "det" / "images").mkdir(parents=True)
        (self.root / "det" / "labels").mkdir(parents=True)
        (self.root / "det" / "images" / "a.png").write_bytes(b"x")
        self.cfg = self.root / "datasets.yaml"
        self.cfg.write_text(CLASSIFICATION_YAML.format(root=self.root.as_posix()))

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_both_tasks(self):
        specs = load_dataset_specs(self.cfg)
        self.assertEqual(set(specs), {"soil", "det"})
        self.assertEqual(specs["soil"].class_map["Dirty"], "dusty")

    def test_valid_entry_has_no_errors_but_warns_about_license(self):
        issues = validate_spec(get_dataset("soil", self.cfg))
        self.assertFalse([i for i in issues if i.level == "error"])
        self.assertTrue(any("license" in i.message for i in issues if i.level == "warning"))

    def test_missing_image_dir_is_an_error(self):
        spec = get_dataset("soil", self.cfg)
        spec.image_dir = self.root / "nope"
        self.assertTrue(any("image_dir" in i.message for i in validate_spec(spec) if i.level == "error"))

    def test_classification_labels_must_map_to_clean_and_dusty(self):
        spec = get_dataset("soil", self.cfg)
        spec.class_map = {}
        spec.class_names = ["Clean", "Dirty"]
        self.assertTrue(any("exactly" in i.message for i in validate_spec(spec) if i.level == "error"))

    def test_regex_group_needs_capture_group(self):
        spec = get_dataset("soil", self.cfg)
        spec.group_arg = "flight"
        self.assertTrue(any("capture group" in i.message for i in validate_spec(spec)))

    def test_group_by_none_warns_about_leakage(self):
        spec = get_dataset("det", self.cfg)
        self.assertTrue(any("leak" in i.message for i in validate_spec(spec) if i.level == "warning"))

    def test_detection_requires_annotation_dir(self):
        spec = get_dataset("det", self.cfg)
        spec.annotation_dir = None
        self.assertTrue(any("annotation_dir" in i.message for i in validate_spec(spec) if i.level == "error"))

    def test_unknown_key_lists_available(self):
        with self.assertRaises(KeyError) as ctx:
            get_dataset("missing", self.cfg)
        self.assertIn("soil", str(ctx.exception))

    def test_missing_required_field_raises(self):
        bad = self.root / "bad.yaml"
        bad.write_text("datasets:\n  x:\n    task: detection\n")
        with self.assertRaises(ValueError):
            load_dataset_specs(bad)


if __name__ == "__main__":
    unittest.main()

"""Public annotation names remain compatible with existing ZIP/JSONL exports."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import prepare_data


def annotation_line():
    return json.dumps({
        "images": ["/workspace/datasets/RefCOCO/train2014/example.jpg"],
        "additional_paras": {"caption": "the red rectangle"},
        "solution": {"arguments": {"coordinate": [100, 200, 500, 700]}},
    }, ensure_ascii=False) + "\n"


def legacy_filename(name):
    if "_eval.jsonl" in name:
        return name.replace("_eval.jsonl", "_iousd_eval.jsonl")
    return name.replace(".jsonl", "_iousd.jsonl")


class DataPreparationTests(unittest.TestCase):
    def test_legacy_archive_exports_supported_public_names_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, output = root / "annotations.zip", root / "prepared"
            line = annotation_line()
            expected_sha256 = hashlib.sha256(line.encode()).hexdigest()
            with zipfile.ZipFile(archive, "w") as handle:
                for name in prepare_data.ARCHIVE_ANNOTATIONS:
                    old_name = legacy_filename(name)
                    handle.writestr(f"refcoco/{old_name}", line)
                    handle.writestr(f"__MACOSX/refcoco/._{old_name}", "not JSON; resource metadata")
                handle.writestr("refcoco/refcoco_30k_train_iousd.jsonl", line)
            with patch("sys.argv", ["prepare_data.py", "--archive", str(archive), "--output", str(output)]):
                prepare_data.main()
            self.assertEqual({path.name for path in output.glob("*.jsonl")}, prepare_data.ARCHIVE_ANNOTATIONS)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(set(manifest["files"]), prepare_data.ARCHIVE_ANNOTATIONS)
            for name, metadata in manifest["files"].items():
                self.assertEqual(metadata["rows"], 1)
                self.assertEqual(metadata["sha256"], expected_sha256)
                self.assertEqual((output / name).read_text(), line)

    def test_full_training_archive_is_prepared_under_the_default_training_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((Path(__file__).resolve().parents[1] / "configs/train/jag.json").read_text())
            required_name = Path(config["training"]["train_jsonl"]).name
            self.assertEqual(required_name, "refcoco_train.jsonl")
            line = annotation_line()
            row = json.loads(line)
            row["images"] = ["workspace/datasets/RefCOCO/train2014/example.jpg"]
            source_line = json.dumps(row, ensure_ascii=False) + "\n"
            for number, filename in enumerate((required_name, legacy_filename(required_name))):
                with self.subTest(filename=filename):
                    archive, output = root / f"{number}.zip", root / f"prepared-{number}"
                    with zipfile.ZipFile(archive, "w") as handle:
                        handle.writestr(f"annotations/{filename}", source_line * 3)
                        handle.writestr("annotations/refcoco_80k_train_iousd.jsonl", source_line)
                    with patch("sys.argv", ["prepare_data.py", "--archive", str(archive),
                                             "--output", str(output)]):
                        prepare_data.main()
                    manifest = json.loads((output / "manifest.json").read_text())
                    self.assertEqual(manifest["files"][required_name]["rows"], 3)
                    self.assertEqual((output / required_name).read_text(), line * 3)
                    self.assertEqual(manifest["files"]["refcoco_80k_train.jsonl"]["rows"], 1)

    def test_old_and_new_jsonl_inputs_produce_identical_public_name_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "refcoco_80k_train.jsonl"
            for number, filename in enumerate([legacy_filename(name), name]):
                source = root / filename
                source.write_text(annotation_line())
                output = root / str(number)
                with patch("sys.argv", ["prepare_data.py", "--jsonl", str(source), "--output", str(output)]):
                    prepare_data.main()
                self.assertTrue((output / name).is_file())
            self.assertEqual((root / "0" / name).read_bytes(), (root / "1" / name).read_bytes())

    def test_normalized_archive_alias_collision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collision.zip"
            with zipfile.ZipFile(path, "w") as handle:
                handle.writestr("refcoco_testA_eval.jsonl", annotation_line())
                handle.writestr("refcoco_testA_iousd_eval.jsonl", annotation_line())
            with zipfile.ZipFile(path) as handle, self.assertRaisesRegex(ValueError, "duplicate normalized"):
                prepare_data.safe_archive_members(handle)

    def test_published_manifest_names_match_the_archive_selection(self):
        path = Path(__file__).resolve().parents[1] / "data/manifest.json"
        manifest = json.loads(path.read_text())
        names = {entry["file"] for entry in manifest["files"]}
        self.assertEqual(names, prepare_data.ARCHIVE_ANNOTATIONS - {"refcoco_80k_train.jsonl"})


if __name__ == "__main__":
    unittest.main()

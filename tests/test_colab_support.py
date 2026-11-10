"""CPU-only checks for Colab storage and resume selection; no model training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("colab_support", ROOT / "tools/colab_support.py")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


class ColabSupportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def run_fixture(self, epoch=8, history_epoch=None, arch="dlavss_34"):
        run = self.root / "run"
        run.mkdir(exist_ok=True)
        cfg = {"arch": arch, "seed": 317, "input_res": 960,
               "train": {"batch_size": 8, "accum_steps": 2, "amp": "fp16", "num_workers": 2}}
        (run / "config.yaml").write_text(yaml.safe_dump(cfg))
        torch.save({"epoch": epoch, "step": 3232, "state_dict": {"weight": torch.ones(1)},
                    "optimizer": {"param_groups": [{"params": [0]}]}, "scaler": {"scale": 65536}},
                   run / "model_last.pth")
        (run / "history.jsonl").write_text(json.dumps({"epoch": history_epoch or epoch, "step": 3232}) + "\n")
        return run

    def test_rejects_unrelated_architecture(self):
        run = self.run_fixture(arch="dla_34")
        with self.assertRaisesRegex(RuntimeError, "Wrong run configuration"):
            support.validate_run(run, "R1_RESUME")

    def test_resumes_later_completed_epoch_without_requiring_epoch8(self):
        run = self.run_fixture(epoch=11)
        self.assertEqual(support.validate_run(run, "R1_RESUME")["epoch"], 11)

    def test_checkpoint_without_matching_evaluation_stops(self):
        run = self.run_fixture(epoch=9, history_epoch=8)
        before = (run / "model_last.pth").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "history disagree"):
            support.validate_run(run, "R1_RESUME")
        self.assertEqual(before, (run / "model_last.pth").read_bytes())

    def test_original_checkpoint_requires_known_full_hash(self):
        run = self.run_fixture()
        with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
            support.validate_run(run, "R1_RESUME", original=True)

    def test_existing_incomplete_directory_is_preserved(self):
        run = self.root / "incomplete"
        run.mkdir()
        (run / "log.txt").write_text("preserve me")
        with self.assertRaisesRegex(RuntimeError, "preserved"):
            support.prepare_run(run, "R2_FRESH", self.root, self.root / "unused.zip")
        self.assertEqual((run / "log.txt").read_text(), "preserve me")

    def test_unregistered_later_run_is_not_adopted(self):
        run = self.run_fixture(epoch=11)
        with self.assertRaisesRegex(RuntimeError, "no registered"):
            support.prepare_run(run, "R1_RESUME", self.root, self.root / "unused.zip")

    def test_registered_later_run_can_continue(self):
        run = self.run_fixture(epoch=11)
        path, identity = support.run_identity(run, "R1_RESUME")
        support.write_json(path, identity)
        self.assertEqual(support.prepare_run(run, "R1_RESUME", self.root, self.root / "unused.zip")["epoch"], 11)

    def test_archive_cannot_escape_destination(self):
        archive = self.root / "bad.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("../outside.txt", "bad")
        with self.assertRaisesRegex(RuntimeError, "Unsafe archive"):
            support.safe_extract(archive, self.root / "extract")
        self.assertFalse((self.root / "outside.txt").exists())

    def test_four_parts_reassemble_and_reject_corruption(self):
        destination = self.root / "download" / "r1_epoch8_checkpoint.zip"
        pieces = [b"a" * 20, b"b" * 20, b"c" * 20, b"last"]
        digest = hashlib.sha256(b"".join(pieces)).hexdigest()
        parts = []
        for i, value in enumerate(pieces, 1):
            name = f"r1_epoch8_checkpoint.zip.part{i:03d}"
            (self.root / name).write_bytes(value)
            parts.append({"name": name, "size": len(value), "sha256": hashlib.sha256(value).hexdigest()})
        manifest = {"filename": destination.name, "sha256": digest, "parts": parts}
        (self.root / "r1_epoch8_checkpoint.manifest.json").write_text(json.dumps(manifest))
        with mock.patch.object(support, "R1_ZIP_SHA", digest):
            support.assemble_r1_bundle(self.root, destination)
            self.assertEqual(destination.read_bytes(), b"".join(pieces))
            destination.unlink()
            (self.root / parts[0]["name"]).write_bytes(b"x" * 20)
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                support.assemble_r1_bundle(self.root, destination)
            self.assertFalse(destination.exists())

    def test_does_not_clone_into_existing_non_repository_data(self):
        target = self.root / "existing"
        target.mkdir()
        (target / "checkpoint.pth").write_text("preserve")
        with mock.patch.object(support, "checked") as checked:
            with self.assertRaisesRegex(RuntimeError, "not empty"):
                support.prepare_repo(target)
            checked.assert_not_called()

    def test_completed_run_does_not_start_another_training_process(self):
        run = self.run_fixture(epoch=15)
        with mock.patch.object(support, "checked") as checked:
            support.launch_training(self.root, run, "R1_RESUME", {}, self.root / "session.json")
            checked.assert_not_called()

    def test_coco_fingerprint_ignores_format_and_metadata(self):
        data = {"images": [{"id": 1}], "annotations": [{"id": 3}, {"id": 2}],
                "categories": [{"id": 1}], "info": {"description": "a"}}
        a = self.root / "a.json"
        b = self.root / "b.json"
        a.write_text(json.dumps(data))
        data["info"]["description"] = "b"
        data["annotations"].reverse()
        b.write_text(json.dumps(data, indent=2))
        self.assertEqual(support.coco_fingerprint(a), support.coco_fingerprint(b))


if __name__ == "__main__":
    unittest.main()

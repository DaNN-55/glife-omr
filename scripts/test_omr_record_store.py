import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omr_record_store import LabelSchema, OmrRecordStore
from run_local_omr_ui import PYTHON


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class OmrRecordStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.records_dir = self.root / "omr-records"
        self.image = self.root / "sample.png"
        self.image.write_bytes(PNG_1X1)
        self.vocab = self.root / "vocab.json"
        self.vocab.write_text(
            json.dumps(["BAR", "BEAT", "DUR_4", "N_S1_F0", "N_S1_F1", "END_BAR"]),
            encoding="utf-8",
        )
        self.checkpoint = self.root / "best.pt"
        self.checkpoint.write_bytes(b"checkpoint-v1")
        self.store = OmrRecordStore(
            self.records_dir,
            LabelSchema("guitar-tab-omr", 1, self.vocab),
            PYTHON,
            self.checkpoint,
        )
        self.predicted = "BAR BEAT DUR_4 N_S1_F0 END_BAR"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def record(self):
        return self.store.record_inference(
            source_image=self.image,
            predicted_tokens=self.predicted,
            source_label="unit-test.png",
            decode="greedy",
            constrained=True,
            max_decode_len=384,
        )

    def review(self, record_id: str, corrected: str, **kwargs):
        return self.store.review(
            record_id=record_id,
            label_source=kwargs.pop("label_source", "corrected_error"),
            corrected_tokens=corrected,
            categories=kwargs.pop("categories", ["omr_string_fret"]),
            include_training=kwargs.pop("include_training", True),
            **kwargs,
        )

    def test_inference_is_saved_before_review(self) -> None:
        record = self.record()
        metadata = json.loads((record.path / "record.json").read_text())

        self.assertEqual(metadata["status"], "unreviewed")
        self.assertEqual(metadata["source"], "unit-test.png")
        self.assertEqual(
            metadata["checkpointSha256"], hashlib.sha256(b"checkpoint-v1").hexdigest()
        )
        self.assertEqual((record.path / "predicted.tokens.txt").read_text().strip(), self.predicted)
        self.assertFalse((self.records_dir / "manifest.json").exists())

    def test_history_groups_identical_predictions_and_loads_review(self) -> None:
        reviewed = self.record()
        self.review(
            reviewed.record_id,
            self.predicted,
            label_source="verified_correct",
            categories=[],
        )
        self.record()

        history = self.store.list_records()
        detail = self.store.load_record(history[0]["recordId"])

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["duplicateCount"], 2)
        self.assertEqual(history[0]["status"], "reviewed")
        self.assertTrue(history[0]["trainingCandidate"])
        self.assertEqual(history[0]["createdAt"], detail["createdAt"])
        self.assertEqual(detail["predictedTokenText"], self.predicted)
        self.assertEqual(detail["correctedTokenText"], self.predicted)
        self.assertEqual(detail["labelSource"], "verified_correct")
        self.assertTrue((detail["recordPath"] / detail["inputFilename"]).is_file())

    def test_history_stays_readable_when_candidate_schema_is_old(self) -> None:
        record = self.record()
        self.review(
            record.record_id,
            self.predicted,
            label_source="verified_correct",
            categories=[],
        )
        manifest_path = self.records_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["labelSchema"]["version"] = 0
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        history = self.store.list_records()
        detail = self.store.load_record(record.record_id)

        self.assertTrue(history[0]["trainingCandidate"])
        self.assertTrue(detail["trainingCandidate"])

    def test_legacy_records_without_prediction_hashes_are_not_grouped(self) -> None:
        first = self.record()
        self.store.record_inference(
            source_image=self.image,
            predicted_tokens="BAR BEAT DUR_4 N_S1_F1 END_BAR",
            decode="greedy",
            constrained=True,
            max_decode_len=384,
        )
        for record_dir in (self.records_dir / "records").iterdir():
            metadata_path = record_dir / "record.json"
            metadata = json.loads(metadata_path.read_text())
            metadata.pop("imageSha256")
            metadata.pop("predictedTokensSha256")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        history = self.store.list_records()

        self.assertEqual(len(history), 2)
        self.assertIn(first.record_id, {item["recordId"] for item in history})

    def test_reuse_requires_same_image_checkpoint_and_decode_config(self) -> None:
        recorded = self.record()

        reused = self.store.find_reusable_inference(
            source_image=self.image,
            decode="greedy",
            constrained=True,
            max_decode_len=384,
        )
        different_config = self.store.find_reusable_inference(
            source_image=self.image,
            decode="greedy",
            constrained=True,
            max_decode_len=385,
        )

        self.assertEqual(reused["recordId"], recorded.record_id)
        self.assertIsNone(different_config)

        metadata_path = recorded.path / "record.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.pop("checkpointSha256")
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.assertIsNone(
            self.store.find_reusable_inference(
                source_image=self.image,
                decode="greedy",
                constrained=True,
                max_decode_len=384,
            )
        )

    def test_history_record_and_training_candidate_can_be_deleted_together(self) -> None:
        removable = self.record()
        self.store.delete_record(removable.record_id)
        self.assertFalse(removable.path.exists())

        candidate = self.record()
        self.review(
            candidate.record_id,
            self.predicted,
            label_source="verified_correct",
            categories=[],
        )
        self.store.delete_record(candidate.record_id)

        manifest = json.loads((self.records_dir / "manifest.json").read_text())
        self.assertFalse(candidate.path.exists())
        self.assertEqual(manifest["samples"], [])

    def test_delete_manifest_failure_restores_record_and_candidate(self) -> None:
        candidate = self.record()
        self.review(
            candidate.record_id,
            self.predicted,
            label_source="verified_correct",
            categories=[],
        )
        (self.records_dir / "manifest.json.tmp").mkdir()

        with self.assertRaises(OSError):
            self.store.delete_record(candidate.record_id)

        manifest = json.loads((self.records_dir / "manifest.json").read_text())
        self.assertTrue(candidate.path.exists())
        self.assertEqual(manifest["samples"][0]["id"], candidate.record_id)

    def test_delete_cleanup_failure_keeps_record_and_manifest_consistent(self) -> None:
        candidate = self.record()
        self.review(
            candidate.record_id,
            self.predicted,
            label_source="verified_correct",
            categories=[],
        )

        with patch("omr_record_store.shutil.rmtree", side_effect=OSError("cleanup failed")):
            self.store.delete_record(candidate.record_id)

        manifest = json.loads((self.records_dir / "manifest.json").read_text())
        self.assertFalse(candidate.path.exists())
        self.assertEqual(manifest["samples"], [])

    def test_corrected_error_and_verified_correct_can_enter_manifest(self) -> None:
        corrected_record = self.record()
        corrected = "BAR BEAT DUR_4 N_S1_F1 END_BAR"
        corrected_result = self.review(corrected_record.record_id, corrected)

        correct_record = self.record()
        correct_result = self.review(
            correct_record.record_id,
            self.predicted,
            label_source="verified_correct",
            categories=[],
        )

        manifest = json.loads((self.records_dir / "manifest.json").read_text())
        self.assertEqual(corrected_result.disposition, "candidate")
        self.assertEqual(correct_result.disposition, "candidate")
        self.assertEqual(
            [sample["labelSource"] for sample in manifest["samples"]],
            ["corrected_error", "verified_correct"],
        )

    def test_saved_conclusion_can_update_its_training_candidate(self) -> None:
        record = self.record()
        self.review(
            record.record_id,
            self.predicted,
            label_source="verified_correct",
            categories=[],
        )

        result = self.review(
            record.record_id,
            "BAR BEAT DUR_4 N_S1_F1 END_BAR",
            note="重新检查后修正品位",
        )

        manifest = json.loads((self.records_dir / "manifest.json").read_text())
        detail = self.store.load_record(record.record_id)
        self.assertEqual(result.disposition, "updated")
        self.assertEqual([sample["id"] for sample in manifest["samples"]], [record.record_id])
        self.assertEqual(manifest["samples"][0]["labelSource"], "corrected_error")
        self.assertEqual(detail["correctedTokenText"], "BAR BEAT DUR_4 N_S1_F1 END_BAR")
        self.assertEqual(detail["note"], "重新检查后修正品位")
        self.assertTrue(detail["trainingCandidate"])

    def test_failed_manifest_write_rolls_back_review_files(self) -> None:
        record = self.record()
        self.review(
            record.record_id,
            self.predicted,
            label_source="verified_correct",
            categories=[],
        )
        self.records_dir.chmod(0o555)
        try:
            with self.assertRaises(OSError):
                self.review(
                    record.record_id,
                    "BAR BEAT DUR_4 N_S1_F1 END_BAR",
                    note="不应保留",
                )
        finally:
            self.records_dir.chmod(0o755)

        detail = self.store.load_record(record.record_id)
        manifest = json.loads((self.records_dir / "manifest.json").read_text())
        self.assertEqual(detail["correctedTokenText"], self.predicted)
        self.assertEqual(detail["labelSource"], "verified_correct")
        self.assertEqual(detail["note"], "")
        self.assertEqual(manifest["samples"][0]["labelSource"], "verified_correct")

    def test_failed_first_candidate_write_removes_new_vocab(self) -> None:
        record = self.record()
        (self.records_dir / "manifest.json.tmp").mkdir()

        with self.assertRaises(OSError):
            self.review(
                record.record_id,
                self.predicted,
                label_source="verified_correct",
                categories=[],
            )

        detail = self.store.load_record(record.record_id)
        self.assertEqual(detail["status"], "unreviewed")
        self.assertIsNone(detail["correctedTokenText"])
        self.assertFalse((self.records_dir / "manifest.json").exists())
        self.assertFalse((self.records_dir / "vocab.json").exists())

    def test_review_without_training_does_not_change_manifest(self) -> None:
        record = self.record()
        result = self.review(
            record.record_id,
            "BAR BEAT DUR_4 N_S1_F1 END_BAR",
            include_training=False,
        )

        metadata = json.loads((record.path / "record.json").read_text())
        self.assertEqual(result.disposition, "reviewed")
        self.assertFalse(metadata["trainingRequested"])
        self.assertFalse((self.records_dir / "manifest.json").exists())

    def test_updating_without_training_removes_existing_candidate(self) -> None:
        record = self.record()
        self.review(record.record_id, "BAR BEAT DUR_4 N_S1_F1 END_BAR")

        result = self.review(
            record.record_id,
            self.predicted,
            label_source="verified_correct",
            categories=[],
            include_training=False,
        )

        manifest = json.loads((self.records_dir / "manifest.json").read_text())
        self.assertEqual(result.disposition, "reviewed")
        self.assertEqual(manifest["samples"], [])
        self.assertFalse(self.store.load_record(record.record_id)["trainingCandidate"])

    def test_duplicate_label_keeps_one_candidate(self) -> None:
        corrected = "BAR BEAT DUR_4 N_S1_F1 END_BAR"
        first = self.review(self.record().record_id, corrected)
        duplicate = self.review(self.record().record_id, corrected)

        manifest = json.loads((self.records_dir / "manifest.json").read_text())
        self.assertEqual(duplicate.disposition, "duplicate")
        self.assertEqual(duplicate.candidate_id, first.candidate_id)
        self.assertEqual(len(manifest["samples"]), 1)
        self.assertEqual(len(list((self.records_dir / "records").iterdir())), 2)

    def test_updating_candidate_reapplies_duplicate_check(self) -> None:
        corrected = "BAR BEAT DUR_4 N_S1_F1 END_BAR"
        first = self.review(self.record().record_id, corrected)
        second_record = self.record()
        self.review(
            second_record.record_id,
            self.predicted,
            label_source="verified_correct",
            categories=[],
        )

        duplicate = self.review(second_record.record_id, corrected)

        manifest = json.loads((self.records_dir / "manifest.json").read_text())
        self.assertEqual(duplicate.disposition, "duplicate")
        self.assertEqual(duplicate.candidate_id, first.candidate_id)
        self.assertEqual([sample["id"] for sample in manifest["samples"]], [first.candidate_id])

    def test_label_source_must_match_human_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "必须修改"):
            self.review(self.record().record_id, self.predicted)

        with self.assertRaisesRegex(ValueError, "必须与模型输出一致"):
            self.review(
                self.record().record_id,
                "BAR BEAT DUR_4 N_S1_F1 END_BAR",
                label_source="verified_correct",
                categories=[],
            )

    def test_incompatible_label_schema_is_rejected_for_training(self) -> None:
        record = self.record()
        self.review(record.record_id, "BAR BEAT DUR_4 N_S1_F1 END_BAR")
        incompatible = OmrRecordStore(
            self.records_dir,
            LabelSchema("guitar-tab-omr", 2, self.vocab),
            PYTHON,
            self.root / "best.pt",
        )
        next_record = incompatible.record_inference(
            source_image=self.image,
            predicted_tokens=self.predicted,
        )
        with self.assertRaisesRegex(ValueError, "Label Schema"):
            incompatible.review(
                record_id=next_record.record_id,
                label_source="corrected_error",
                corrected_tokens="BAR BEAT DUR_4 N_S1_F1 END_BAR",
                categories=["omr_string_fret"],
                include_training=True,
            )

    def test_invalid_token_structure_is_rejected_for_training(self) -> None:
        with self.assertRaisesRegex(ValueError, "Label Schema"):
            self.review(self.record().record_id, "BAR N_S1_F1 END_BAR")


if __name__ == "__main__":
    unittest.main()

import base64
import json
import tempfile
import unittest
from pathlib import Path

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
        self.store = OmrRecordStore(
            self.records_dir,
            LabelSchema("guitar-tab-omr", 1, self.vocab),
            PYTHON,
            self.root / "best.pt",
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
        self.assertEqual((record.path / "predicted.tokens.txt").read_text().strip(), self.predicted)
        self.assertFalse((self.records_dir / "manifest.json").exists())

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

    def test_duplicate_label_keeps_one_candidate(self) -> None:
        corrected = "BAR BEAT DUR_4 N_S1_F1 END_BAR"
        first = self.review(self.record().record_id, corrected)
        duplicate = self.review(self.record().record_id, corrected)

        manifest = json.loads((self.records_dir / "manifest.json").read_text())
        self.assertEqual(duplicate.disposition, "duplicate")
        self.assertEqual(duplicate.candidate_id, first.candidate_id)
        self.assertEqual(len(manifest["samples"]), 1)
        self.assertEqual(len(list((self.records_dir / "records").iterdir())), 2)

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

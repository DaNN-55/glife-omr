import unittest

from run_local_omr_ui import (
    INFER_SCRIPT,
    MODEL_DIR,
    PYTHON,
    RECORDS_DIR,
    ROOT,
    detect_image_suffix,
    normalize_error_categories,
    token_manifest_stats,
)


class OmrUiHelpersTest(unittest.TestCase):
    def test_local_player_asset_is_wired(self) -> None:
        soundfont = ROOT / "static/alphatab/soundfont/sonivox.sf2"
        page = (ROOT / "review/omr-workbench.html").read_text()
        self.assertGreater(soundfont.stat().st_size, 1_000_000)
        self.assertIn("/static/alphatab/soundfont/sonivox.sf2", page)
        self.assertIn("scoreApi.playPause()", page)

    def test_upload_and_clipboard_fallback_are_wired(self) -> None:
        page = (ROOT / "review/omr-workbench.html").read_text()
        self.assertIn("当前浏览器未授权读取剪贴板", page)
        self.assertIn("没有重复加入训练候选", page)
        self.assertIn("/api/records/review", page)
        self.assertIn("value=\"verified_correct\"", page)
        self.assertIn("id=\"includeTraining\"", page)
        self.assertIn('id="saveRecordReview" class="save-sample" type="button" disabled', page)

    def test_workbench_starts_without_a_bundled_sample(self) -> None:
        page = (ROOT / "review/omr-workbench.html").read_text()
        self.assertIn("select('empty')", page)

    def test_runtime_paths_stay_inside_project(self) -> None:
        self.assertEqual(PYTHON, ROOT / ".venv-omr/bin/python")
        self.assertEqual(INFER_SCRIPT, ROOT / "vendor/guitar-tab-omr/scripts/guitar_omr_infer.py")
        self.assertEqual(MODEL_DIR, ROOT / "models/guitar-tab-omr")
        self.assertEqual(RECORDS_DIR, ROOT / "data/omr-records")

    def test_detect_image_suffix_rejects_spoofed_content(self) -> None:
        self.assertEqual(detect_image_suffix(b"\x89PNG\r\n\x1a\nrest"), ".png")
        self.assertEqual(detect_image_suffix(b"\xff\xd8\xffrest"), ".jpg")
        self.assertEqual(detect_image_suffix(b"RIFF1234WEBPrest"), ".webp")
        self.assertIsNone(detect_image_suffix(b"not-an-image"))

    def test_token_manifest_stats_matches_training_schema(self) -> None:
        tokens = "BAR\nBEAT DUR_4 N_S1_F0\nEND_BAR\nBAR\nBEAT DUR_8 N_S2_FX\nEND_BAR\n"
        self.assertEqual(token_manifest_stats(tokens), (2, [0, 3]))

    def test_error_categories_accept_old_and_new_payloads(self) -> None:
        self.assertEqual(
            normalize_error_categories({"category": "omr_string_fret"}),
            ["omr_string_fret"],
        )
        self.assertEqual(
            normalize_error_categories(
                {"categories": ["omr_string_fret", "omr_technique", "omr_string_fret"]}
            ),
            ["omr_string_fret", "omr_technique"],
        )
        with self.assertRaisesRegex(ValueError, "至少选择一种"):
            normalize_error_categories({"categories": []})
        with self.assertRaisesRegex(ValueError, "有效的错误类型"):
            normalize_error_categories({"categories": ["unknown"]})


if __name__ == "__main__":
    unittest.main()

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
        soundfont = ROOT / "static/alphatab/soundfont/sonivox.sf3"
        page = (ROOT / "review/omr-workbench.html").read_text()
        self.assertGreater(soundfont.stat().st_size, 900_000)
        self.assertIn("/static/alphatab/soundfont/sonivox.sf3", page)
        self.assertIn("scoreApi.playPause()", page)
        self.assertIn("scoreApi.isLooping = !scoreApi.isLooping", page)
        self.assertIn('class="player-toolbar"', page)
        self.assertNotIn('id="speedButton"', page)
        self.assertIn('id="timbreButton"', page)
        self.assertIn('data-timbre="25"', page)
        self.assertNotIn('id="timbreSelect"', page)
        self.assertLess(page.index('id="playPauseButton"'), page.index('id="loopButton"'))
        self.assertLess(page.index('id="loopButton"'), page.index('id="timbreButton"'))
        self.assertLess(page.index('id="timbreButton"'), page.index('id="exportGpButton"'))
        self.assertIn("`\\\\instrument ${instrument}`", page)
        self.assertIn("playbackInstrument = playbackInstrument === 25 ? 27 : 25", page)
        self.assertIn("elements.timbreButton.addEventListener('click'", page)

    def test_decode_settings_help_explains_every_control(self) -> None:
        page = (ROOT / "review/omr-workbench.html").read_text()
        self.assertIn('aria-label="查看解码参数说明"', page)
        self.assertIn("greedy（默认）", page)
        self.assertIn("constrained beam", page)
        self.assertIn("Token constraints（默认开启）", page)
        self.assertIn("Max decode length（默认 384）", page)

    def test_current_preview_can_export_guitar_pro(self) -> None:
        page = (ROOT / "review/omr-workbench.html").read_text()
        self.assertIn('id="exportGpButton"', page)
        self.assertIn("下载 Guitar Pro 文件", page)
        self.assertIn("导出后请在 Guitar Pro 复核", page)
        self.assertIn("new alphaTab.exporter.Gp7Exporter()", page)
        self.assertIn("scoreApi.scoreLoaded.on(score =>", page)
        self.assertIn(
            "elements.correctedTokens.addEventListener('input', () => {\n"
            "      currentPreviewScore = null;\n"
            "      elements.exportGpButton.disabled = true;",
            page,
        )

    def test_upload_and_clipboard_fallback_are_wired(self) -> None:
        page = (ROOT / "review/omr-workbench.html").read_text()
        self.assertIn("当前浏览器未授权读取剪贴板", page)
        self.assertNotIn("摄像头输入", page)
        self.assertIn("没有重复加入训练候选", page)
        self.assertIn("/api/records/review", page)
        self.assertIn("value=\"verified_correct\"", page)
        self.assertNotIn('id="includeTraining"', page)
        self.assertNotIn("elements.includeTraining", page)
        self.assertIn("includeTraining: true", page)
        self.assertIn("人工结论保存后会更新训练候选；候选素材尚未用于训练", page)
        self.assertNotIn("可明确选择是否加入训练候选", page)
        self.assertIn('id="saveRecordReview" class="save-sample" type="button" disabled', page)

    def test_model_input_preview_is_wired(self) -> None:
        page = (ROOT / "review/omr-workbench.html").read_text()
        self.assertIn('id="rawImageView"', page)
        self.assertIn('id="modelInputView"', page)
        self.assertIn("async function createModelInputImage", page)
        self.assertIn("current.modelInputImage = await createModelInputImage", page)

    def test_workbench_starts_without_a_bundled_sample(self) -> None:
        page = (ROOT / "review/omr-workbench.html").read_text()
        self.assertIn("select('empty')", page)

    def test_history_can_list_and_reopen_saved_prediction_records(self) -> None:
        page = (ROOT / "review/omr-workbench.html").read_text()
        server = (ROOT / "scripts/run_local_omr_ui.py").read_text()

        self.assertIn('id="historyButton"', page)
        self.assertIn('title="历史记录" aria-label="历史记录"', page)
        self.assertNotIn("\n            历史记录\n", page)
        self.assertIn('id="historyDialog"', page)
        self.assertIn("max-height: min(516px, calc(82vh - 82px))", page)
        self.assertIn("fetch('/api/records')", page)
        self.assertIn("async function openHistoryRecord", page)
        self.assertIn("async function deleteHistoryRecord", page)
        self.assertIn("window.confirm", page)
        self.assertIn("method: 'DELETE'", page)
        self.assertIn("已打开历史记录 · 未重新运行 OMR", page)
        self.assertIn('route == "/api/records"', server)
        self.assertIn('route.startswith("/api/records/")', server)
        self.assertIn("def do_DELETE", server)
        self.assertIn('"reused": True', server)

    def test_saved_conclusion_stays_editable_and_auto_saves(self) -> None:
        page = (ROOT / "review/omr-workbench.html").read_text()

        self.assertIn("setTimeout(() => saveRecordReview(true), 500)", page)
        self.assertIn("window.addEventListener('beforeunload'", page)
        self.assertIn("有尚未保存的人工结论", page)
        self.assertIn("人工结论正在保存", page)
        self.assertIn("'立即保存'", page)
        self.assertIn("'保存中…'", page)
        self.assertIn("'已保存'", page)
        self.assertIn("'重试保存'", page)
        self.assertIn("'已保存结论'", page)
        self.assertIn("'训练候选'", page)
        self.assertNotIn("elements.correctedTokens.readOnly = verifiedCorrect || reviewed", page)

    def test_left_workflow_cards_share_a_sticky_sidebar(self) -> None:
        page = (ROOT / "review/omr-workbench.html").read_text()
        self.assertIn('<div class="sticky-controls">', page)
        self.assertIn(".sticky-controls { position: sticky;", page)
        self.assertIn(".sticky-controls { position: static;", page)
        self.assertIn('</aside>\n      </div>\n      <p class="model-footnote">', page)

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

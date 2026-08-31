#!/usr/bin/env python3
import argparse
import base64
import binascii
import json
import subprocess
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from omr_record_store import LabelSchema, OmrRecordStore, token_manifest_stats


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-omr/bin/python"
INFER_SCRIPT = ROOT / "vendor/guitar-tab-omr/scripts/guitar_omr_infer.py"
MODEL_DIR = ROOT / "models/guitar-tab-omr"
RECORDS_DIR = ROOT / "data/omr-records"
MAX_REQUEST_BYTES = 25 * 1024 * 1024
MAX_IMAGE_BYTES = 18 * 1024 * 1024
IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
RUN_LOCK = threading.Lock()
ERROR_CATEGORIES = {
    "omr_string_fret",
    "omr_duration_rhythm",
    "omr_technique",
    "omr_structure",
    "omr_metadata",
    "omr_other",
    "input_preprocessing",
    "decode_rule",
    "conversion_preview",
}


def normalize_error_categories(payload: dict) -> list[str]:
    values = payload.get("categories")
    if values is None:
        values = [payload.get("category")]
    if not isinstance(values, list) or not values:
        raise ValueError("请至少选择一种错误类型")

    categories = []
    for value in values:
        if not isinstance(value, str) or value not in ERROR_CATEGORIES:
            raise ValueError("请选择有效的错误类型")
        if value not in categories:
            categories.append(value)
    return categories


def detect_image_suffix(image_bytes: bytes) -> str | None:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    return None


class OmrHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_json(self, status: int, value: dict) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route not in {"/api/omr", "/api/records/review"}:
            self.send_json(404, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(400, {"error": "Invalid Content-Length"})
            return
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self.send_json(413, {"error": "图片请求必须小于 25 MB"})
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError as error:
            self.send_json(400, {"error": str(error)})
            return

        if route == "/api/records/review":
            try:
                self.send_json(200, self.review_record(payload))
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
            except Exception as error:
                self.send_json(500, {"error": f"{type(error).__name__}: {error}"})
            return

        try:
            decode = payload.get("decode", "greedy")
            constrained = payload.get("constrained", True)
            max_decode_len = int(payload.get("maxDecodeLen", 384))
            if decode not in {"greedy", "beam"}:
                raise ValueError("decode 只能是 greedy 或 beam")
            if not isinstance(constrained, bool):
                raise ValueError("constrained 必须是布尔值")
            if not 64 <= max_decode_len <= 512:
                raise ValueError("maxDecodeLen 必须在 64 到 512 之间")
        except (TypeError, ValueError) as error:
            self.send_json(400, {"error": str(error)})
            return

        if not RUN_LOCK.acquire(blocking=False):
            self.send_json(409, {"error": "已有一个识别任务正在运行，请等待它完成"})
            return
        try:
            result = self.run_inference(payload, decode, constrained, max_decode_len)
            self.send_json(200, result)
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
        except subprocess.TimeoutExpired:
            self.send_json(504, {"error": "识别超过 20 分钟，已停止"})
        except Exception as error:
            self.send_json(500, {"error": f"{type(error).__name__}: {error}"})
        finally:
            RUN_LOCK.release()

    def resolve_image(self, payload: dict, temp_dir: Path) -> Path:
        source_path = payload.get("sourcePath")
        if isinstance(source_path, str) and source_path:
            image_path = (ROOT / source_path).resolve()
            if not image_path.is_relative_to(ROOT) or not image_path.is_file():
                raise ValueError("内置图片路径无效")
            return image_path

        image_data = payload.get("imageData")
        if not isinstance(image_data, str) or not image_data.startswith("data:"):
            raise ValueError("请选择一张图片")
        header, separator, encoded = image_data.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("上传图片不是有效的 base64 数据")
        mime_type = header[5:].split(";", 1)[0]
        suffix = IMAGE_TYPES.get(mime_type)
        if suffix is None:
            raise ValueError("只支持 PNG、JPEG 或 WebP 图片")
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except binascii.Error as error:
            raise ValueError("上传图片数据损坏") from error
        if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("图片必须小于 18 MB")
        if detect_image_suffix(image_bytes) != suffix:
            raise ValueError("图片内容与文件类型不一致")
        image_path = temp_dir / f"upload{suffix}"
        image_path.write_bytes(image_bytes)
        return image_path

    def run_inference(
        self, payload: dict, decode: str, constrained: bool, max_decode_len: int
    ) -> dict:
        with tempfile.TemporaryDirectory(prefix="glife-omr-") as temp_name:
            temp_dir = Path(temp_name)
            image_path = self.resolve_image(payload, temp_dir)
            input_path = temp_dir / "input.json"
            output_path = temp_dir / "output.json"
            input_path.write_text(
                json.dumps({"clips": [{"id": "local-ui", "imagePath": str(image_path)}]}),
                encoding="utf-8",
            )
            command = [
                str(PYTHON),
                str(INFER_SCRIPT),
                "--input-json",
                str(input_path),
                "--output-json",
                str(output_path),
                "--model-dir",
                str(MODEL_DIR),
                "--device",
                "cpu",
                "--decode",
                decode,
                "--max-decode-len",
                str(max_decode_len),
            ]
            if decode == "beam":
                command.extend(["--beam-size", "3"])
            if constrained:
                command.append("--constrained")

            started = time.monotonic()
            completed = subprocess.run(
                command,
                cwd=INFER_SCRIPT.parent,
                capture_output=True,
                text=True,
                timeout=1200,
                check=False,
            )
            if completed.returncode != 0:
                message = completed.stderr.strip().splitlines()
                raise RuntimeError(message[-1] if message else "推理脚本运行失败")
            output = json.loads(output_path.read_text(encoding="utf-8"))
            predictions = output.get("predictions", [])
            if not predictions or not isinstance(predictions[0].get("tokenText"), str):
                raise RuntimeError("推理脚本没有返回 tokenText")
            prediction = predictions[0]
            record = self.record_store().record_inference(
                source_image=image_path,
                predicted_tokens=prediction["tokenText"],
                source_label=payload.get("sourceLabel"),
                decode=decode,
                constrained=constrained,
                max_decode_len=max_decode_len,
            )
            return {
                "recordId": record.record_id,
                "tokenText": prediction["tokenText"],
                "warnings": prediction.get("warnings", []),
                "metadata": output.get("metadata", {}),
                "elapsedSeconds": round(time.monotonic() - started, 1),
            }

    def review_record(self, payload: dict) -> dict:
        record_id = payload.get("recordId")
        label_source = payload.get("labelSource")
        corrected = payload.get("correctedTokenText")
        include_training = payload.get("includeTraining", False)
        categories = (
            normalize_error_categories(payload)
            if label_source == "corrected_error"
            else []
        )
        note = payload.get("note", "")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("当前没有可审核的识别记录")
        if not isinstance(corrected, str) or not corrected.strip():
            raise ValueError("请填写人工确认后的正确 tokens")
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError("备注不能超过 2000 个字符")
        if not isinstance(include_training, bool):
            raise ValueError("加入训练素材必须是布尔值")

        result = self.record_store().review(
            record_id=record_id,
            label_source=label_source,
            corrected_tokens=corrected,
            categories=categories,
            note=note,
            include_training=include_training,
            supersedes_id=payload.get("supersedesId"),
        )

        return {
            "recordId": result.record_id,
            "candidateId": result.candidate_id,
            "disposition": result.disposition,
            "supersededCandidateId": result.superseded_candidate_id,
            "trainingIncluded": result.disposition == "candidate",
            "path": str(result.path),
        }

    @staticmethod
    def record_store() -> OmrRecordStore:
        return OmrRecordStore(
            RECORDS_DIR,
            LabelSchema("guitar-tab-omr", 1, MODEL_DIR / "vocab.json"),
            PYTHON,
            MODEL_DIR / "best.pt",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Guitar Tab OMR comparison UI.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    for path in (PYTHON, INFER_SCRIPT, MODEL_DIR / "best.pt", MODEL_DIR / "vocab.json"):
        if not path.exists():
            raise FileNotFoundError(f"Required OMR file not found: {path}")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), OmrHandler)
    print(f"Local OMR UI: http://127.0.0.1:{args.port}/review/omr-workbench.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

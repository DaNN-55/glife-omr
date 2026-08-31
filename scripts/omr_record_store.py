#!/usr/bin/env python3
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


_WRITE_LOCK = threading.Lock()
_OMR_SCRIPTS = Path(__file__).resolve().parents[1] / "vendor/guitar-tab-omr/scripts"
_LABEL_SOURCES = {"corrected_error", "verified_correct"}


@dataclass(frozen=True)
class LabelSchema:
    id: str
    version: int
    vocab_path: Path

    def metadata(self) -> dict:
        return {
            "id": self.id,
            "version": self.version,
            "vocabSha256": hashlib.sha256(self.vocab_path.read_bytes()).hexdigest(),
        }


@dataclass(frozen=True)
class RecordResult:
    record_id: str
    path: Path


@dataclass(frozen=True)
class ReviewResult:
    record_id: str
    disposition: str
    candidate_id: str | None
    superseded_candidate_id: str | None
    path: Path


def token_manifest_stats(token_text: str) -> tuple[int, list[int]]:
    lines = [line.strip() for line in token_text.splitlines() if line.strip()]
    note_count = sum(
        1 for token in token_text.split() if re.fullmatch(r"N_S\d+_F(?:X|\d+)", token)
    )
    return note_count, [index for index, line in enumerate(lines) if line == "BAR"]


class OmrRecordStore:
    def __init__(
        self,
        root: Path,
        label_schema: LabelSchema,
        python: Path,
        checkpoint: Path,
    ) -> None:
        self.root = root
        self.label_schema = label_schema
        self.python = python
        self.checkpoint = checkpoint

    def record_inference(
        self,
        *,
        source_image: Path,
        predicted_tokens: str,
        source_label: str | None = None,
        decode: str | None = None,
        constrained: bool | None = None,
        max_decode_len: int | None = None,
    ) -> RecordResult:
        predicted = predicted_tokens.strip()
        if not predicted:
            raise ValueError("当前没有可记录的模型 tokens")
        if not source_image.is_file():
            raise ValueError("识别图片不存在")

        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".record-", dir=self.root) as stage_name:
            stage_dir = Path(stage_name) / "record"
            stage_dir.mkdir()
            original_suffix = (
                source_image.suffix.lower()
                if source_image.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                else ".img"
            )
            original_path = stage_dir / f"input{original_suffix}"
            shutil.copyfile(source_image, original_path)
            training_image = stage_dir / "image.png"
            conversion = subprocess.run(
                [
                    str(self.python),
                    "-c",
                    "from PIL import Image; import sys; Image.open(sys.argv[1]).convert('RGB').save(sys.argv[2])",
                    str(original_path),
                    str(training_image),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if conversion.returncode != 0:
                raise RuntimeError("无法把识别图片转换为训练用 PNG")

            image_sha256 = hashlib.sha256(training_image.read_bytes()).hexdigest()
            predicted_sha256 = self._tokens_sha256(predicted)
            with _WRITE_LOCK:
                records_dir = self.root / "records"
                records_dir.mkdir(exist_ok=True)
                record_id = self._next_record_id(image_sha256, predicted_sha256)
                metadata = {
                    "id": record_id,
                    "createdAt": self._now(),
                    "status": "unreviewed",
                    "source": source_label,
                    "decode": decode,
                    "constrained": constrained,
                    "maxDecodeLen": max_decode_len,
                    "checkpoint": str(self.checkpoint),
                    "imageSha256": image_sha256,
                    "predictedTokensSha256": predicted_sha256,
                }
                (stage_dir / "predicted.tokens.txt").write_text(
                    predicted + "\n", encoding="utf-8"
                )
                self._write_json_atomic(stage_dir / "record.json", metadata)
                record_dir = records_dir / record_id
                stage_dir.replace(record_dir)

        return RecordResult(record_id=record_id, path=record_dir)

    def review(
        self,
        *,
        record_id: str,
        label_source: str,
        corrected_tokens: str,
        categories: list[str],
        note: str = "",
        include_training: bool = False,
        supersedes_id: str | None = None,
    ) -> ReviewResult:
        if label_source not in _LABEL_SOURCES:
            raise ValueError("请选择有效的人工结论")
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError("备注不能超过 2000 个字符")
        if not isinstance(include_training, bool):
            raise ValueError("加入训练素材必须是布尔值")
        if label_source == "corrected_error" and not categories:
            raise ValueError("识别错误时请至少选择一种错误类型")
        if include_training and label_source == "corrected_error" and not any(
            category.startswith("omr_") for category in categories
        ):
            raise ValueError("只有 OMR 识别错误可以加入训练素材")

        corrected = corrected_tokens.strip()
        if not corrected:
            raise ValueError("请填写人工确认后的正确 tokens")
        schema = self.label_schema.metadata()

        # ponytail: one process-wide lock and a linear scan are enough until concurrent
        # writers or measured dataset size require a file lock or an index.
        with _WRITE_LOCK:
            record_dir, metadata = self._load_record(record_id)
            if metadata.get("status") != "unreviewed":
                raise ValueError("这条识别记录已经完成人工审核")
            predicted = (record_dir / "predicted.tokens.txt").read_text(encoding="utf-8").strip()
            same_tokens = predicted.split() == corrected.split()
            if label_source == "verified_correct" and not same_tokens:
                raise ValueError("确认识别正确时，人工 tokens 必须与模型输出一致")
            if label_source == "corrected_error" and same_tokens:
                raise ValueError("修正错误时必须修改至少一个 token")

            corrected_sha256 = self._tokens_sha256(corrected)
            image_sha256 = hashlib.sha256((record_dir / "image.png").read_bytes()).hexdigest()
            manifest = None
            active = []
            duplicate = None
            superseded = None
            if include_training:
                self._validate_training_tokens(corrected)
                manifest = self._load_manifest(schema)
                active = manifest["samples"]
                superseded = self._find_active(active, supersedes_id) if supersedes_id else None
                if superseded and self._entry_hashes(superseded)[0] != image_sha256:
                    raise ValueError("被取代候选与当前谱面样本不是同一张图片")
                duplicate = next(
                    (
                        entry
                        for entry in active
                        if self._entry_hashes(entry) == (image_sha256, corrected_sha256)
                    ),
                    None,
                )

            disposition = "duplicate" if duplicate else "candidate" if include_training else "reviewed"
            candidate_id = duplicate["id"] if duplicate else record_id if include_training else None
            reviewed = {
                **metadata,
                "status": "reviewed",
                "reviewedAt": self._now(),
                "labelSource": label_source,
                "category": categories[0] if categories else None,
                "categories": categories,
                "note": note.strip(),
                "trainingRequested": include_training,
                "disposition": disposition,
                "candidateId": candidate_id,
                "supersedesId": supersedes_id,
                "correctedTokensSha256": corrected_sha256,
                "labelSchema": schema,
            }

            (record_dir / "corrected.tokens.txt").write_text(corrected + "\n", encoding="utf-8")
            self._write_json_atomic(record_dir / "record.json", reviewed)

            if include_training and manifest is not None:
                self._ensure_vocab(schema)
                if superseded and (duplicate is None or superseded["id"] != duplicate["id"]):
                    active.remove(superseded)
                if duplicate is None:
                    note_count, bar_indexes = token_manifest_stats(corrected)
                    active.append(
                        {
                            "id": record_id,
                            "png": f"records/{record_id}/image.png",
                            "tokens": f"records/{record_id}/corrected.tokens.txt",
                            "labelSource": label_source,
                            "category": categories[0] if categories else None,
                            "categories": categories,
                            "noteCount": note_count,
                            "barIndexes": bar_indexes,
                        }
                    )
                manifest["schemaVersion"] = 3
                manifest["purpose"] = "Human-verified OMR training candidates"
                manifest["labelSchema"] = schema
                self._write_json_atomic(self.root / "manifest.json", manifest)

        return ReviewResult(
            record_id=record_id,
            disposition=disposition,
            candidate_id=candidate_id,
            superseded_candidate_id=superseded["id"] if superseded else None,
            path=record_dir,
        )

    def _validate_training_tokens(self, corrected: str) -> None:
        vocab = json.loads(self.label_schema.vocab_path.read_text(encoding="utf-8"))
        unknown_tokens = sorted({token for token in corrected.split() if token not in vocab})
        if unknown_tokens:
            raise ValueError("正确 tokens 包含当前词表之外的内容：" + ", ".join(unknown_tokens[:8]))
        validation = subprocess.run(
            [
                str(self.python),
                "-c",
                (
                    "import sys; sys.path.insert(0, sys.argv[1]); "
                    "from guitar_omr_train import validate_token_structure; "
                    "valid, _ = validate_token_structure(sys.stdin.read().split()); "
                    "raise SystemExit(0 if valid else 2)"
                ),
                str(_OMR_SCRIPTS),
            ],
            input=corrected,
            capture_output=True,
            text=True,
            check=False,
        )
        if validation.returncode == 2:
            raise ValueError("正确 tokens 不符合当前 Label Schema 的结构")
        if validation.returncode != 0:
            raise RuntimeError("无法校验正确 tokens 的结构")

    def _load_record(self, record_id: str) -> tuple[Path, dict]:
        if not isinstance(record_id, str) or not re.fullmatch(r"[A-Za-z0-9-]+", record_id):
            raise ValueError("识别记录 ID 无效")
        record_dir = (self.root / "records" / record_id).resolve()
        records_root = (self.root / "records").resolve()
        if not record_dir.is_relative_to(records_root) or not record_dir.is_dir():
            raise ValueError("识别记录不存在")
        try:
            metadata = json.loads((record_dir / "record.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("识别记录已损坏") from error
        return record_dir, metadata

    def _load_manifest(self, schema: dict) -> dict:
        path = self.root / "manifest.json"
        if not path.exists():
            return {
                "schemaVersion": 3,
                "purpose": "Human-verified OMR training candidates",
                "labelSchema": schema,
                "samples": [],
            }
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(manifest.get("samples"), list):
                raise ValueError
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError("训练候选 manifest 已损坏") from error

        existing_schema = manifest.get("labelSchema")
        if existing_schema is not None and existing_schema != schema:
            raise ValueError("训练候选数据集的 Label Schema 与当前模型不兼容")
        vocab_path = self.root / "vocab.json"
        if existing_schema is None and vocab_path.exists():
            if hashlib.sha256(vocab_path.read_bytes()).hexdigest() != schema["vocabSha256"]:
                raise ValueError("训练候选数据集的词表与当前模型不兼容")
        return manifest

    def _find_active(self, samples: list[dict], candidate_id: str) -> dict:
        matches = [entry for entry in samples if entry.get("id") == candidate_id]
        if not matches:
            raise ValueError("要取代的 Training Candidate 不存在或已被取代")
        return matches[0]

    def _entry_hashes(self, entry: dict) -> tuple[str, str]:
        try:
            image_path = self._dataset_path(entry["png"])
            token_path = self._dataset_path(entry["tokens"])
            image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
            corrected = token_path.read_text(encoding="utf-8")
        except (KeyError, OSError) as error:
            raise ValueError("训练候选 manifest 引用了缺失或无效的文件") from error
        return image_sha256, self._tokens_sha256(corrected)

    def _dataset_path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("训练候选 manifest 包含越界路径")
        return path

    def _next_record_id(self, image_sha256: str, predicted_sha256: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        digest = hashlib.sha256(f"{image_sha256}:{predicted_sha256}".encode()).hexdigest()[:8]
        base = f"record-{timestamp}-{digest}"
        record_id = base
        suffix = 1
        while (self.root / "records" / record_id).exists():
            suffix += 1
            record_id = f"{base}-{suffix}"
        return record_id

    def _ensure_vocab(self, schema: dict) -> None:
        path = self.root / "vocab.json"
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != schema["vocabSha256"]:
                raise ValueError("训练候选数据集的词表与当前模型不兼容")
            return
        temp_path = path.with_suffix(".json.tmp")
        shutil.copyfile(self.label_schema.vocab_path, temp_path)
        temp_path.replace(path)

    @staticmethod
    def _tokens_sha256(token_text: str) -> str:
        return hashlib.sha256(" ".join(token_text.split()).encode()).hexdigest()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _write_json_atomic(path: Path, value: dict) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)

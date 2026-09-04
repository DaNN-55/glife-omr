#!/usr/bin/env python3
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import threading
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


_WRITE_LOCK = threading.Lock()
_OMR_SCRIPTS = Path(__file__).resolve().parents[1] / "vendor/guitar-tab-omr/scripts"
_LABEL_SOURCES = {"corrected_error", "verified_correct"}


@lru_cache(maxsize=8)
def _file_sha256(path: str, modified_ns: int, size: int) -> str:
    del modified_ns, size
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
            self._convert_to_png(original_path, training_image)

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
                    "checkpointSha256": self._checkpoint_sha256(),
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

    def list_records(self) -> list[dict]:
        records_dir = self.root / "records"
        if not records_dir.is_dir():
            return []
        candidate_ids = self._candidate_ids()

        grouped: dict[tuple, list[dict]] = {}
        for record_dir in records_dir.iterdir():
            if not record_dir.is_dir():
                continue
            _, metadata = self._load_record(record_dir.name)
            key = (
                (
                    metadata.get("imageSha256"),
                    metadata.get("predictedTokensSha256"),
                    metadata.get("checkpointSha256") or metadata.get("checkpoint"),
                    metadata.get("decode"),
                    metadata.get("constrained"),
                    metadata.get("maxDecodeLen"),
                )
                if metadata.get("imageSha256") and metadata.get("predictedTokensSha256")
                else (metadata["id"],)
            )
            grouped.setdefault(key, []).append(metadata)

        history = []
        for records in grouped.values():
            representative = max(
                records,
                key=lambda item: (
                    item["id"] in candidate_ids,
                    item.get("status") == "reviewed",
                    item.get("createdAt", ""),
                ),
            )
            history.append(
                {
                    "recordId": representative["id"],
                    "source": representative.get("source"),
                    "createdAt": representative.get("createdAt", ""),
                    "updatedAt": representative.get("updatedAt"),
                    "status": representative.get("status", "unreviewed"),
                    "trainingCandidate": any(item["id"] in candidate_ids for item in records),
                    "decode": representative.get("decode"),
                    "constrained": representative.get("constrained"),
                    "maxDecodeLen": representative.get("maxDecodeLen"),
                    "duplicateCount": len(records),
                }
            )
        return sorted(history, key=lambda item: item["createdAt"], reverse=True)

    def load_record(self, record_id: str) -> dict:
        record_dir, metadata = self._load_record(record_id)
        inputs = sorted(record_dir.glob("input.*"))
        if not inputs:
            raise ValueError("识别记录缺少原始图片")
        corrected_path = record_dir / "corrected.tokens.txt"
        candidate_ids = self._candidate_ids()
        return {
            "recordId": record_id,
            "recordPath": record_dir,
            "inputFilename": inputs[0].name,
            "source": metadata.get("source"),
            "createdAt": metadata.get("createdAt"),
            "status": metadata.get("status", "unreviewed"),
            "reviewedAt": metadata.get("reviewedAt"),
            "updatedAt": metadata.get("updatedAt"),
            "trainingCandidate": record_id in candidate_ids,
            "decode": metadata.get("decode"),
            "constrained": metadata.get("constrained"),
            "maxDecodeLen": metadata.get("maxDecodeLen"),
            "predictedTokenText": (record_dir / "predicted.tokens.txt").read_text(encoding="utf-8").strip(),
            "correctedTokenText": corrected_path.read_text(encoding="utf-8").strip()
            if corrected_path.is_file()
            else None,
            "labelSource": metadata.get("labelSource"),
            "categories": metadata.get("categories", []),
            "note": metadata.get("note", ""),
        }

    def find_reusable_inference(
        self,
        *,
        source_image: Path,
        decode: str,
        constrained: bool,
        max_decode_len: int,
    ) -> dict | None:
        records_dir = self.root / "records"
        if not records_dir.is_dir():
            return None
        with tempfile.TemporaryDirectory(prefix="glife-image-hash-") as temp_name:
            canonical = Path(temp_name) / "image.png"
            self._convert_to_png(source_image, canonical)
            image_sha256 = hashlib.sha256(canonical.read_bytes()).hexdigest()
        checkpoint_sha256 = self._checkpoint_sha256()

        matches = []
        for record_dir in records_dir.iterdir():
            if not record_dir.is_dir():
                continue
            _, metadata = self._load_record(record_dir.name)
            if (
                metadata.get("imageSha256") == image_sha256
                and metadata.get("checkpointSha256") == checkpoint_sha256
                and metadata.get("decode") == decode
                and metadata.get("constrained") is constrained
                and metadata.get("maxDecodeLen") == max_decode_len
            ):
                matches.append(metadata)
        if not matches:
            return None
        selected = max(
            matches,
            key=lambda item: (item.get("status") == "reviewed", item.get("createdAt", "")),
        )
        return self.load_record(selected["id"])

    def delete_record(self, record_id: str) -> None:
        with _WRITE_LOCK:
            record_dir, _ = self._load_record(record_id)
            manifest_path = self.root / "manifest.json"
            manifest = self._read_manifest()
            manifest_changed = False
            if manifest is not None:
                samples = manifest["samples"]
                remaining = [sample for sample in samples if sample.get("id") != record_id]
                if len(remaining) != len(samples):
                    manifest["samples"] = remaining
                    manifest_changed = True

            staged_dir = self.root / f".deleted-{record_id}-{uuid4().hex}"
            record_dir.replace(staged_dir)
            try:
                if manifest_changed:
                    self._write_json_atomic(manifest_path, manifest)
            except Exception:
                staged_dir.replace(record_dir)
                raise

            try:
                shutil.rmtree(staged_dir)
            except OSError:
                # ponytail: logical deletion is complete; retry orphan cleanup only if this recurs.
                pass

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
        corrected = corrected_tokens.strip()
        if not corrected:
            raise ValueError("请填写人工确认后的正确 tokens")
        schema = self.label_schema.metadata()

        # ponytail: one process-wide lock and a linear scan are enough until concurrent
        # writers or measured dataset size require a file lock or an index.
        with _WRITE_LOCK:
            record_dir, metadata = self._load_record(record_id)
            if metadata.get("status", "unreviewed") not in {"unreviewed", "reviewed"}:
                raise ValueError("识别记录状态无效")
            was_reviewed = metadata.get("status") == "reviewed"
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
            existing = None
            superseded = None
            manifest_path = self.root / "manifest.json"
            if include_training:
                self._validate_training_tokens(corrected)
            if include_training or manifest_path.is_file():
                manifest = self._load_manifest(schema)
                active = manifest["samples"]
                existing = next((entry for entry in active if entry.get("id") == record_id), None)
            if include_training:
                superseded = self._find_active(active, supersedes_id) if supersedes_id else None
                if superseded and self._entry_hashes(superseded)[0] != image_sha256:
                    raise ValueError("被取代候选与当前谱面样本不是同一张图片")
                duplicate = next(
                    (
                        entry
                        for entry in active
                        if entry.get("id") != record_id
                        if self._entry_hashes(entry) == (image_sha256, corrected_sha256)
                    ),
                    None,
                )

            disposition = (
                "duplicate"
                if duplicate
                else "updated"
                if include_training and was_reviewed
                else "candidate"
                if include_training
                else "reviewed"
            )
            candidate_id = duplicate["id"] if duplicate else record_id if include_training else None
            updated_at = self._now()
            reviewed = {
                **metadata,
                "status": "reviewed",
                "reviewedAt": metadata.get("reviewedAt") or updated_at,
                "updatedAt": updated_at,
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

            if manifest is not None:
                if existing:
                    active.remove(existing)
                if include_training:
                    if (
                        superseded
                        and superseded in active
                        and (duplicate is None or superseded["id"] != duplicate["id"])
                    ):
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

            corrected_path = record_dir / "corrected.tokens.txt"
            metadata_path = record_dir / "record.json"
            snapshots = [
                (corrected_path, corrected_path.read_bytes() if corrected_path.exists() else None),
                (metadata_path, metadata_path.read_bytes()),
            ]
            if manifest is not None:
                snapshots.append(
                    (manifest_path, manifest_path.read_bytes() if manifest_path.exists() else None)
                )
                if include_training:
                    vocab_path = self.root / "vocab.json"
                    snapshots.append(
                        (vocab_path, vocab_path.read_bytes() if vocab_path.exists() else None)
                    )
            try:
                if include_training:
                    self._ensure_vocab(schema)
                self._write_bytes_atomic(corrected_path, (corrected + "\n").encode())
                self._write_json_atomic(metadata_path, reviewed)
                if manifest is not None:
                    self._write_json_atomic(manifest_path, manifest)
            except Exception:
                rollback_error = None
                for path, previous in reversed(snapshots):
                    try:
                        if previous is None:
                            path.unlink(missing_ok=True)
                        elif not path.exists() or path.read_bytes() != previous:
                            self._write_bytes_atomic(path, previous)
                    except OSError as error:
                        rollback_error = error
                if rollback_error:
                    raise RuntimeError("保存失败，且无法完整恢复原记录") from rollback_error
                raise

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

    def _convert_to_png(self, source: Path, destination: Path) -> None:
        conversion = subprocess.run(
            [
                str(self.python),
                "-c",
                "from PIL import Image; import sys; Image.open(sys.argv[1]).convert('RGB').save(sys.argv[2])",
                str(source),
                str(destination),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if conversion.returncode != 0:
            raise RuntimeError("无法把识别图片转换为训练用 PNG")

    def _checkpoint_sha256(self) -> str:
        stat = self.checkpoint.stat()
        return _file_sha256(str(self.checkpoint.resolve()), stat.st_mtime_ns, stat.st_size)

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
        manifest = self._read_manifest()
        if manifest is None:
            return {
                "schemaVersion": 3,
                "purpose": "Human-verified OMR training candidates",
                "labelSchema": schema,
                "samples": [],
            }
        existing_schema = manifest.get("labelSchema")
        if existing_schema is not None and existing_schema != schema:
            raise ValueError("训练候选数据集的 Label Schema 与当前模型不兼容")
        vocab_path = self.root / "vocab.json"
        if existing_schema is None and vocab_path.exists():
            if hashlib.sha256(vocab_path.read_bytes()).hexdigest() != schema["vocabSha256"]:
                raise ValueError("训练候选数据集的词表与当前模型不兼容")
        return manifest

    def _candidate_ids(self) -> set[str]:
        manifest = self._read_manifest()
        if manifest is None:
            return set()
        return {
            sample["id"]
            for sample in manifest["samples"]
            if isinstance(sample.get("id"), str)
        }

    def _read_manifest(self) -> dict | None:
        path = self.root / "manifest.json"
        if not path.exists():
            return None
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            samples = manifest.get("samples")
            if not isinstance(samples, list) or not all(isinstance(sample, dict) for sample in samples):
                raise ValueError
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError("训练候选 manifest 已损坏") from error
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
        OmrRecordStore._write_bytes_atomic(
            path,
            (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode(),
        )

    @staticmethod
    def _write_bytes_atomic(path: Path, value: bytes) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            temp_path.write_bytes(value)
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)

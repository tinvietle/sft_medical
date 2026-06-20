from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset


class SmolJsonDatasetParser:
    """Parse local JSON case files into the trainer's messages format."""

    REQUIRED_KEYS = ("full_prompt", "reasoning", "answer")

    def __init__(
        self,
        dataset_dir: str | Path,
        system_prompt: str,
        *,
        skip_empty_targets: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.system_prompt = system_prompt
        self.skip_empty_targets = skip_empty_targets
        self.skipped_files: list[str] = []

    def load(self) -> Dataset:
        json_files = sorted(self.dataset_dir.rglob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {self.dataset_dir}")

        self.skipped_files = []
        records = []
        for path in json_files:
            record = self._parse_file(path)
            if record is None:
                self.skipped_files.append(str(path.relative_to(self.dataset_dir)))
                continue
            records.append(record)

        if not records:
            raise ValueError(f"No usable JSON samples found in {self.dataset_dir}")
        return Dataset.from_list(records)

    def _parse_file(self, path: Path) -> dict[str, Any] | None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object in {path}, found {type(payload).__name__}")

        missing_keys = [key for key in self.REQUIRED_KEYS if key not in payload]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise ValueError(f"Missing required key(s) in {path}: {missing}")

        full_prompt = self._require_non_empty_string(payload["full_prompt"], path, "full_prompt")
        reasoning = self._require_string(payload["reasoning"], path, "reasoning")
        answer = self._require_string(payload["answer"], path, "answer")

        if self.skip_empty_targets and not reasoning.strip() and not answer.strip():
            return None

        return {
            "source_file": str(path.relative_to(self.dataset_dir)),
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                    "thinking": None,
                },
                {
                    "role": "user",
                    "content": full_prompt,
                    "thinking": None,
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{reasoning}\n</think>\n\n{answer}",
                    "thinking": None,
                },
            ],
        }

    @staticmethod
    def _require_string(value: Any, path: Path, key: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"Expected '{key}' in {path} to be a string.")
        return value

    @staticmethod
    def _require_non_empty_string(value: Any, path: Path, key: str) -> str:
        value = SmolJsonDatasetParser._require_string(value, path, key)
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"Expected '{key}' in {path} to be non-empty.")
        return normalized

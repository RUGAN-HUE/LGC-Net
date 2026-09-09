"""Pixel-free BookLayout-Bi loader using explicit public split files."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .model import LABEL_TO_INDEX


def read_split_ids(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_annotation_records(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    records = payload["records"]
    ids = [record["sample_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Annotation file contains duplicate sample IDs.")
    return records


class BookLayoutDataset(Dataset):
    def __init__(
        self,
        annotation_file: str | Path,
        split_file: str | Path,
        max_seq_len: int = 15,
    ) -> None:
        all_records = load_annotation_records(annotation_file)
        by_id = {record["sample_id"]: record for record in all_records}
        split_ids = read_split_ids(split_file)
        missing = [sample_id for sample_id in split_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"Split contains unknown sample IDs: {missing[:5]}")
        if len(split_ids) != len(set(split_ids)):
            raise ValueError(f"Split contains duplicate sample IDs: {split_file}")
        self.records = [by_id[sample_id] for sample_id in split_ids]
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        elements = record["elements"][: self.max_seq_len]
        sequence_length = len(elements)

        labels = torch.zeros(self.max_seq_len, dtype=torch.long)
        boxes = torch.zeros(self.max_seq_len, 4, dtype=torch.float32)
        mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        for element_index, element in enumerate(elements):
            labels[element_index] = LABEL_TO_INDEX[element["role"]]
            boxes[element_index] = torch.tensor(
                element["bbox_xyxy_normalized"], dtype=torch.float32
            )
            mask[element_index] = False

        return {
            "sample_id": record["sample_id"],
            "labels": labels,
            "boxes": boxes,
            "mask": mask,
            "subject_id": torch.tensor(record["subject_id"], dtype=torch.long),
            "language_id": torch.tensor(record["language_id"], dtype=torch.long),
            "annotation_type": record["annotation_type"],
            "seq_len": sequence_length,
        }


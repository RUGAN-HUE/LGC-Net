"""Export the fixed BookLayout-Bi splits to the four-class COCO-style format.

The exporter consumes only the public structured annotations. It never reads or
copies the original book-cover images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROLE_TO_COARSE = {
    "Title": "Title",
    "Subtitle": "Text",
    "Author": "Text",
    "Publisher": "Text",
    "Endorsement": "Text",
    "Series": "Text",
    "Main_Image": "Figure",
    "Logo": "List",
    "Badge": "List",
}

CATEGORIES = [
    {"id": 1, "name": "Title", "supercategory": "text"},
    {"id": 2, "name": "Text", "supercategory": "text"},
    {"id": 3, "name": "Figure", "supercategory": "image"},
    {"id": 4, "name": "List", "supercategory": "text"},
]
CATEGORY_ID = {item["name"]: item["id"] for item in CATEGORIES}

SPLITS = (
    "stage1_train",
    "stage1_validation",
    "stage2_train",
    "stage2_validation",
    "test",
)


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def export_records(records: dict[str, dict], sample_ids: list[str]) -> dict:
    images: list[dict] = []
    annotations: list[dict] = []
    annotation_id = 1
    for image_id, sample_id in enumerate(sample_ids, start=1):
        record = records[sample_id]
        width = int(record["image_width"])
        height = int(record["image_height"])
        images.append(
            {
                "id": image_id,
                "file_name": f"{sample_id}.png",
                "sample_id": sample_id,
                "width": width,
                "height": height,
            }
        )
        for element in record["elements"]:
            role = element["role"]
            coarse = ROLE_TO_COARSE[role]
            x1, y1, x2, y2 = map(float, element["bbox_xyxy_normalized"])
            x = x1 * width
            y = y1 * height
            box_width = (x2 - x1) * width
            box_height = (y2 - y1) * height
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": CATEGORY_ID[coarse],
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "fine_role": role,
                }
            )
            annotation_id += 1
    return {"images": images, "annotations": annotations, "categories": CATEGORIES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    records = {record["sample_id"]: record for record in payload["records"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_ids: dict[str, list[str]] = {}
    for split in SPLITS:
        ids = read_ids(args.split_root / f"{split}.txt")
        split_ids[split] = ids
        output = export_records(records, ids)
        (args.output_dir / f"{split}.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{split}: {len(output['images'])} layouts, {len(output['annotations'])} boxes")

    non_test_ids = split_ids["stage1_train"] + split_ids["stage1_validation"]
    if len(non_test_ids) != len(set(non_test_ids)):
        raise ValueError("The Stage 1 training and validation identifiers overlap")
    non_test = export_records(records, non_test_ids)
    (args.output_dir / "non_test.json").write_text(
        json.dumps(non_test, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"non_test: {len(non_test['images'])} layouts, {len(non_test['annotations'])} boxes")


if __name__ == "__main__":
    main()

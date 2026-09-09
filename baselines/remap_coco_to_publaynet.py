"""Remap the shared four-class COCO files to PubLayNet category identifiers.

LayoutGAN++ and LayoutDM use the PubLayNet category vocabulary in their
released data loaders. The unused ``table`` class remains in the category list
for compatibility with those upstream implementations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


NAME_MAP = {"Title": "title", "Text": "text", "Figure": "figure", "List": "list"}
PUBLAYNET_CATEGORIES = [
    {"id": 1, "name": "text", "supercategory": ""},
    {"id": 2, "name": "title", "supercategory": ""},
    {"id": 3, "name": "list", "supercategory": ""},
    {"id": 4, "name": "table", "supercategory": ""},
    {"id": 5, "name": "figure", "supercategory": ""},
]
PUBLAYNET_ID = {item["name"]: item["id"] for item in PUBLAYNET_CATEGORIES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    source_names = {int(item["id"]): item["name"] for item in payload["categories"]}
    annotations = []
    for source in payload["annotations"]:
        annotation = dict(source)
        source_name = source_names[int(annotation["category_id"])]
        annotation["category_id"] = PUBLAYNET_ID[NAME_MAP[source_name]]
        annotations.append(annotation)

    output = {
        "images": payload["images"],
        "annotations": annotations,
        "categories": PUBLAYNET_CATEGORIES,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{args.output_json}: {len(output['images'])} layouts, {len(annotations)} boxes")


if __name__ == "__main__":
    main()

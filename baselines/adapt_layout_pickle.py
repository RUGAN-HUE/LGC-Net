"""Convert trusted LayoutGAN++ or LayoutDM pickle output to the common schema.

Only open pickle files produced by a trusted local experiment. Python pickle is
not a safe interchange format for untrusted files.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


PUBLAYNET_CONTIGUOUS_TO_COARSE = {
    0: 1,  # text -> Text
    1: 0,  # title -> Title
    2: 3,  # list -> List
    4: 2,  # figure -> Figure
}


def as_list(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value.tolist() if hasattr(value, "tolist") else list(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--payload-key", default="results")
    parser.add_argument(
        "--filter-landscape",
        action="store_true",
        help="Apply the H < W exclusion used by the upstream PubLayNet loaders.",
    )
    args = parser.parse_args()

    dataset = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    records = {record["sample_id"]: record for record in dataset["records"]}
    sample_ids = [
        line.strip()
        for line in args.test_split.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.filter_landscape:
        sample_ids = [
            sample_id
            for sample_id in sample_ids
            if records[sample_id]["image_height"] >= records[sample_id]["image_width"]
        ]

    with args.pickle.open("rb") as handle:
        payload = pickle.load(handle)
    results = payload.get(args.payload_key) if isinstance(payload, dict) else payload
    if results is None:
        raise KeyError(f"Pickle payload does not contain {args.payload_key!r}")
    if len(results) != len(sample_ids):
        raise ValueError(
            f"Prediction count {len(results)} does not match expected layout count {len(sample_ids)}"
        )

    output_records = []
    for sample_id, item in zip(sample_ids, results):
        centers, labels = item
        boxes = []
        for x_center, y_center, width, height in as_list(centers):
            boxes.append(
                [
                    float(x_center - width / 2),
                    float(y_center - height / 2),
                    float(x_center + width / 2),
                    float(y_center + height / 2),
                ]
            )
        coarse_labels = [PUBLAYNET_CONTIGUOUS_TO_COARSE[int(value)] for value in as_list(labels)]
        if len(boxes) != len(coarse_labels):
            raise ValueError(f"Box/label length mismatch for {sample_id}")
        output_records.append(
            {"sample_id": sample_id, "boxes_xyxy_normalized": boxes, "coarse_labels": coarse_labels}
        )

    output = {
        "meta": {
            "method": args.method,
            "coordinate_format": "normalized_xyxy",
            "coarse_label_ids": {"Title": 0, "Text": 1, "Figure": 2, "List": 3},
            "landscape_filter_applied": bool(args.filter_landscape),
        },
        "records": output_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"{args.output}: {len(output_records)} predictions")


if __name__ == "__main__":
    main()

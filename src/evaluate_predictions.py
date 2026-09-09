"""Evaluate standardized predictions against the fixed BookLayout-Bi test set."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .evaluation import canonicalize_boxes
from .metrics import elementwise_iou, layout_metrics, maximum_iou


ROLE_TO_COARSE = {
    "Title": 0,
    "Subtitle": 1,
    "Author": 1,
    "Publisher": 1,
    "Endorsement": 1,
    "Series": 1,
    "Main_Image": 2,
    "Logo": 3,
    "Badge": 3,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    dataset_payload = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    dataset = {record["sample_id"]: record for record in dataset_payload["records"]}
    test_ids = [
        line.strip()
        for line in args.test_split.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prediction_payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions = {record["sample_id"]: record for record in prediction_payload["records"]}

    unexpected = sorted(set(predictions).difference(test_ids))
    missing = sorted(set(test_ids).difference(predictions))
    if unexpected:
        raise ValueError(f"Predictions contain {len(unexpected)} identifiers outside the test split")
    if args.require_all and missing:
        raise ValueError(f"Predictions are missing {len(missing)} test identifiers")

    accumulator: dict[str, list[float]] = defaultdict(list)
    truth_layouts = []
    generated_layouts = []
    skipped_short = 0
    for sample_id in test_ids:
        if sample_id not in predictions:
            continue
        truth_record = dataset[sample_id]
        prediction = predictions[sample_id]
        predicted_boxes = [list(map(float, box)) for box in prediction["boxes_xyxy_normalized"]]
        predicted_labels = np.asarray(prediction["coarse_labels"], dtype=int)
        target_boxes = [
            list(map(float, element["bbox_xyxy_normalized"]))
            for element in truth_record["elements"]
        ]
        target_labels = np.asarray(
            [ROLE_TO_COARSE[element["role"]] for element in truth_record["elements"]], dtype=int
        )
        if len(predicted_boxes) != len(predicted_labels):
            raise ValueError(f"Box/label length mismatch for {sample_id}")
        if len(predicted_boxes) < 2:
            skipped_short += 1
            continue

        overlap, max_overlap, alignment, violation = layout_metrics(predicted_boxes)
        accumulator["overlap"].append(overlap)
        accumulator["max_overlap"].append(max_overlap)
        accumulator["alignment"].append(alignment)
        accumulator["violation"].append(violation)
        accumulator["iou"].append(elementwise_iou(predicted_boxes, target_boxes))
        truth_layouts.append((canonicalize_boxes(target_boxes), target_labels))
        generated_layouts.append((canonicalize_boxes(predicted_boxes), predicted_labels))

    output = {
        "method": prediction_payload.get("meta", {}).get("method", "unspecified"),
        "test_records_total": len(test_ids),
        "prediction_records": len(predictions),
        "missing_test_records": len(missing),
        "evaluated_records": len(accumulator["iou"]),
        "skipped_predictions_with_fewer_than_two_boxes": skipped_short,
        "metrics": {key: float(np.mean(values)) for key, values in accumulator.items()},
    }
    output["metrics"]["max_iou"] = maximum_iou(truth_layouts, generated_layouts)
    rendered = json.dumps(output, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

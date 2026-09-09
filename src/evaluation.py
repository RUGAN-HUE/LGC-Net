"""Evaluation routines shared by training and command-line evaluation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .metrics import elementwise_iou, layout_metrics, maximum_iou


# Shared four-class space used by the reported Max IoU comparison:
# Title, Text, Figure, and List.
FINE_TO_COARSE = np.asarray([0, 1, 1, 1, 3, 2, 1, 3, 1], dtype=int)


def canonicalize_boxes(boxes: list[list[float]]) -> np.ndarray:
    values = np.asarray(boxes, dtype=float)
    x1 = np.minimum(values[:, 0], values[:, 2])
    y1 = np.minimum(values[:, 1], values[:, 3])
    x2 = np.maximum(values[:, 0], values[:, 2])
    y2 = np.maximum(values[:, 1], values[:, 3])
    return np.stack([x1, y1, x2, y2], axis=1)


@torch.no_grad()
def evaluate_model(model, dataset, device: torch.device, evaluation_seed: int = 42) -> dict:
    model.eval()
    torch.manual_seed(evaluation_seed)
    accumulator: dict[str, list[float]] = defaultdict(list)
    ground_truth_layouts = []
    generated_layouts = []

    for sample in dataset:
        sequence_length = int(sample["seq_len"])
        predicted = model.forward_generate(
            sample["labels"].unsqueeze(0).to(device),
            sample["subject_id"].unsqueeze(0).to(device),
            sample["language_id"].unsqueeze(0).to(device),
            sample["mask"].unsqueeze(0).to(device),
        )[0].cpu().numpy()
        predicted_boxes = [list(map(float, predicted[index])) for index in range(sequence_length)]
        target_boxes = [
            list(map(float, sample["boxes"][index])) for index in range(sequence_length)
        ]
        if len(predicted_boxes) < 2:
            continue

        overlap, max_overlap, alignment, violation = layout_metrics(predicted_boxes)
        accumulator["overlap"].append(overlap)
        accumulator["max_overlap"].append(max_overlap)
        accumulator["alignment"].append(alignment)
        accumulator["violation"].append(violation)
        accumulator["iou"].append(elementwise_iou(predicted_boxes, target_boxes))
        fine_labels = sample["labels"][:sequence_length].cpu().numpy()
        coarse_labels = FINE_TO_COARSE[fine_labels]
        ground_truth_layouts.append((canonicalize_boxes(target_boxes), coarse_labels))
        generated_layouts.append((canonicalize_boxes(predicted_boxes), coarse_labels))

    output = {
        name: float(np.mean(values))
        for name, values in accumulator.items()
    }
    output["max_iou"] = maximum_iou(ground_truth_layouts, generated_layouts)
    return output

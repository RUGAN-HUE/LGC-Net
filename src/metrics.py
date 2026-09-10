"""Unified layout metrics used by the final LGC-Net evaluation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def intersection_over_union(box1: list[float], box2: list[float]) -> float:
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return float(intersection / (area1 + area2 - intersection + 1e-6))


def layout_metrics(boxes: list[list[float]]) -> tuple[float, float, float, float]:
    if len(boxes) < 2:
        return 0.0, 0.0, 0.0, 0.0

    overlaps = [
        intersection_over_union(boxes[i], boxes[j])
        for i in range(len(boxes))
        for j in range(i + 1, len(boxes))
    ]
    alignment_errors = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            horizontal = min(
                abs(boxes[i][0] - boxes[j][0]),
                abs(boxes[i][2] - boxes[j][2]),
                abs((boxes[i][0] + boxes[i][2]) / 2 - (boxes[j][0] + boxes[j][2]) / 2),
            )
            vertical = min(
                abs(boxes[i][1] - boxes[j][1]),
                abs(boxes[i][3] - boxes[j][3]),
                abs((boxes[i][1] + boxes[i][3]) / 2 - (boxes[j][1] + boxes[j][3]) / 2),
            )
            alignment_errors.append(min(horizontal, vertical))

    violation = sum(
        1 for box in boxes
        if box[0] < 0 or box[1] < 0 or box[2] > 1 or box[3] > 1
    ) / float(len(boxes))
    return (
        float(np.mean(overlaps)),
        float(np.max(overlaps)),
        float(np.mean(alignment_errors)),
        float(violation),
    )


def elementwise_iou(predicted: list[list[float]], target: list[list[float]]) -> float:
    if not predicted or not target:
        return 0.0
    count = min(len(predicted), len(target))
    return float(
        np.mean([intersection_over_union(predicted[i], target[i]) for i in range(count)])
    )


def _layout_matching_score(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
) -> float:
    first_boxes, first_labels = first
    second_boxes, second_labels = second
    if len(first_boxes) == 0 or len(first_boxes) != len(second_boxes):
        return 0.0
    score = 0.0
    for label in np.unique(first_labels):
        first_group = first_boxes[first_labels == label]
        second_group = second_boxes[second_labels == label]
        if len(first_group) != len(second_group):
            return 0.0
        matrix = np.asarray(
            [
                [intersection_over_union(a.tolist(), b.tolist()) for b in second_group]
                for a in first_group
            ],
            dtype=float,
        )
        rows, columns = linear_sum_assignment(matrix, maximize=True)
        score += float(matrix[rows, columns].sum())
    return score / len(first_boxes)


def maximum_iou(
    ground_truth_layouts: list[tuple[np.ndarray, np.ndarray]],
    generated_layouts: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Best-match consistency using category and layout-level Hungarian matching.

    Layouts are grouped by their label multisets. Boxes are first matched within
    each category; layouts are then matched within each label-multiset group.
    This is the dataset-level Max IoU reported in the manuscript.
    """

    def group(layouts):
        grouped: dict[tuple[int, ...], list[tuple[np.ndarray, np.ndarray]]] = {}
        for boxes, labels in layouts:
            key = tuple(sorted(int(value) for value in labels.tolist()))
            grouped.setdefault(key, []).append((boxes, labels))
        return grouped

    truth_groups = group(ground_truth_layouts)
    generated_groups = group(generated_layouts)
    matched_scores = []
    for key in sorted(set(truth_groups).intersection(generated_groups)):
        truth = truth_groups[key]
        generated = generated_groups[key]
        matrix = np.asarray(
            [
                [_layout_matching_score(first, second) for second in generated]
                for first in truth
            ],
            dtype=float,
        )
        rows, columns = linear_sum_assignment(matrix, maximize=True)
        matched_scores.extend(matrix[rows, columns].tolist())
    return float(np.mean(matched_scores)) if matched_scores else 0.0


# Backward-compatible function names used by the original experiment scripts.
calculate_iou_or_overlap = intersection_over_union
evaluate_comprehensive_metrics = layout_metrics
evaluate_real_iou = elementwise_iou

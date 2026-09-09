"""Run prior-sampled LayoutVAE inference in the common prediction schema.

The adapter imports ``layout_vae/box.py`` from a separately cloned DeepLayout
checkout. Ground-truth coordinates are never passed to the encoder or decoder.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


COARSE_LABEL_ID = {"Title": 0, "Text": 1, "Figure": 2, "List": 3}


def load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    category_names = {int(item["id"]): item["name"] for item in payload["categories"]}
    annotations: dict[int, list[int]] = {}
    for annotation in payload["annotations"]:
        annotations.setdefault(int(annotation["image_id"]), []).append(
            int(annotation["category_id"])
        )
    return [
        {
            "sample_id": image.get("sample_id") or Path(image["file_name"]).stem,
            "labels": annotations.get(int(image["id"]), []),
            "category_names": category_names,
        }
        for image in payload["images"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-coco", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--representation-size", type=int, default=32)
    parser.add_argument("--conditioning-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    module_root = args.upstream_root / "layout_vae"
    sys.path.insert(0, str(module_root))
    upstream_box = importlib.import_module("box")

    class BoxVAE(torch.nn.Module):
        def __init__(self, number_labels: int):
            super().__init__()
            self.encoder = upstream_box.AutoregressiveBoxEncoder(
                number_labels, args.conditioning_size, args.representation_size
            )
            self.decoder = upstream_box.AutoregressiveBoxDecoder(
                args.conditioning_size, args.representation_size
            )

        @torch.no_grad()
        def generate_step(self, label_set, current_label, previous_labels, previous_boxes, state):
            condition, state = self.encoder.conditioning(
                label_set, current_label, previous_labels, previous_boxes, state=state
            )
            latent = torch.randn(
                (label_set.size(0), args.representation_size), device=label_set.device
            )
            return self.decoder(latent, condition), state

    cases = load_cases(args.test_coco)
    number_labels = max(max(case["category_names"]) for case in cases)
    model = BoxVAE(number_labels).to(device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    state_dict = OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in checkpoint.items()
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    label_encoding = torch.eye(number_labels, device=device)
    output_records = []
    for case in cases:
        labels = case["labels"]
        label_counts = np.zeros(number_labels, dtype=np.float32)
        for label in labels:
            label_counts[label - 1] += 1
        label_set = torch.tensor(label_counts, device=device).unsqueeze(0)
        predicted = torch.zeros((len(labels), 4), device=device)
        hidden = cell = None
        for step, label in enumerate(labels):
            current = label_encoding[label - 1].unsqueeze(0)
            if step == 0:
                previous_labels = torch.zeros((1, 0, number_labels), device=device)
                previous_boxes = torch.zeros((1, 0, 4), device=device)
                state = None
            else:
                previous_labels = label_encoding[labels[step - 1] - 1].view(1, 1, -1)
                previous_boxes = predicted[step - 1].view(1, 1, 4)
                state = (
                    (hidden.unsqueeze(0), cell.unsqueeze(0))
                    if hidden is not None and cell is not None
                    else None
                )
            box, state = model.generate_step(
                label_set, current, previous_labels, previous_boxes, state
            )
            predicted[step] = box[0]
            if state is not None:
                hidden, cell = state[0][-1], state[1][-1]

        boxes = []
        for x, y, width, height in predicted.detach().cpu().tolist():
            boxes.append([x, y, x + width, y + height])
        names = [case["category_names"][label] for label in labels]
        output_records.append(
            {
                "sample_id": case["sample_id"],
                "boxes_xyxy_normalized": boxes,
                "coarse_labels": [COARSE_LABEL_ID[name] for name in names],
            }
        )

    output = {
        "meta": {
            "method": "LayoutVAE",
            "upstream": "https://github.com/kampta/DeepLayout",
            "coordinate_format": "normalized_xyxy",
            "latent_sampling": "standard Gaussian prior",
            "seed": args.seed,
        },
        "records": output_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"{args.output}: {len(output_records)} predictions")


if __name__ == "__main__":
    main()

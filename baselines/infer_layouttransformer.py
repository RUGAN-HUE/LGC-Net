"""Run conditional LayoutTransformer inference in the common prediction schema.

This adapter imports the official DeepLayout implementation from a separately
cloned upstream checkout. It does not redistribute upstream source code.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


PRECISION = 8
SIZE = 2**PRECISION
COARSE_LABEL_ID = {"Title": 0, "Text": 1, "Figure": 2, "List": 3}


def load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    category_names = {int(item["id"]): item["name"] for item in payload["categories"]}
    annotations: dict[int, list[str]] = {}
    for annotation in payload["annotations"]:
        annotations.setdefault(int(annotation["image_id"]), []).append(
            category_names[int(annotation["category_id"])]
        )
    return [
        {
            "sample_id": image.get("sample_id") or Path(image["file_name"]).stem,
            "labels": annotations.get(int(image["id"]), []),
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
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    module_root = args.upstream_root / "layout_transformer"
    sys.path.insert(0, str(module_root))
    upstream_model = importlib.import_module("model")

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    checkpoint = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }
    vocab_size = int(checkpoint["tok_emb.weight"].shape[0])
    block_size = int(checkpoint["pos_emb"].shape[1])
    config = upstream_model.GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = upstream_model.GPT(config).to(device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    bos = vocab_size - 3
    label_token = {
        "Title": SIZE,
        "Text": SIZE + 1,
        "Figure": SIZE + 2,
        "List": SIZE + 3,
    }
    output_records = []
    with torch.no_grad():
        for case in load_cases(args.test_coco):
            sequence = [bos]
            boxes = []
            for label in case["labels"]:
                sequence.append(label_token[label])
                coordinates = []
                for _ in range(4):
                    tokens = torch.tensor(
                        sequence[-block_size:], dtype=torch.long, device=device
                    ).unsqueeze(0)
                    logits, _ = model(tokens)
                    logits = logits[0, -1] / args.temperature
                    masked = torch.full_like(logits, float("-inf"))
                    masked[:SIZE] = logits[:SIZE]
                    logits = masked
                    if args.sample:
                        token = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
                    else:
                        token = int(torch.argmax(logits).item())
                    sequence.append(token)
                    coordinates.append(token)
                x, y, width, height = [value / float(SIZE - 1) for value in coordinates]
                boxes.append([x, y, x + width, y + height])
            output_records.append(
                {
                    "sample_id": case["sample_id"],
                    "boxes_xyxy_normalized": boxes,
                    "coarse_labels": [COARSE_LABEL_ID[label] for label in case["labels"]],
                }
            )

    output = {
        "meta": {
            "method": "LayoutTransformer",
            "upstream": "https://github.com/kampta/DeepLayout",
            "coordinate_format": "normalized_xyxy",
            "inference": "sampling" if args.sample else "greedy",
            "seed": args.seed,
        },
        "records": output_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"{args.output}: {len(output_records)} predictions")


if __name__ == "__main__":
    main()

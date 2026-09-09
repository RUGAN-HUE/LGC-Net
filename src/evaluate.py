"""Evaluate one trained LGC-Net checkpoint on the fixed held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import BookLayoutDataset
from .evaluation import evaluate_model
from .model import LGCNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--evaluation-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_keys = {
        "d_model",
        "nhead",
        "latent_dim",
        "num_encoder_layers",
        "num_decoder_layers",
        "max_seq_len",
    }
    model_config = {
        key: value for key, value in checkpoint["config"].items() if key in model_keys
    }
    model = LGCNet(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = BookLayoutDataset(
        args.dataset_json, args.test_split, model.max_seq_len
    )
    metrics = evaluate_model(model, dataset, device, args.evaluation_seed)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

"""Reproduce the final M-32-G five-seed LGC-Net training protocol.

Run from the repository root with::

    python -m src.train --dataset-json PATH --split-root PATH

The script consumes only the pixel-free BookLayout-Bi annotation JSON and
explicit split files. It has no dependency on the unpublished negative-space
project.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .data import BookLayoutDataset
from .evaluation import evaluate_model
from .model import LGCNet


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def box_reconstruction_loss(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    valid = (~mask).float().unsqueeze(-1)
    absolute = ((predicted - target).abs() * valid).sum()
    squared = ((predicted - target).pow(2) * valid).sum()
    return (absolute + squared) / (valid.sum() + 1e-6)


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float) -> torch.Tensor:
    per_dimension = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return torch.clamp(per_dimension, min=free_bits).sum(dim=-1).mean()


def overlap_loss(predicted: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    _, sequence_length, _ = predicted.shape
    valid = (~mask).float()
    left = predicted.unsqueeze(2)
    right = predicted.unsqueeze(1)
    x1 = torch.maximum(left[..., 0], right[..., 0])
    y1 = torch.maximum(left[..., 1], right[..., 1])
    x2 = torch.minimum(left[..., 2], right[..., 2])
    y2 = torch.minimum(left[..., 3], right[..., 3])
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_left = (
        (left[..., 2] - left[..., 0]) * (left[..., 3] - left[..., 1])
    ).clamp(min=1e-6)
    area_right = (
        (right[..., 2] - right[..., 0]) * (right[..., 3] - right[..., 1])
    ).clamp(min=1e-6)
    iou = intersection / (area_left + area_right - intersection + 1e-6)
    valid_pairs = valid.unsqueeze(2) * valid.unsqueeze(1)
    diagonal = torch.eye(sequence_length, device=predicted.device).unsqueeze(0).bool()
    valid_pairs = valid_pairs * (~diagonal).float()
    return (iou * valid_pairs).sum() / (valid_pairs.sum() + 1e-6)


def structural_prior_loss(
    predicted: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    vertical_centers = (predicted[..., 1] + predicted[..., 3]) / 2
    title_mask = (labels == 0) & ~mask
    author_mask = ((labels == 2) | (labels == 3)) & ~mask
    title_penalty = torch.relu(vertical_centers - 0.5) * title_mask.float()
    author_penalty = torch.relu(0.5 - vertical_centers) * author_mask.float()
    return (title_penalty.sum() + author_penalty.sum()) / (
        title_mask.sum() + author_mask.sum() + 1e-6
    )


def set_dropout(model: nn.Module, probability: float) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = probability


def train_epoch(model, loader, optimizer, kl_weight, config, device) -> dict:
    model.train()
    totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
    batches = 0
    for batch in loader:
        labels = batch["labels"].to(device)
        boxes = batch["boxes"].to(device)
        mask = batch["mask"].to(device)
        subject_ids = batch["subject_id"].to(device)
        language_ids = batch["language_id"].to(device)
        predicted, mu, logvar = model.forward_train(
            labels, boxes, subject_ids, language_ids, mask
        )
        reconstruction = box_reconstruction_loss(predicted, boxes, mask)
        kl = kl_loss(mu, logvar, config["kl_free_bits"])
        loss = (
            config["reconstruction_weight"] * reconstruction
            + kl_weight * kl
            + config["overlap_weight"] * overlap_loss(predicted, mask)
            + config["structure_weight"] * structural_prior_loss(predicted, labels, mask)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        totals["loss"] += loss.item()
        totals["reconstruction"] += reconstruction.item()
        totals["kl"] += kl.item()
        batches += 1
    return {name: value / batches for name, value in totals.items()}


@torch.no_grad()
def validate(model, loader, device) -> float:
    model.eval()
    total = 0.0
    batches = 0
    for batch in loader:
        labels = batch["labels"].to(device)
        boxes = batch["boxes"].to(device)
        mask = batch["mask"].to(device)
        predicted, _, _ = model.forward_train(
            labels,
            boxes,
            batch["subject_id"].to(device),
            batch["language_id"].to(device),
            mask,
        )
        total += box_reconstruction_loss(predicted, boxes, mask).item()
        batches += 1
    return total / batches


def build_datasets(dataset_json: Path, split_root: Path, max_seq_len: int) -> dict:
    names = [
        "stage1_train",
        "stage1_validation",
        "stage2_train",
        "stage2_validation",
        "test",
    ]
    return {
        name: BookLayoutDataset(dataset_json, split_root / f"{name}.txt", max_seq_len)
        for name in names
    }


def check_split_counts(datasets: dict, expected: dict) -> None:
    actual = {name: len(dataset) for name, dataset in datasets.items()}
    if actual != expected:
        raise ValueError(f"Split counts differ from final protocol: {actual} != {expected}")


def run_seed(seed, config, datasets, checkpoint_root, device, logger) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    output = checkpoint_root / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=True)

    model_config = config["model"]
    stage1 = config["stage1"]
    stage2 = config["stage2"]
    model = LGCNet(**model_config).to(device)
    set_dropout(model, stage1["dropout"])

    stage1_train = DataLoader(
        datasets["stage1_train"],
        batch_size=stage1["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    stage1_validation = DataLoader(
        datasets["stage1_validation"],
        batch_size=stage1["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    optimizer = AdamW(
        model.parameters(), lr=stage1["learning_rate"], weight_decay=stage1["weight_decay"]
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=stage1["epochs"],
        eta_min=stage1["scheduler_eta_min"],
    )
    best_stage1 = float("inf")
    for epoch in range(1, stage1["epochs"] + 1):
        weight = min(
            stage1["kl_max_weight"],
            stage1["kl_max_weight"] * epoch / stage1["kl_warmup_epochs"],
        )
        train_metrics = train_epoch(model, stage1_train, optimizer, weight, stage1, device)
        validation = validate(model, stage1_validation, device)
        scheduler.step()
        if validation < best_stage1:
            best_stage1 = validation
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_recon": validation,
                    "config": model_config,
                    "stage_config": stage1,
                    "protocol": "mixed",
                    "seed": seed,
                },
                output / "stage1_best.pt",
            )
        logger(
            f"seed={seed} stage=1 epoch={epoch}/{stage1['epochs']} "
            f"train_recon={train_metrics['reconstruction']:.6f} "
            f"val_recon={validation:.6f} best={best_stage1:.6f}"
        )

    checkpoint1 = torch.load(
        output / "stage1_best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint1["model_state_dict"])
    set_dropout(model, stage2["dropout"])
    stage2_train = DataLoader(
        datasets["stage2_train"],
        batch_size=stage2["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    stage2_validation = DataLoader(
        datasets["stage2_validation"],
        batch_size=stage2["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    optimizer2 = AdamW(
        model.parameters(), lr=stage2["learning_rate"], weight_decay=stage2["weight_decay"]
    )
    scheduler2 = CosineAnnealingLR(
        optimizer2,
        T_max=stage2["epochs"],
        eta_min=stage2["scheduler_eta_min"],
    )
    initial_validation = validate(model, stage2_validation, device)
    best_stage2 = float("inf")
    best_epoch = 0
    patience = 0
    for epoch in range(1, stage2["epochs"] + 1):
        weight = min(
            stage2["kl_max_weight"],
            stage2["kl_max_weight"] * epoch / stage2["kl_warmup_epochs"],
        )
        train_metrics = train_epoch(model, stage2_train, optimizer2, weight, stage2, device)
        validation = validate(model, stage2_validation, device)
        scheduler2.step()
        logger(
            f"seed={seed} stage=2 epoch={epoch}/{stage2['epochs']} "
            f"train_recon={train_metrics['reconstruction']:.6f} "
            f"val_recon={validation:.6f} best={min(best_stage2, validation):.6f}"
        )
        if validation < best_stage2:
            best_stage2 = validation
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_recon": validation,
                    "initial_val_recon": initial_validation,
                    "improvement_pct": (initial_validation - validation)
                    / initial_validation
                    * 100,
                    "config": model_config,
                    "stage_config": stage2,
                    "protocol": "mixed",
                    "seed": seed,
                },
                output / "stage2_best.pt",
            )
        else:
            patience += 1
            if patience >= stage2["early_stop_patience"]:
                break

    checkpoint2 = torch.load(
        output / "stage2_best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint2["model_state_dict"])
    metrics = evaluate_model(
        model, datasets["test"], device, config["evaluation_seed"]
    )
    return {
        "seed": seed,
        **metrics,
        "stage1_val_recon": float(checkpoint1["val_recon"]),
        "stage2_val_recon": float(best_stage2),
        "stage2_best_epoch": best_epoch,
    }


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=repository_root / "configs" / "final_protocol.json"
    )
    parser.add_argument("--checkpoint-root", type=Path, default=repository_root / "checkpoints")
    parser.add_argument("--result-file", type=Path, default=repository_root / "outputs" / "m32g_results.json")
    parser.add_argument("--log-file", type=Path, default=repository_root / "outputs" / "training.log")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_json(args.config)
    device = torch.device(args.device)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)

    def logger(message: str) -> None:
        print(message, flush=True)
        with args.log_file.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    datasets = build_datasets(
        args.dataset_json, args.split_root, config["model"]["max_seq_len"]
    )
    check_split_counts(datasets, config["expected_split_counts"])
    runs = [
        run_seed(seed, config, datasets, args.checkpoint_root, device, logger)
        for seed in config["training_seeds"]
    ]

    aggregate = {}
    for metric in ("iou", "max_iou", "overlap", "max_overlap", "alignment", "violation"):
        values = np.array([run[metric] for run in runs], dtype=float)
        aggregate[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "runs": [float(value) for value in values],
        }
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": config["protocol_name"],
        "config": config,
        "per_seed": runs,
        "aggregate": aggregate,
    }
    with args.result_file.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    logger(f"results={args.result_file}")


if __name__ == "__main__":
    main()


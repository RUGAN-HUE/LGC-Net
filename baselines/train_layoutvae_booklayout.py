"""Train the LayoutVAE baseline used for the BookLayout-Bi comparison.

This author-created training wrapper imports the official DeepLayout
``layout_vae`` modules. The reconstruction weighting, KL warm-up, free-bits
floor, and L1 term are the settings used for the reported baseline runs.
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


RECONSTRUCTION_WEIGHT = 100.0
KL_MAX_WEIGHT = 0.01
KL_WARMUP_EPOCHS = 15
FREE_BITS = 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--train-coco", type=Path, required=True)
    parser.add_argument("--validation-coco", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--conditioning-size", type=int, default=128)
    parser.add_argument("--representation-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    module_root = args.upstream_root / "layout_vae"
    sys.path.insert(0, str(module_root))
    box_module = importlib.import_module("box")
    layout_module = importlib.import_module("layout")

    class BoxVAE(torch.nn.Module):
        def __init__(self, number_labels: int):
            super().__init__()
            self.encoder = box_module.AutoregressiveBoxEncoder(
                number_labels, args.conditioning_size, args.representation_size
            )
            self.decoder = box_module.AutoregressiveBoxDecoder(
                args.conditioning_size, args.representation_size
            )

        def forward(self, box, label_set, current_label, previous_labels, previous_boxes, state):
            mean, log_variance, condition, state = self.encoder(
                box, label_set, current_label, previous_labels, previous_boxes, state
            )
            standard_normal = torch.randn_like(mean)
            latent = mean + standard_normal * torch.exp(0.5 * log_variance)
            kl_per_dimension = -0.5 * (
                1 + log_variance - mean.square() - torch.exp(log_variance)
            )
            kl_per_dimension = torch.clamp(
                kl_per_dimension, min=FREE_BITS / args.representation_size
            )
            kl = torch.sum(kl_per_dimension, dim=1)
            return self.decoder(latent, condition), kl, state

    def reconstruction_loss(predicted, expected):
        mse = torch.mean((predicted - expected).square(), dim=-1)
        l1 = torch.mean(torch.abs(predicted - expected), dim=-1)
        return mse + l1

    def run_epoch(model, loader, optimizer, kl_weight: float) -> dict[str, float]:
        training = optimizer is not None
        model.train(training)
        reconstruction_values = []
        kl_values = []
        for _, targets in loader:
            label_set = torch.stack([target.label_set for target in targets]).to(device)
            boxes = [target.bbox.to(device) for target in targets]
            labels = [target.label.to(device) for target in targets]
            counts = np.asarray([len(target) for target in targets])
            batch_size = len(targets)
            max_boxes = int(counts.max())
            reconstruction = torch.zeros((batch_size, max_boxes), device=device)
            divergence = torch.zeros((batch_size, max_boxes), device=device)
            hidden = cell = None

            with torch.set_grad_enabled(training):
                for step in range(max_boxes):
                    has_box = counts > step
                    has_box_tensor = torch.as_tensor(has_box, dtype=torch.bool, device=device)
                    selected_boxes = [boxes[index] for index, keep in enumerate(has_box) if keep]
                    selected_labels = [labels[index] for index, keep in enumerate(has_box) if keep]
                    current_label = torch.stack(
                        [label[step] for label in selected_labels]
                    ).long()
                    current_label = label_encodings[current_label - 1]
                    current_box = torch.stack([box[step] for box in selected_boxes])
                    if step == 0:
                        previous_labels = torch.zeros(
                            (batch_size, 0, number_labels), device=device
                        )
                        previous_boxes = torch.zeros((batch_size, 0, 4), device=device)
                    else:
                        previous_labels = torch.stack(
                            [label[step - 1] for label in selected_labels]
                        ).long().unsqueeze(1)
                        previous_labels = label_encodings[previous_labels - 1]
                        previous_boxes = torch.stack(
                            [box[step - 1] for box in selected_boxes]
                        ).unsqueeze(1)
                    state = None
                    if step > 1 and hidden is not None and cell is not None:
                        state = (
                            hidden[has_box_tensor].unsqueeze(0),
                            cell[has_box_tensor].unsqueeze(0),
                        )
                    predicted, kl, state = model(
                        current_box,
                        label_set[has_box_tensor],
                        current_label,
                        previous_labels,
                        previous_boxes,
                        state,
                    )
                    reconstruction[has_box_tensor, step] = reconstruction_loss(predicted, current_box)
                    divergence[has_box_tensor, step] = kl
                    if state is not None:
                        hidden = torch.zeros((batch_size, args.conditioning_size), device=device)
                        cell = torch.zeros((batch_size, args.conditioning_size), device=device)
                        hidden[has_box_tensor] = state[0][-1]
                        cell[has_box_tensor] = state[1][-1]

                count_tensor = torch.from_numpy(counts).to(device=device, dtype=torch.float32)
                reconstruction_mean = torch.mean(reconstruction.sum(dim=1) / count_tensor)
                kl_mean = torch.mean(divergence.sum(dim=1) / count_tensor)
                loss = RECONSTRUCTION_WEIGHT * reconstruction_mean + kl_weight * kl_mean
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            reconstruction_values.append(float(reconstruction_mean.detach().cpu()))
            kl_values.append(float(kl_mean.detach().cpu()))
        return {
            "reconstruction": float(np.mean(reconstruction_values)),
            "kl": float(np.mean(kl_values)),
        }

    collator = layout_module.BatchCollator()
    training_dataset = layout_module.LayoutDataset(args.train_coco, args.max_length)
    validation_dataset = layout_module.LayoutDataset(args.validation_coco, args.max_length)
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    number_labels = training_dataset.number_labels
    label_encodings = torch.eye(number_labels, device=device)
    model = BoxVAE(number_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        kl_weight = min(KL_MAX_WEIGHT, KL_MAX_WEIGHT * epoch / KL_WARMUP_EPOCHS)
        training_metrics = run_epoch(model, training_loader, optimizer, kl_weight)
        with torch.no_grad():
            validation_metrics = run_epoch(model, validation_loader, None, kl_weight)
        history.append(
            {
                "epoch": epoch,
                "kl_weight": kl_weight,
                "training": training_metrics,
                "validation": validation_metrics,
            }
        )
        print(json.dumps(history[-1]))
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "seed": args.seed,
                    "model_state_dict": model.state_dict(),
                    "settings": {
                        "reconstruction_weight": RECONSTRUCTION_WEIGHT,
                        "kl_max_weight": KL_MAX_WEIGHT,
                        "kl_warmup_epochs": KL_WARMUP_EPOCHS,
                        "free_bits": FREE_BITS,
                        "representation_size": args.representation_size,
                    },
                },
                args.output_dir / f"epoch_{epoch:03d}.pt",
            )
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

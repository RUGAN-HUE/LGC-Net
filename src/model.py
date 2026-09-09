"""LGC-Net model used in the final five-seed experiments.

The architecture matches the private ``layout_model_v6b.py`` implementation:
nine role embeddings, subject and language conditions, a 32-dimensional CVAE
latent variable, and a non-autoregressive Transformer decoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


LABEL_NAMES = [
    "Title",
    "Subtitle",
    "Author",
    "Publisher",
    "Logo",
    "Main_Image",
    "Endorsement",
    "Badge",
    "Series",
]
LABEL_TO_INDEX = {name: index for index, name in enumerate(LABEL_NAMES)}
NUM_LABELS = 9
NUM_SUBJECTS = 3
NUM_LANGUAGES = 2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 20) -> None:
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        # Keep the original buffer name for checkpoint compatibility.
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.pe[:, : inputs.size(1)]


class LGCNet(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        latent_dim: int = 32,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 4,
        max_seq_len: int = 15,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len

        self.label_emb = nn.Embedding(NUM_LABELS, d_model)
        self.category_emb = nn.Embedding(NUM_SUBJECTS, d_model)
        self.language_emb = nn.Embedding(NUM_LANGUAGES, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len)

        self.box_embed = nn.Sequential(
            nn.Linear(4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.posterior_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)

        self.z_proj = nn.Linear(latent_dim, d_model)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=num_decoder_layers
        )
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),
            nn.Sigmoid(),
        )

    def encode(
        self,
        labels: torch.Tensor,
        boxes: torch.Tensor,
        subject_ids: torch.Tensor,
        language_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        label_embedding = self.label_emb(labels)
        box_embedding = self.box_embed(boxes)
        subject_embedding = self.category_emb(subject_ids).unsqueeze(1)
        language_embedding = self.language_emb(language_ids).unsqueeze(1)
        encoded = label_embedding + box_embedding + subject_embedding + language_embedding
        encoded = self.pos_enc(encoded)
        encoded = self.posterior_encoder(encoded, src_key_padding_mask=mask)

        if mask is not None:
            valid = (~mask).float().unsqueeze(-1)
            pooled = (encoded * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-6)
        else:
            pooled = encoded.mean(dim=1)
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled).clamp(-10, 10)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(
        self,
        latent: torch.Tensor,
        labels: torch.Tensor,
        subject_ids: torch.Tensor,
        language_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        label_embedding = self.label_emb(labels)
        subject_embedding = self.category_emb(subject_ids).unsqueeze(1)
        language_embedding = self.language_emb(language_ids).unsqueeze(1)
        decoded = label_embedding + subject_embedding + language_embedding
        decoded = self.pos_enc(decoded)
        latent_embedding = self.z_proj(latent).unsqueeze(1).expand(-1, labels.size(1), -1)
        decoded = self.decoder(decoded + latent_embedding, src_key_padding_mask=mask)
        return self.box_head(decoded)

    def forward_train(
        self,
        labels: torch.Tensor,
        boxes: torch.Tensor,
        subject_ids: torch.Tensor,
        language_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(labels, boxes, subject_ids, language_ids, mask)
        latent = self.reparameterize(mu, logvar)
        predicted_boxes = self.decode(latent, labels, subject_ids, language_ids, mask)
        return predicted_boxes, mu, logvar

    def forward_generate(
        self,
        labels: torch.Tensor,
        subject_ids: torch.Tensor,
        language_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latent is None:
            latent = torch.randn(labels.size(0), self.latent_dim, device=labels.device)
        return self.decode(latent, labels, subject_ids, language_ids, mask)


# Backward-compatible name used by the private experiment scripts.
LGC_Net_V6b = LGCNet

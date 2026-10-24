"""Shared-weight dual-encoder DistilBERT with gated fusion."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import AutoModel


class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(2 * dim, dim)

    def forward(self, title: Tensor, body: Tensor) -> tuple[Tensor, Tensor]:
        gate = torch.sigmoid(self.gate(torch.cat([title, body], dim=-1)))
        return gate * title + (1.0 - gate) * body, gate.mean(dim=-1)


class DualDistilBert(nn.Module):
    def __init__(self, model_name: str, condition: str = "fusion", dropout: float = 0.1):
        super().__init__()
        if condition not in ("title_only", "body_only", "fusion"):
            raise ValueError(f"unsupported condition: {condition}")
        self.condition = condition
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = int(self.encoder.config.hidden_size)
        self.fusion = GatedFusion(hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, title_ids: Tensor, title_mask: Tensor,
                body_ids: Tensor, body_mask: Tensor) -> dict[str, Tensor]:
        title = self.encoder(input_ids=title_ids, attention_mask=title_mask).last_hidden_state[:, 0]
        body = self.encoder(input_ids=body_ids, attention_mask=body_mask).last_hidden_state[:, 0]
        if self.condition == "title_only":
            hidden, gate = title, torch.ones(title.size(0), device=title.device)
        elif self.condition == "body_only":
            hidden, gate = body, torch.zeros(body.size(0), device=body.device)
        else:
            hidden, gate = self.fusion(title, body)
        logits = self.classifier(self.dropout(hidden)).squeeze(-1)
        return {"logit": logits, "gate": gate}

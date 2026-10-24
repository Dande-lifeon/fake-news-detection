"""Dual-encoder Bi-LSTM with gated fusion."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMEncoder(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden_dim: int, dropout: float):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(
            embedding.embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(4 * hidden_dim, hidden_dim)

    def forward(self, ids: Tensor, lengths: Tensor) -> Tensor:
        embedded = self.dropout(self.embedding(ids))
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        mask = (ids != 0).unsqueeze(-1)
        mean = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        maximum = output.masked_fill(~mask, -1e4).max(dim=1).values
        return self.proj(torch.cat([mean, maximum], dim=-1))


class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(2 * dim, dim)

    def forward(self, title: Tensor, body: Tensor) -> tuple[Tensor, Tensor]:
        gate = torch.sigmoid(self.gate(torch.cat([title, body], dim=-1)))
        return gate * title + (1.0 - gate) * body, gate.mean(dim=-1)


class DualBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, cfg, condition: str = "fusion"):
        super().__init__()
        if condition not in ("title_only", "body_only", "fusion"):
            raise ValueError(f"unsupported condition: {condition}")
        self.condition = condition
        dropout = float(cfg.dropout)
        hidden = int(cfg.hidden_dim)
        embedding = nn.Embedding(vocab_size, int(cfg.embed_dim), padding_idx=0)
        self.title_encoder = BiLSTMEncoder(embedding, hidden, dropout)
        self.body_encoder = BiLSTMEncoder(embedding, hidden, dropout)
        self.fusion = GatedFusion(hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, title_ids: Tensor, title_lens: Tensor,
                body_ids: Tensor, body_lens: Tensor) -> dict[str, Tensor]:
        title = self.title_encoder(title_ids, title_lens)
        body = self.body_encoder(body_ids, body_lens)
        if self.condition == "title_only":
            hidden, gate = title, torch.ones(title.size(0), device=title.device)
        elif self.condition == "body_only":
            hidden, gate = body, torch.zeros(body.size(0), device=body.device)
        else:
            hidden, gate = self.fusion(title, body)
        logits = self.classifier(self.dropout(hidden)).squeeze(-1)
        return {"logit": logits, "gate": gate}

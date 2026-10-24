"""Datasets, vocabulary, and collators for the neural models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass
class Vocab:
    stoi: dict[str, int]

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        min_freq: int = 2,
        max_size: int = 30000,
    ) -> "Vocab":
        counts: dict[str, int] = {}
        for text in texts:
            for token in TOKEN_RE.findall(str(text).lower()):
                counts[token] = counts.get(token, 0) + 1
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        tokens = [token for token, freq in ordered if freq >= min_freq][: max_size - 2]
        stoi = {"<pad>": 0, "<unk>": 1}
        for index, token in enumerate(tokens, start=2):
            stoi[token] = index
        return cls(stoi=stoi)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(token, 1) for token in TOKEN_RE.findall(str(text).lower())]
        return ids[:max_len] if ids else [1]

    def save(self, path: Path) -> None:
        path.write_text(
            "\n".join(f"{token}\t{index}" for token, index in self.stoi.items()),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        stoi = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            token, index = line.split("\t")
            stoi[token] = int(index)
        return cls(stoi=stoi)


class BiLSTMDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, vocab: Vocab, max_title: int, max_body: int):
        self.frame = frame.reset_index(drop=True)
        self.vocab = vocab
        self.max_title = max_title
        self.max_body = max_body

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        title_ids = self.vocab.encode(row["title"], self.max_title)
        body_ids = self.vocab.encode(row["text_clean"], self.max_body)
        return {
            "id": int(row["id"]),
            "label": float(row["label"]),
            "title_ids": torch.tensor(title_ids, dtype=torch.long),
            "body_ids": torch.tensor(body_ids, dtype=torch.long),
        }


def collate_bilstm(batch: list[dict]) -> dict:
    title_ids = pad_sequence([item["title_ids"] for item in batch], batch_first=True, padding_value=0)
    body_ids = pad_sequence([item["body_ids"] for item in batch], batch_first=True, padding_value=0)
    return {
        "id": torch.tensor([item["id"] for item in batch], dtype=torch.long),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "title_ids": title_ids,
        "title_lens": (title_ids != 0).sum(dim=1),
        "body_ids": body_ids,
        "body_lens": (body_ids != 0).sum(dim=1),
    }


def encode_body(tokenizer, text: str, head: int, tail: int, max_len: int) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) > head + tail:
        ids = ids[:head] + ids[-tail:]
    ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
    ids = ids[:max_len]
    return ids + [tokenizer.pad_token_id] * (max_len - len(ids))


class DistilBertDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_title: int, max_body: int,
                 head: int, tail: int):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_title = max_title
        self.max_body = max_body
        self.head = head
        self.tail = tail

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        title = self.tokenizer(
            str(row["title"]),
            truncation=True,
            max_length=self.max_title,
            padding="max_length",
        )
        body_ids = encode_body(
            self.tokenizer, str(row["text_clean"]), self.head, self.tail, self.max_body
        )
        return {
            "id": int(row["id"]),
            "label": float(row["label"]),
            "title_ids": torch.tensor(title["input_ids"], dtype=torch.long),
            "title_mask": torch.tensor(title["attention_mask"], dtype=torch.long),
            "body_ids": torch.tensor(body_ids, dtype=torch.long),
            "body_mask": torch.tensor([int(token != self.tokenizer.pad_token_id) for token in body_ids],
                                     dtype=torch.long),
        }


def collate_bert(batch: list[dict]) -> dict:
    return {
        "id": torch.tensor([item["id"] for item in batch], dtype=torch.long),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "title_ids": torch.stack([item["title_ids"] for item in batch]),
        "title_mask": torch.stack([item["title_mask"] for item in batch]),
        "body_ids": torch.stack([item["body_ids"] for item in batch]),
        "body_mask": torch.stack([item["body_mask"] for item in batch]),
    }


def class_balance(frame: pd.DataFrame) -> float:
    """Return the positive-class share; useful for logging and sanity checks."""
    return float(np.mean(frame["label"].to_numpy() == 1))

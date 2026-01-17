"""
Training PPI models using mask, binary classification,
and optional residue-residue auxiliary supervision.
"""

import os
import random
import csv
import logging
from datetime import datetime
from typing import Dict, Type, Callable, NamedTuple

import torch
from torch import nn
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling
)
from sentence_transformers import LoggingHandler, util, SentenceTransformer

from utils.ddp import ddp_setup
from utils.data_load import load_train_objs

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Helper: online negative sampling for residue pairs
# ------------------------------------------------------------

def sample_residue_pairs(pos_pairs, lenA, lenB, neg_ratio=10):
    pos_pairs = list(pos_pairs)
    pos_set = set(pos_pairs)
    neg_pairs = set()

    while len(neg_pairs) < neg_ratio * len(pos_pairs):
        i = random.randint(1, lenA)
        j = random.randint(lenA + 2, lenA + lenB + 1)
        if (i, j) not in pos_set:
            neg_pairs.add((i, j))

    pairs = pos_pairs + list(neg_pairs)
    labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

    return (
        torch.tensor(pairs, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float),
    )

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class PLMinteract(nn.Module):
    def __init__(
        self,
        checkpoint,
        num_labels,
        config,
        device,
        embedding_size,
        weight_loss_class,
        weight_loss_mlm,
        lambda_residue=0.1,
    ):
        super().__init__()

        self.esm_mask = AutoModelForMaskedLM.from_pretrained(
            checkpoint, config=config
        )

        self.classifier = nn.Linear(embedding_size, 1)

        # Residue-residue head
        self.residue_proj = nn.Linear(embedding_size, embedding_size, bias=False)

        self.device = device
        self.weight_loss_class = weight_loss_class
        self.weight_loss_mlm = weight_loss_mlm
        self.lambda_residue = lambda_residue

    def forward(
        self,
        labels,
        lm_dataloader,
        residue_pairs=None,
        residue_labels=None,
    ):
        for lm_features in lm_dataloader:
            lm_features = lm_features.to(self.device)
            features = {
                "input_ids": lm_features["input_ids"],
                "attention_mask": lm_features["attention_mask"],
            }

        # MLM loss
        mlm_out = self.esm_mask(**lm_features)
        MLM_loss = mlm_out.loss

        # Encoder output
        enc_out = self.esm_mask.base_model(**features, return_dict=True)
        hidden = enc_out.last_hidden_state  # [B, L, D]

        # CLS PPI head
        cls_emb = F.relu(hidden[:, 0, :])
        logits = self.classifier(cls_emb).view(-1)

        pos_weight = torch.tensor([10.0], device=self.device)
        class_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
            logits, labels.view(-1)
        )

        # Residue auxiliary loss
        residue_loss = torch.tensor(0.0, device=self.device)

        if residue_pairs is not None and residue_labels is not None:
            i_idx = residue_pairs[:, 0]
            j_idx = residue_pairs[:, 1]

            h_i = self.residue_proj(hidden[:, i_idx, :])
            h_j = self.residue_proj(hidden[:, j_idx, :])

            scores = torch.sum(h_i * h_j, dim=-1).view(-1)

            residue_loss = nn.BCEWithLogitsLoss()(scores, residue_labels)

        total_loss = (
            self.weight_loss_class * class_loss
            + self.weight_loss_mlm * MLM_loss
            + self.lambda_residue * residue_loss
        )

        return total_loss, class_loss, MLM_loss, residue_loss, logits

# ------------------------------------------------------------
# CrossEncoder Trainer
# ------------------------------------------------------------

class CrossEncoder:
    def __init__(
        self,
        model_name,
        num_labels,
        max_length,
        embedding_size,
        weight_loss_class,
        weight_loss_mlm,
        checkpoint=None,
    ):
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        seed_offset, rank, local_rank, device = ddp_setup()
        self.device = device
        self.master_process = rank == 0

        self.model = PLMinteract(
            model_name,
            num_labels,
            self.config,
            device,
            embedding_size,
            weight_loss_class,
            weight_loss_mlm,
        )

        if checkpoint is not None:
            ckpt = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(ckpt["model"])

        self.model = DDP(
            self.model.to(device),
            device_ids=[local_rank],
            find_unused_parameters=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm_probability=0.22
        )

    def smart_batching_collate(self, batch):
        texts = [[], []]
        labels = []

        residue_pairs, residue_labels = None, None

        for ex in batch:
            texts[0].append(ex.texts[0].strip())
            texts[1].append(ex.texts[1].strip())
            labels.append(ex.label)

            if hasattr(ex, "residue_pairs") and ex.residue_pairs is not None:
                rp, rl = sample_residue_pairs(
                    ex.residue_pairs, ex.lenA, ex.lenB
                )
                residue_pairs = rp.to(self.device)
                residue_labels = rl.to(self.device)

        tokenized = self.tokenizer(
            *texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        labels = torch.tensor(labels, dtype=torch.float).to(self.device)

        return tokenized, labels, residue_pairs, residue_labels

    def train(
        self,
        args,
        train_dataloader,
        epochs,
        warmup_steps,
        output_path,
        gradient_accumulation_steps,
    ):
        train_dataloader.collate_fn = self.smart_batching_collate

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        scheduler = SentenceTransformer._get_scheduler(
            optimizer,
            scheduler="WarmupLinear",
            warmup_steps=warmup_steps,
            t_total=len(train_dataloader) * epochs,
        )

        self.model.train()

        for epoch in range(epochs):
            train_dataloader.sampler.set_epoch(epoch)
            optimizer.zero_grad()

            for step, (features, labels, rp, rl) in enumerate(train_dataloader):
                lm_features = Dataset.from_dict(features)
                lm_loader = DataLoader(
                    lm_features,
                    batch_size=features["input_ids"].size(0),
                    collate_fn=self.data_collator,
                )

                loss, _, _, _, _ = self.model(
                    labels, lm_loader, rp, rl
                )
                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            if self.master_process:
                os.makedirs(output_path, exist_ok=True)
                torch.save(
                    {"model": self.model.module.state_dict()},
                    os.path.join(output_path, f"epoch_{epoch}.pt"),
                )

# ------------------------------------------------------------
# Args / main
# ------------------------------------------------------------

class TrainArgs(NamedTuple):
    epochs: int
    train_filepath: str
    output_filepath: str
    model_name: str
    embedding_size: int
    max_length: int
    warmup_steps: int
    gradient_accumulation_steps: int
    weight_loss_class: int
    weight_loss_mlm: int
    resume_from_checkpoint: str

def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    trainer = CrossEncoder(
        args.model_name,
        num_labels=1,
        max_length=args.max_length,
        embedding_size=args.embedding_size,
        weight_loss_class=args.weight_loss_class,
        weight_loss_mlm=args.weight_loss_mlm,
        checkpoint=args.resume_from_checkpoint,
    )

    train_samples = load_train_objs(args.train_filepath)
    train_loader = DataLoader(
        train_samples,
        batch_size=1,
        sampler=DistributedSampler(train_samples),
    )

    trainer.train(
        args,
        train_loader,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        output_path=args.output_filepath,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_filepath", required=True)
    parser.add_argument("--output_filepath", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--embedding_size", type=int, required=True)
    parser.add_argument("--max_length", type=int, default=1600)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--weight_loss_class", type=int, default=10)
    parser.add_argument("--weight_loss_mlm", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", default=None)

    main(parser.parse_args())

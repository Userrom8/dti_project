# src/models/esm_encoder.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class ProteinEncoderESM(nn.Module):
    """
    ESM2 Protein Encoder.
    Converts a protein FASTA string into a fixed-length embedding vector.

    Recommended models:
        - facebook/esm2_t6_8M_UR50D (small, fast)
        - facebook/esm2_t12_35M_UR50D (medium)
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        device: str = "cpu",
        max_length: int = 1024,
        freeze: bool = True,
    ):
        super().__init__()

        self.device = device
        self.max_length = max_length

        # Load ESM2 tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name).to(device)

        # Freeze for efficiency
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.output_dim = self.model.config.hidden_size

    def forward(self, sequences):
        """
        sequences: list[str] of protein sequences (FASTA, raw AA strings)

        Returns:
            Tensor of shape [batch_size, embedding_dim]
        """

        # Tokenize batch
        enc = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**enc)

        # Mean pool over sequence length
        # out.last_hidden_state shape = [B, L, D]
        last_hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)

        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)

        return pooled  # shape [B, D]

"""Production screening model: ESM-2 + multi-task heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict


class ProductionFunctionalHead(nn.Module):
    """Functional embedding with residual connections."""

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0.1):
        super().__init__()
        blocks = []
        residual_projs = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            blocks.append(nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
            residual_projs.append(
                nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
            )
            in_dim = hidden_dim
        self.blocks = nn.ModuleList(blocks)
        self.residual_projs = nn.ModuleList(residual_projs)
        self.final_proj = nn.Linear(in_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block, res_proj in zip(self.blocks, self.residual_projs):
            residual = res_proj(x)
            x = block(x) + residual
        x = self.final_proj(x)
        return F.normalize(x, p=2, dim=-1)


class MechanismClassifier(nn.Module):
    """Multi-class mechanism-of-harm classifier with confidence."""

    def __init__(self, input_dim: int, num_mechanisms: int, dropout: float = 0.15):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
        )
        self.class_head = nn.Linear(128, num_mechanisms)
        self.confidence_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.network(x)
        return {
            "mechanism_logits": self.class_head(features),
            "confidence": torch.sigmoid(self.confidence_head(features)),
        }


class RiskScorer(nn.Module):
    """Continuous risk scoring with uncertainty estimation."""

    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
        )
        self.mean_head = nn.Linear(64, 1)
        self.logvar_head = nn.Linear(64, 1)
        # CHANGED: Initialize logvar to output small values (fixes ±1.0 saturation)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -3.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.network(x)
        risk_mean = torch.sigmoid(self.mean_head(features))
        # CHANGED: tighter clamp range
        risk_logvar = self.logvar_head(features).clamp(-10, 2)
        risk_std = torch.exp(0.5 * risk_logvar)
        return {
            "risk_score": risk_mean.squeeze(-1),
            "risk_uncertainty": risk_std.squeeze(-1),
        }


class ContrastiveProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


class ProductionScreeningModel(nn.Module):
    """
    Production multi-task screening model.
    Tasks: binary detection, mechanism classification, risk scoring, embedding.
    """

    def __init__(
        self,
        esm_model_name: str = "facebook/esm2_t36_3B_UR50D",
        esm_embedding_dim: int = 2560,
        functional_dim: int = 256,
        projection_dim: int = 128,
        num_mechanisms: int = 8,
        freeze_layers: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.esm_model_name = esm_model_name
        self.functional_dim = functional_dim
        self.num_mechanisms = num_mechanisms

        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
        self.esm = AutoModel.from_pretrained(esm_model_name, torch_dtype=torch.float32)

        total_layers = len(self.esm.encoder.layer)
        actual_freeze = min(freeze_layers, total_layers)
        for i, layer in enumerate(self.esm.encoder.layer):
            if i < actual_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

        self.functional_head = ProductionFunctionalHead(
            esm_embedding_dim, [1024, 512], functional_dim, dropout
        )
        self.projection_head = ContrastiveProjectionHead(functional_dim, projection_dim)
        self.binary_head = nn.Sequential(
            nn.Linear(functional_dim, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 2),
        )
        self.mechanism_head = MechanismClassifier(functional_dim, num_mechanisms, dropout)
        self.risk_head = RiskScorer(functional_dim, dropout)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.07)))

    @property
    def temperature(self):
        return torch.clamp(self.log_temperature.exp(), min=0.01, max=1.0)

    def encode_sequences(self, sequences: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            sequences, return_tensors="pt", padding=True,
            truncation=True, max_length=1024,
        ).to(next(self.esm.parameters()).device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = self.esm(**tokens)
        attention_mask = tokens["attention_mask"].unsqueeze(-1).float()
        hidden_states = outputs.last_hidden_state.float()
        pooled = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        return pooled

    def forward(self, sequences: List[str], return_all: bool = False) -> Dict[str, torch.Tensor]:
        esm_output = self.encode_sequences(sequences)
        functional_emb = self.functional_head(esm_output)
        binary_logits = self.binary_head(functional_emb)
        mechanism_out = self.mechanism_head(functional_emb)
        risk_out = self.risk_head(functional_emb)
        result = {
            "binary_logits": binary_logits,
            "mechanism_logits": mechanism_out["mechanism_logits"],
            "mechanism_confidence": mechanism_out["confidence"],
            "risk_score": risk_out["risk_score"],
            "risk_uncertainty": risk_out["risk_uncertainty"],
            "functional_embedding": functional_emb,
        }
        if return_all:
            result["projection"] = self.projection_head(functional_emb)
            result["esm_output"] = esm_output
        return result

"""
Production BioScreener - the main entry point.

Usage:
    from bioscreen import BioScreener

    screener = BioScreener.from_pretrained("bioscreen-v1")
    results = screener.screen(["MKFLVL..."])

    for result in results:
        print(result.decision, result.predicted_mechanism)
"""

import hashlib
import time
import logging
from typing import List, Optional, Dict, Union
from pathlib import Path

import torch
import torch.nn.functional as F

from bioscreen.screening.result import ScreeningResult, ScreeningDecision, RiskLevel

logger = logging.getLogger("bioscreen")

MIN_SEQUENCE_LENGTH = 10


class BioScreener:
    """
    Production-grade protein sequence screener.

    Screens protein sequences for biological threat potential using
    a fine-tuned ESM-2 model with multi-task prediction:
    - Binary threat detection
    - Mechanism-of-harm classification
    - Continuous risk scoring with uncertainty
    """

    DEFAULT_BLOCK_THRESHOLD = 0.9
    DEFAULT_REVIEW_THRESHOLD = 0.5

    MECHANISMS = [
        "enzymatic_disruption",
        "hemolysis",
        "host_adhesion",
        "immune_evasion",
        "membrane_disruption",
        "neurotoxicity",
        "viral_entry",
        "benign",
    ]

    def __init__(
        self,
        model,
        device: str = "cuda",
        block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
        review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
        batch_size: int = 16,
        max_sequence_length: int = 1500,
    ):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.block_threshold = block_threshold
        self.review_threshold = review_threshold
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: str = "cuda",
        **kwargs,
    ) -> "BioScreener":
        """Load a pretrained screening model."""
        import json
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        from bioscreen.models.production_model import ProductionScreeningModel

        model = ProductionScreeningModel(
            num_mechanisms=config.get("num_mechanisms", 8),
        )

        weights_path = model_path / "model_weights.pt"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded model from {weights_path}")

        return cls(model=model, device=device, **kwargs)

    def _validate_sequence(self, sequence: str) -> Optional[str]:
        if not sequence:
            return "Empty sequence"
        valid_aa = set("ACDEFGHIKLMNPQRSTVWXY")
        invalid = set(sequence.upper()) - valid_aa
        if invalid:
            return f"Invalid amino acids: {invalid}"
        if len(sequence) > self.max_sequence_length:
            return f"Sequence too long: {len(sequence)} > {self.max_sequence_length}"
        if len(sequence) < MIN_SEQUENCE_LENGTH:
            return f"Sequence too short: {len(sequence)} < {MIN_SEQUENCE_LENGTH}"
        return None

    @staticmethod
    def _hash_sequence(sequence: str) -> str:
        return hashlib.sha256(sequence.encode()).hexdigest()[:16]

    def _risk_level_from_score(self, score: float) -> RiskLevel:
        if score >= 0.8:
            return RiskLevel.CRITICAL
        elif score >= 0.6:
            return RiskLevel.HIGH
        elif score >= 0.4:
            return RiskLevel.MEDIUM
        elif score >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.NONE

    @torch.no_grad()
    def screen(
        self,
        sequences: Union[str, List[str]],
        request_id: Optional[str] = None,
    ) -> List[ScreeningResult]:
        """Screen one or more protein sequences."""
        if isinstance(sequences, str):
            sequences = [sequences]

        results = []
        valid_indices = []
        for i, seq in enumerate(sequences):
            error = self._validate_sequence(seq)
            if error:
                results.append(ScreeningResult.error(error, self._hash_sequence(seq)))
            else:
                valid_indices.append(i)
                results.append(None)

        valid_sequences = [sequences[i] for i in valid_indices]

        for batch_start in range(0, len(valid_sequences), self.batch_size):
            batch = valid_sequences[batch_start:batch_start + self.batch_size]
            batch_indices = valid_indices[batch_start:batch_start + self.batch_size]

            t_start = time.perf_counter()
            outputs = self.model(batch)
            binary_probs = F.softmax(outputs["binary_logits"], dim=-1)
            mechanism_probs = F.softmax(outputs["mechanism_logits"], dim=-1)
            screening_time = (time.perf_counter() - t_start) * 1000 / len(batch)

            for j, idx in enumerate(batch_indices):
                threat_prob = binary_probs[j, 1].item()
                top_mech_prob, top_mech_id = mechanism_probs[j].max(dim=-1)
                mechanism_name = (
                    self.MECHANISMS[top_mech_id.item()]
                    if top_mech_id.item() < len(self.MECHANISMS)
                    else "unknown"
                )
                risk_score = outputs["risk_score"][j].item()
                risk_uncertainty = outputs["risk_uncertainty"][j].item()

                if threat_prob >= self.block_threshold:
                    decision = ScreeningDecision.BLOCK
                elif threat_prob >= self.review_threshold:
                    decision = ScreeningDecision.MANUAL_REVIEW
                else:
                    decision = ScreeningDecision.PASS

                all_mech_probs = {
                    self.MECHANISMS[k] if k < len(self.MECHANISMS) else f"class_{k}":
                    round(mechanism_probs[j, k].item(), 4)
                    for k in range(mechanism_probs.shape[1])
                }

                results[idx] = ScreeningResult(
                    decision=decision,
                    sequence_hash=self._hash_sequence(sequences[idx]),
                    threat_probability=round(threat_prob, 4),
                    predicted_mechanism=mechanism_name,
                    mechanism_confidence=round(top_mech_prob.item(), 4),
                    all_mechanism_probabilities=all_mech_probs,
                    risk_score=round(risk_score, 4),
                    risk_uncertainty=round(risk_uncertainty, 4),
                    risk_level=self._risk_level_from_score(risk_score),
                    sequence_length=len(sequences[idx]),
                    screening_time_ms=round(screening_time, 2),
                    request_id=request_id,
                )

        return results

    def screen_fasta(self, fasta_path: Union[str, Path]) -> List[ScreeningResult]:
        """Screen all sequences in a FASTA file."""
        from Bio import SeqIO
        sequences = [str(record.seq) for record in SeqIO.parse(str(fasta_path), "fasta")]
        return self.screen(sequences)

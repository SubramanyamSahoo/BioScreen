"""Screening result data structures."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
from enum import Enum
import json


class ScreeningDecision(str, Enum):
    PASS = "PASS"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    BLOCK = "BLOCK"
    ERROR = "ERROR"


class RiskLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ScreeningResult:
    """
    Structured screening result for a single sequence.

    Designed to be compatible with SecureDNA and IBBIS Common
    Mechanism response formats.
    """
    decision: ScreeningDecision
    sequence_hash: str

    threat_probability: float

    predicted_mechanism: str
    mechanism_confidence: float
    all_mechanism_probabilities: Dict[str, float] = field(default_factory=dict)

    risk_score: float = 0.0
    risk_uncertainty: float = 0.0
    risk_level: RiskLevel = RiskLevel.NONE

    sequence_length: int = 0
    model_version: str = "1.0.0"
    screening_time_ms: float = 0.0

    request_id: Optional[str] = None

    def to_dict(self) -> Dict:
        result = asdict(self)
        result["decision"] = self.decision.value
        result["risk_level"] = self.risk_level.value
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def error(cls, message: str, sequence_hash: str = "") -> "ScreeningResult":
        return cls(
            decision=ScreeningDecision.ERROR,
            sequence_hash=sequence_hash,
            threat_probability=0.0,
            predicted_mechanism="error",
            mechanism_confidence=0.0,
        )

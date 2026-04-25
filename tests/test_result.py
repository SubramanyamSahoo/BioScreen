"""Tests for ScreeningResult data structures."""

import json
import pytest
from bioscreen.screening.result import (
    ScreeningResult,
    ScreeningDecision,
    RiskLevel,
)


class TestScreeningDecision:
    def test_enum_values(self):
        assert ScreeningDecision.PASS.value == "PASS"
        assert ScreeningDecision.MANUAL_REVIEW.value == "MANUAL_REVIEW"
        assert ScreeningDecision.BLOCK.value == "BLOCK"
        assert ScreeningDecision.ERROR.value == "ERROR"


class TestRiskLevel:
    def test_enum_values(self):
        assert RiskLevel.NONE.value == "none"
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"


class TestScreeningResult:
    @pytest.fixture
    def sample_result(self):
        return ScreeningResult(
            decision=ScreeningDecision.BLOCK,
            sequence_hash="abc123def456",
            threat_probability=0.95,
            predicted_mechanism="neurotoxicity",
            mechanism_confidence=0.88,
            all_mechanism_probabilities={
                "neurotoxicity": 0.88,
                "membrane_disruption": 0.05,
                "benign": 0.02,
            },
            risk_score=0.91,
            risk_uncertainty=0.03,
            risk_level=RiskLevel.CRITICAL,
            sequence_length=150,
        )

    def test_to_dict(self, sample_result):
        d = sample_result.to_dict()
        assert isinstance(d, dict)
        assert d["decision"] == "BLOCK"
        assert d["risk_level"] == "critical"
        assert d["threat_probability"] == 0.95
        assert d["predicted_mechanism"] == "neurotoxicity"

    def test_to_json(self, sample_result):
        j = sample_result.to_json()
        parsed = json.loads(j)
        assert parsed["decision"] == "BLOCK"
        assert parsed["sequence_hash"] == "abc123def456"
        assert parsed["risk_score"] == 0.91

    def test_to_json_is_valid_json(self, sample_result):
        j = sample_result.to_json()
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_error_factory(self):
        r = ScreeningResult.error("Something broke", "hash123")
        assert r.decision == ScreeningDecision.ERROR
        assert r.sequence_hash == "hash123"
        assert r.predicted_mechanism == "error"
        assert r.threat_probability == 0.0
        assert r.mechanism_confidence == 0.0

    def test_error_factory_default_hash(self):
        r = ScreeningResult.error("fail")
        assert r.sequence_hash == ""

    def test_default_values(self):
        r = ScreeningResult(
            decision=ScreeningDecision.PASS,
            sequence_hash="test",
            threat_probability=0.1,
            predicted_mechanism="benign",
            mechanism_confidence=0.9,
        )
        assert r.risk_score == 0.0
        assert r.risk_uncertainty == 0.0
        assert r.risk_level == RiskLevel.NONE
        assert r.sequence_length == 0
        assert r.model_version == "1.0.0"
        assert r.request_id is None
        assert r.all_mechanism_probabilities == {}

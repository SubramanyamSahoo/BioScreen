"""Tests for BioScreener (validation + hashing, no GPU needed)."""

import pytest
from unittest.mock import MagicMock, patch
from bioscreen.screening.screener import BioScreener
from bioscreen.screening.result import ScreeningDecision, RiskLevel


class TestSequenceValidation:
    """Test input validation (no model needed)."""

    @pytest.fixture
    def screener(self):
        """Create a screener with a mock model."""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model.parameters = MagicMock(return_value=iter([]))
        with patch("torch.cuda.is_available", return_value=False):
            return BioScreener(model=mock_model, device="cpu")

    def test_empty_sequence(self, screener):
        error = screener._validate_sequence("")
        assert error is not None
        assert "Empty" in error

    def test_too_short(self, screener):
        error = screener._validate_sequence("MKVL")
        assert error is not None
        assert "too short" in error

    def test_too_long(self, screener):
        error = screener._validate_sequence("M" * 2000)
        assert error is not None
        assert "too long" in error

    def test_invalid_amino_acids(self, screener):
        error = screener._validate_sequence("MKVLB123ZZZZ")
        assert error is not None
        assert "Invalid" in error

    def test_valid_sequence(self, screener):
        seq = "MKFLVLLFNILCLFPVLAADNHGVSMRV"
        error = screener._validate_sequence(seq)
        assert error is None

    def test_valid_with_all_amino_acids(self, screener):
        seq = "ACDEFGHIKLMNPQRSTVWXY"
        error = screener._validate_sequence(seq)
        assert error is None

    def test_min_length_boundary(self, screener):
        seq = "MKFLVLLFNI"  # exactly 10
        error = screener._validate_sequence(seq)
        assert error is None

    def test_max_length_boundary(self, screener):
        seq = "M" * 1500  # exactly max
        error = screener._validate_sequence(seq)
        assert error is None

    def test_over_max_length(self, screener):
        seq = "M" * 1501
        error = screener._validate_sequence(seq)
        assert error is not None


class TestSequenceHashing:
    def test_hash_deterministic(self):
        h1 = BioScreener._hash_sequence("MKFLVLLFNI")
        h2 = BioScreener._hash_sequence("MKFLVLLFNI")
        assert h1 == h2

    def test_hash_length(self):
        h = BioScreener._hash_sequence("MKFLVLLFNI")
        assert len(h) == 16

    def test_hash_different_sequences(self):
        h1 = BioScreener._hash_sequence("MKFLVLLFNI")
        h2 = BioScreener._hash_sequence("MVLSPADKTN")
        assert h1 != h2


class TestRiskLevelMapping:
    @pytest.fixture
    def screener(self):
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model.parameters = MagicMock(return_value=iter([]))
        with patch("torch.cuda.is_available", return_value=False):
            return BioScreener(model=mock_model, device="cpu")

    def test_none_risk(self, screener):
        assert screener._risk_level_from_score(0.0) == RiskLevel.NONE
        assert screener._risk_level_from_score(0.19) == RiskLevel.NONE

    def test_low_risk(self, screener):
        assert screener._risk_level_from_score(0.2) == RiskLevel.LOW
        assert screener._risk_level_from_score(0.39) == RiskLevel.LOW

    def test_medium_risk(self, screener):
        assert screener._risk_level_from_score(0.4) == RiskLevel.MEDIUM
        assert screener._risk_level_from_score(0.59) == RiskLevel.MEDIUM

    def test_high_risk(self, screener):
        assert screener._risk_level_from_score(0.6) == RiskLevel.HIGH
        assert screener._risk_level_from_score(0.79) == RiskLevel.HIGH

    def test_critical_risk(self, screener):
        assert screener._risk_level_from_score(0.8) == RiskLevel.CRITICAL
        assert screener._risk_level_from_score(1.0) == RiskLevel.CRITICAL


class TestMechanismList:
    def test_mechanism_count(self):
        assert len(BioScreener.MECHANISMS) == 8

    def test_benign_included(self):
        assert "benign" in BioScreener.MECHANISMS

    def test_threat_mechanisms(self):
        expected_threats = {
            "enzymatic_disruption", "hemolysis", "host_adhesion",
            "immune_evasion", "membrane_disruption", "neurotoxicity",
            "viral_entry",
        }
        actual = set(BioScreener.MECHANISMS) - {"benign"}
        assert actual == expected_threats

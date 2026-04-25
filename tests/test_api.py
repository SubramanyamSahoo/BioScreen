"""Tests for API request/response models (no server needed)."""

import pytest
from pydantic import ValidationError


def test_screening_request_valid():
    """Test that valid requests pass validation."""
    from bioscreen.api.server import ScreeningRequest
    req = ScreeningRequest(sequences=["MKFLVLLFNILCLFPVLAADNHGVSMRV"])
    assert len(req.sequences) == 1
    assert req.sequences[0] == "MKFLVLLFNILCLFPVLAADNHGVSMRV"


def test_screening_request_uppercased():
    """Test that sequences are uppercased."""
    from bioscreen.api.server import ScreeningRequest
    req = ScreeningRequest(sequences=["mkflvllfnilclfpvlaad"])
    assert req.sequences[0] == "MKFLVLLFNILCLFPVLAAD"


def test_screening_request_too_short():
    """Test that short sequences are rejected."""
    from bioscreen.api.server import ScreeningRequest
    with pytest.raises(ValidationError):
        ScreeningRequest(sequences=["MKF"])


def test_screening_request_too_long():
    """Test that long sequences are rejected."""
    from bioscreen.api.server import ScreeningRequest
    with pytest.raises(ValidationError):
        ScreeningRequest(sequences=["M" * 1501])


def test_screening_request_empty():
    """Test that empty sequence list is rejected."""
    from bioscreen.api.server import ScreeningRequest
    with pytest.raises(ValidationError):
        ScreeningRequest(sequences=[])


def test_screening_request_with_id():
    """Test optional request_id."""
    from bioscreen.api.server import ScreeningRequest
    req = ScreeningRequest(
        sequences=["MKFLVLLFNILCLFPVLAAD"],
        request_id="test-123",
    )
    assert req.request_id == "test-123"


def test_health_response_model():
    """Test HealthResponse serialization."""
    from bioscreen.api.server import HealthResponse
    resp = HealthResponse(
        status="healthy",
        model_loaded=True,
        gpu_available=True,
        gpu_memory_gb=79.6,
    )
    assert resp.status == "healthy"
    assert resp.gpu_memory_gb == 79.6

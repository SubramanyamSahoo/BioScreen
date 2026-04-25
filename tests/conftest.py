"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def sample_sequences():
    """A set of test protein sequences."""
    return {
        "short_valid": "MKFLVLLFNI",
        "medium_valid": "MKFLVLLFNILCLFPVLAADNHGVSMRVSKDALPGCHTSNFM",
        "hemoglobin": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
        "too_short": "MKVL",
        "invalid_chars": "MKFL123ZZZ",
        "empty": "",
    }

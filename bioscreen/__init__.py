"""
BioScreen: Function-Aware DNA Synthesis Screening
Against AI-Designed Biological Threats

Production-grade screening using protein language models with
supervised contrastive learning for mechanism-of-harm detection.
"""

__version__ = "1.0.0"
__author__ = "AIxBio Hackathon Team"

from bioscreen.screening.screener import BioScreener
from bioscreen.screening.result import ScreeningResult

__all__ = ["BioScreener", "ScreeningResult"]

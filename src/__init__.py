"""
BRSR Sustainability Regime Analysis Package

This package provides tools for analyzing corporate sustainability disclosures
using Hidden Markov Models and Natural Language Processing techniques.
"""

__version__ = "1.0.0"
__author__ = "K R Hari Krishna"
__email__ = "your.email@example.com"

from .data_scraper import BRSRScraper
from .feature_extractor import FeatureExtractor
from .hmm_model import SustainabilityHMM
from .visualization import SustainabilityVisualizer

__all__ = [
    "BRSRScraper",
    "FeatureExtractor",
    "SustainabilityHMM",
    "SustainabilityVisualizer",
]
"""
Feature Extractor Module

This module handles NLP feature extraction from BRSR PDF text.
Extracts six theoretically motivated dimensions of sustainability disclosure.
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extractor for sustainability disclosure features from text.

    This class implements feature extraction for six dimensions:
    - Commitment: Aspirational language and targets
    - Metric: Quantitative indicators and measurements
    - Governance: Board and committee references
    - Capital: Investment and expenditure signaling
    - Enforcement: Audit and verification mechanisms
    - Supply Chain: Value chain due diligence
    """

    def __init__(self):
        """Initialize the feature extractor with precompiled regex patterns."""
        self.feature_patterns = self._compile_patterns()
        self.feature_names = ['Commitment', 'Metric', 'Governance', 'Capital', 'Enforcement', 'Supply']

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """
        Compile regex patterns for feature extraction.

        Returns:
            Dictionary of compiled regex patterns for each feature
        """
        patterns = {
            'commitment': re.compile(
                r'\b(target|commit|pledge|goal|objective|aspiration|aim|intent|promise)\b',
                re.I
            ),
            'metric': re.compile(
                r'\b(\d+%|percent|tonnes|mwh|kwh|kg|liter|yoy|growth|reduction|increase)\b',
                re.I
            ),
            'governance': re.compile(
                r'\b(board|committee|oversight|accountability|responsibility|governance|compliance|reporting)\b',
                re.I
            ),
            'capital': re.compile(
                r'\b(invest|capex|allocation|funding|expenditure|spend|budget|capital|finance)\b',
                re.I
            ),
            'enforcement': re.compile(
                r'\b(audit|monitor|verif|penalt|control|standard|assurance|certif)\b',
                re.I
            ),
            'supply': re.compile(
                r'\b(supplier|value\s+chain|scope|due\s+diligence|vendor|procurement)\b',
                re.I
            )
        }
        return patterns

    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract sustainability features from text.

        Args:
            text: Raw text content from BRSR PDF

        Returns:
            Numpy array of 6 normalized feature values
        """
        if not text or len(text.strip()) < 100:
            logger.warning("Text too short or empty for feature extraction")
            return np.zeros(6)

        word_count = len(text.split())
        if word_count == 0:
            return np.zeros(6)

        # Count matches for each pattern and normalize by word count
        features = []
        for pattern_name, pattern in self.feature_patterns.items():
            count = len(pattern.findall(text))
            normalized_count = count / word_count
            features.append(normalized_count)

        return np.array(features)

    def extract_features_batch(self, texts: List[str]) -> np.ndarray:
        """
        Extract features from multiple texts.

        Args:
            texts: List of text strings

        Returns:
            2D numpy array of shape (n_texts, 6)
        """
        features_list = []
        for text in texts:
            features = self.extract_features(text)
            features_list.append(features)

        return np.array(features_list)

    def get_feature_names(self) -> List[str]:
        """Get the names of extracted features."""
        return self.feature_names.copy()

    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of each feature dimension.

        Returns:
            Dictionary mapping feature names to descriptions
        """
        descriptions = {
            'Commitment': 'Prevalence of targets, pledges, and aspirational statements',
            'Metric': 'Frequency of quantified indicators, percentages, and year-on-year comparisons',
            'Governance': 'References to boards, committees, and accountability mechanisms',
            'Capital': 'Investment and capex mentions indicating resource commitment',
            'Enforcement': 'Audit, verification, and penalty references signaling external accountability',
            'Supply': 'Scope-3 due diligence and supplier standard mentions'
        }
        return descriptions

    def validate_features(self, features: np.ndarray) -> bool:
        """
        Validate extracted features.

        Args:
            features: Feature array to validate

        Returns:
            True if features are valid
        """
        if features.shape != (6,):
            logger.error(f"Invalid feature shape: {features.shape}, expected (6,)")
            return False

        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            logger.error("Features contain NaN or infinite values")
            return False

        if np.any(features < 0):
            logger.warning("Negative feature values detected")

        return True
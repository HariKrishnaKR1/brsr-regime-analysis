import re
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract sustainability features from BRSR text."""

    def __init__(self):
        self.feature_patterns = self._compile_patterns()
        self.feature_names = ['Commitment', 'Metric', 'Governance', 'Capital', 'Enforcement', 'Supply']

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for features."""
        return {
            'commitment': re.compile(
                r'\b(target|commit|pledge|goal|objective|aspiration|aim|intent|promise)\b', re.I
            ),
            'metric': re.compile(
                r'\b(\d+%|percent|tonnes|mwh|kwh|kg|liter|yoy|growth|reduction|increase)\b', re.I
            ),
            'governance': re.compile(
                r'\b(board|committee|oversight|accountability|responsibility|governance|compliance|reporting)\b', re.I
            ),
            'capital': re.compile(
                r'\b(invest|capex|allocation|funding|expenditure|spend|budget|capital|finance)\b', re.I
            ),
            'enforcement': re.compile(
                r'\b(audit|monitor|verif|penalt|control|standard|assurance|certif)\b', re.I
            ),
            'supply': re.compile(
                r'\b(supplier|value\s+chain|scope|due\s+diligence|vendor|procurement)\b', re.I
            )
        }

    def extract_features(self, text: str) -> np.ndarray:
        """Extract features from text."""
        if not text or len(text.strip()) < 100:
            return np.zeros(6)

        words = text.split()
        word_count = len(words)
        if word_count == 0:
            return np.zeros(6)

        # Count matches and normalize
        features = [len(pattern.findall(text)) / word_count
                   for pattern in self.feature_patterns.values()]
        return np.array(features)

    def extract_features_batch(self, texts: List[str]) -> np.ndarray:
        """Extract features from multiple texts."""
        return np.array([self.extract_features(text) for text in texts])

    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names.copy()

    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get feature descriptions."""
        return {
            'Commitment': 'Targets, pledges, and aspirational statements',
            'Metric': 'Quantified indicators and measurements',
            'Governance': 'Board and committee references',
            'Capital': 'Investment and expenditure signaling',
            'Enforcement': 'Audit and verification mechanisms',
            'Supply': 'Value chain due diligence'
        }

    def validate_features(self, features: np.ndarray) -> bool:
        """Validate features."""
        if features.shape != (6,):
            return False
        return not (np.any(np.isnan(features)) or np.any(np.isinf(features)) or np.any(features < 0))
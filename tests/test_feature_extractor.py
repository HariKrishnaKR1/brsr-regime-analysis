"""
Tests for Feature Extractor Module
"""

import pytest
import numpy as np
from src.feature_extractor import FeatureExtractor


class TestFeatureExtractor:
    """Test cases for feature extractor."""

    def test_extractor_initialization(self):
        """Test feature extractor initialization."""
        extractor = FeatureExtractor()
        assert len(extractor.feature_patterns) == 6
        assert len(extractor.feature_names) == 6
        assert extractor.feature_names == ['Commitment', 'Metric', 'Governance', 'Capital', 'Enforcement', 'Supply']

    def test_extract_features_empty_text(self):
        """Test feature extraction from empty text."""
        extractor = FeatureExtractor()
        features = extractor.extract_features("")
        assert isinstance(features, np.ndarray)
        assert features.shape == (6,)
        assert np.all(features == 0)

    def test_extract_features_short_text(self):
        """Test feature extraction from very short text."""
        extractor = FeatureExtractor()
        features = extractor.extract_features("hi")
        assert features.shape == (6,)
        assert np.all(features == 0)

    def test_extract_features_commitment(self):
        """Test commitment feature extraction."""
        extractor = FeatureExtractor()
        text = "We commit to target our goals and objectives for sustainability."
        features = extractor.extract_features(text)

        # Commitment feature should be > 0
        assert features[0] > 0  # Commitment is first feature
        # Other features should be 0 or very low
        assert all(features[i] <= features[0] for i in range(1, 6))

    def test_extract_features_metric(self):
        """Test metric feature extraction."""
        extractor = FeatureExtractor()
        text = "We achieved 25% reduction in emissions and 50 tonnes of waste."
        features = extractor.extract_features(text)

        # Metric feature should be > 0
        assert features[1] > 0  # Metric is second feature

    def test_extract_features_governance(self):
        """Test governance feature extraction."""
        extractor = FeatureExtractor()
        text = "Our board committee oversees governance and accountability."
        features = extractor.extract_features(text)

        # Governance feature should be > 0
        assert features[2] > 0  # Governance is third feature

    def test_extract_features_capital(self):
        """Test capital feature extraction."""
        extractor = FeatureExtractor()
        text = "We invested capital in green technology and funded sustainability projects."
        features = extractor.extract_features(text)

        # Capital feature should be > 0
        assert features[3] > 0  # Capital is fourth feature

    def test_extract_features_enforcement(self):
        """Test enforcement feature extraction."""
        extractor = FeatureExtractor()
        text = "We conduct audits and verify our compliance standards."
        features = extractor.extract_features(text)

        # Enforcement feature should be > 0
        assert features[4] > 0  # Enforcement is fifth feature

    def test_extract_features_supply_chain(self):
        """Test supply chain feature extraction."""
        extractor = FeatureExtractor()
        text = "We assess our suppliers and manage value chain due diligence."
        features = extractor.extract_features(text)

        # Supply chain feature should be > 0
        assert features[5] > 0  # Supply is sixth feature

    def test_extract_features_batch(self):
        """Test batch feature extraction."""
        extractor = FeatureExtractor()
        texts = [
            "We commit to target sustainability goals.",
            "We achieved 25% reduction in emissions.",
            ""
        ]

        features_batch = extractor.extract_features_batch(texts)

        assert isinstance(features_batch, np.ndarray)
        assert features_batch.shape == (3, 6)

        # First text should have commitment feature
        assert features_batch[0, 0] > 0
        # Second text should have metric feature
        assert features_batch[1, 1] > 0
        # Third text should be all zeros
        assert np.all(features_batch[2] == 0)

    def test_get_feature_names(self):
        """Test getting feature names."""
        extractor = FeatureExtractor()
        names = extractor.get_feature_names()
        assert isinstance(names, list)
        assert len(names) == 6
        assert names[0] == 'Commitment'

    def test_get_feature_descriptions(self):
        """Test getting feature descriptions."""
        extractor = FeatureExtractor()
        descriptions = extractor.get_feature_descriptions()
        assert isinstance(descriptions, dict)
        assert len(descriptions) == 6
        assert 'Commitment' in descriptions
        assert 'aspirational language' in descriptions['Commitment'].lower()

    def test_validate_features_valid(self):
        """Test validation of valid features."""
        extractor = FeatureExtractor()
        features = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        assert extractor.validate_features(features)

    def test_validate_features_wrong_shape(self):
        """Test validation of wrong shape features."""
        extractor = FeatureExtractor()
        features = np.array([0.1, 0.2, 0.3])  # Wrong shape
        assert not extractor.validate_features(features)

    def test_validate_features_with_nan(self):
        """Test validation of features with NaN."""
        extractor = FeatureExtractor()
        features = np.array([0.1, np.nan, 0.3, 0.4, 0.5, 0.6])
        assert not extractor.validate_features(features)

    def test_validate_features_with_inf(self):
        """Test validation of features with infinity."""
        extractor = FeatureExtractor()
        features = np.array([0.1, 0.2, np.inf, 0.4, 0.5, 0.6])
        assert not extractor.validate_features(features)

    def test_validate_features_negative(self):
        """Test validation of features with negative values (should still be valid)."""
        extractor = FeatureExtractor()
        features = np.array([-0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        # Negative values are allowed, just logged as warning
        assert extractor.validate_features(features)
"""
Tests for HMM Model Module
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.hmm_model import SustainabilityHMM


class TestSustainabilityHMM:
    """Test cases for sustainability HMM model."""

    def test_initialization_default(self):
        """Test HMM initialization with defaults."""
        model = SustainabilityHMM()
        assert model.n_states == 3
        assert model.random_state == 42
        assert model.model is None
        assert model.model_type is None
        assert len(model.state_labels) == 3
        assert len(model.feature_names) == 6

    def test_initialization_custom(self):
        """Test HMM initialization with custom parameters."""
        model = SustainabilityHMM(n_states=2, random_state=123)
        assert model.n_states == 2
        assert model.random_state == 123
        assert len(model.state_labels) == 2

    def test_state_labels_custom_n_states(self):
        """Test state labels for non-standard number of states."""
        model = SustainabilityHMM(n_states=5)
        expected_labels = ['Regime_0', 'Regime_1', 'Regime_2', 'Regime_3', 'Regime_4']
        assert model.state_labels == expected_labels

    @patch('src.hmm_model.hmm')
    def test_fit_hmm_success(self, mock_hmm):
        """Test successful HMM fitting."""
        # Mock the HMM model
        mock_model = Mock()
        mock_model.fit.return_value = None
        mock_model.score.return_value = -100.0
        mock_hmm.GaussianHMM.return_value = mock_model

        # Create test data
        test_data = pd.DataFrame({
            'company_name': ['A', 'A', 'B', 'B'],
            'financial_year': ['2022-23', '2023-24', '2022-23', '2023-24'],
            'feature1': [1.0, 2.0, 1.5, 2.5],
            'feature2': [0.5, 1.5, 1.0, 2.0]
        })

        model = SustainabilityHMM(n_states=2)
        result = model.fit(test_data)

        assert result is model
        assert model.model_type == "HMM"
        assert model.model is not None
        mock_hmm.GaussianHMM.assert_called_once()

    def test_fit_kmeans_fallback(self):
        """Test K-means fallback when insufficient time-series data."""
        # Create test data with only single time points
        test_data = pd.DataFrame({
            'company_name': ['A', 'B', 'C'],
            'financial_year': ['2022-23', '2022-23', '2022-23'],
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [0.5, 1.5, 2.5]
        })

        model = SustainabilityHMM(n_states=2)
        result = model.fit(test_data)

        assert result is model
        assert model.model_type == "K-means"

    def test_predict_without_fit(self):
        """Test prediction without fitting raises appropriate error."""
        model = SustainabilityHMM()
        test_data = pd.DataFrame({
            'company_name': ['A', 'B'],
            'financial_year': ['2022-23', '2022-23'],
            'feature1': [1.0, 2.0]
        })

        with pytest.raises(AttributeError):
            model.predict(test_data)

    def test_get_model_info_unfitted(self):
        """Test getting model info for unfitted model."""
        model = SustainabilityHMM()
        info = model.get_model_info()

        expected_keys = ['model_type', 'n_states', 'state_labels', 'feature_names']
        assert all(key in info for key in expected_keys)
        assert info['model_type'] is None
        assert info['n_states'] == 3

    @patch('src.hmm_model.hmm')
    def test_score_hmm(self, mock_hmm):
        """Test scoring for HMM model."""
        # Mock the HMM model
        mock_model = Mock()
        mock_model.score.return_value = -50.0
        mock_hmm.GaussianHMM.return_value = mock_model

        # Create test data and fit model
        test_data = pd.DataFrame({
            'company_name': ['A', 'A', 'B', 'B'],
            'financial_year': ['2022-23', '2023-24', '2022-23', '2023-24'],
            'feature1': [1.0, 2.0, 1.5, 2.5]
        })

        model = SustainabilityHMM(n_states=2)
        model.fit(test_data)

        score = model.score(test_data)
        assert score == -50.0
        mock_model.score.assert_called()

    def test_score_kmeans(self):
        """Test scoring for K-means model (should return 0)."""
        test_data = pd.DataFrame({
            'company_name': ['A', 'B'],
            'financial_year': ['2022-23', '2022-23'],
            'feature1': [1.0, 2.0]
        })

        model = SustainabilityHMM(n_states=2)
        model.fit(test_data)

        score = model.score(test_data)
        assert score == 0.0

    def test_predict_fallback(self):
        """Test fallback prediction."""
        model = SustainabilityHMM()
        model.model_type = "Fallback"

        test_data = pd.DataFrame({
            'company_name': ['A', 'B'],
            'financial_year': ['2022-23', '2022-23'],
            'feature1': [1.0, 2.0]
        })

        result = model.predict(test_data)

        assert len(result) == 2
        assert all(result['regime_id'] == 0)
        assert all(result['regime_label'] == 'Minimal')
        assert all(result['regime_prob'] == 0.5)
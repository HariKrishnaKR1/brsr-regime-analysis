"""
Hidden Markov Model Module for Sustainability Regime Inference

This module implements HMM-based inference of latent corporate sustainability regimes
from temporal disclosure feature sequences.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SustainabilityHMM:
    """
    Hidden Markov Model for sustainability regime inference.

    This class implements regime inference using HMMs for temporal data
    and falls back to K-means clustering for cross-sectional analysis.
    """

    def __init__(self, n_states: int = 3, random_state: int = 42):
        """
        Initialize the sustainability HMM.

        Args:
            n_states: Number of latent regimes to infer
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.random_state = random_state
        self.model = None
        self.model_type = None
        self.scaler = StandardScaler()
        self.state_labels = ['Minimal', 'Diluted', 'Enforcement']
        self.feature_names = ['Commitment', 'Metric', 'Governance', 'Capital', 'Enforcement', 'Supply']

        # Validate state labels
        if len(self.state_labels) != n_states:
            self.state_labels = [f'Regime_{i}' for i in range(n_states)]

    def fit(self, df_features: pd.DataFrame,
            company_col: str = 'company_name',
            time_col: str = 'financial_year',
            feature_cols: Optional[List[str]] = None) -> 'SustainabilityHMM':
        """
        Fit the HMM or clustering model to the data.

        Args:
            df_features: DataFrame with features and metadata
            company_col: Column name for company identifier
            time_col: Column name for time period
            feature_cols: List of feature column names (default: all numeric except metadata)

        Returns:
            Self for method chaining
        """
        if feature_cols is None:
            # Auto-detect feature columns (numeric columns except metadata)
            exclude_cols = {company_col, time_col}
            feature_cols = [col for col in df_features.columns
                          if col not in exclude_cols and df_features[col].dtype in ['float64', 'int64']]

        logger.info(f"Using features: {feature_cols}")

        # Check for time-series data
        companies_multi = df_features.groupby(company_col)[time_col].nunique()
        time_series_companies = companies_multi[companies_multi > 1].index.tolist()

        if len(time_series_companies) > 2:
            logger.info(f"Using HMM for {len(time_series_companies)} time-series companies")
            self._fit_hmm(df_features, time_series_companies, company_col, time_col, feature_cols)
        else:
            logger.info("Using K-means clustering for cross-sectional analysis")
            self._fit_kmeans(df_features, feature_cols)

        return self

    def _fit_hmm(self, df: pd.DataFrame, time_series_companies: List[str],
                 company_col: str, time_col: str, feature_cols: List[str]):
        """Fit Hidden Markov Model for temporal data."""
        observations_list = []
        lengths = []
        firms_list = []

        for company in sorted(time_series_companies):
            company_data = df[df[company_col] == company].sort_values(time_col)
            obs = company_data[feature_cols].values
            observations_list.append(obs)
            lengths.append(len(obs))
            firms_list.append(company)

        X_all = np.vstack(observations_list)

        # Fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type='diag',
            n_iter=500,
            random_state=self.random_state
        )
        self.model.fit(X_all, lengths)
        self.model_type = "HMM"

        logger.info(f"HMM fitted with {self.n_states} states")

    def _fit_kmeans(self, df: pd.DataFrame, feature_cols: List[str]):
        """Fit K-means clustering for cross-sectional data."""
        X = df[feature_cols].values

        try:
            kmeans = KMeans(
                n_clusters=self.n_states,
                random_state=self.random_state,
                n_init=10
            )
            clusters = kmeans.fit_predict(X)

            # Map clusters to regimes based on overall intensity
            intensity = kmeans.cluster_centers_.sum(axis=1)
            cluster_to_regime = {
                np.argsort(intensity)[i]: i for i in range(self.n_states)
            }

            # Store cluster centers for later use
            self.cluster_centers_ = kmeans.cluster_centers_
            self.cluster_to_regime = cluster_to_regime
            self.model_type = "K-means"

            logger.info("K-means clustering completed")

        except Exception as e:
            logger.error(f"K-means failed: {e}. Using fallback assignment.")
            self.model_type = "Fallback"

    def predict(self, df_features: pd.DataFrame,
               company_col: str = 'company_name',
               time_col: str = 'financial_year',
               feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Predict regimes for the input data.

        Args:
            df_features: DataFrame with features and metadata
            company_col: Column name for company identifier
            time_col: Column name for time period
            feature_cols: List of feature column names

        Returns:
            DataFrame with regime predictions
        """
        if feature_cols is None:
            exclude_cols = {company_col, time_col}
            feature_cols = [col for col in df_features.columns
                          if col not in exclude_cols and df_features[col].dtype in ['float64', 'int64']]

        if self.model_type == "HMM":
            return self._predict_hmm(df_features, company_col, time_col, feature_cols)
        elif self.model_type == "K-means":
            return self._predict_kmeans(df_features, company_col, time_col, feature_cols)
        else:
            return self._predict_fallback(df_features, company_col, time_col)

    def _predict_hmm(self, df: pd.DataFrame, company_col: str,
                    time_col: str, feature_cols: List[str]) -> pd.DataFrame:
        """Predict regimes using fitted HMM."""
        regimes = []

        companies_multi = df.groupby(company_col)[time_col].nunique()
        time_series_companies = companies_multi[companies_multi > 1].index.tolist()

        for company in time_series_companies:
            company_data = df[df[company_col] == company].sort_values(time_col)
            obs = company_data[feature_cols].values

            # Viterbi decoding for most likely state sequence
            _, states = self.model.decode(obs, algorithm='viterbi')
            probs = self.model.predict_proba(obs)

            for i, (_, row) in enumerate(company_data.iterrows()):
                regimes.append({
                    company_col: company,
                    time_col: row[time_col],
                    'regime_id': int(states[i]),
                    'regime_label': self.state_labels[int(states[i])],
                    'regime_prob': float(probs[i].max())
                })

        return pd.DataFrame(regimes)

    def _predict_kmeans(self, df: pd.DataFrame, company_col: str,
                       time_col: str, feature_cols: List[str]) -> pd.DataFrame:
        """Predict regimes using K-means clustering."""
        X = df[feature_cols].values
        distances = self.cluster_centers_[np.arange(len(X)), :]

        # This is a simplified version - in practice you'd need to compute proper distances
        # For now, assign based on closest cluster
        clusters = np.argmin(np.sum((X[:, np.newaxis] - self.cluster_centers_) ** 2, axis=2), axis=1)
        regime_ids = [self.cluster_to_regime.get(c, 0) for c in clusters]

        # Calculate confidence as inverse of distance to cluster center
        distances_to_centers = np.sqrt(np.sum((X - self.cluster_centers_[clusters]) ** 2, axis=1))
        max_distance = np.max(distances_to_centers) + 1e-8
        regime_probs = 1 - (distances_to_centers / max_distance)

        df_result = df[[company_col, time_col]].copy()
        df_result['regime_id'] = regime_ids
        df_result['regime_label'] = [self.state_labels[rid] for rid in regime_ids]
        df_result['regime_prob'] = regime_probs

        return df_result

    def _predict_fallback(self, df: pd.DataFrame, company_col: str, time_col: str) -> pd.DataFrame:
        """Fallback prediction when model fitting failed."""
        df_result = df[[company_col, time_col]].copy()
        df_result['regime_id'] = 0
        df_result['regime_label'] = self.state_labels[0]
        df_result['regime_prob'] = 0.5  # Low confidence

        return df_result

    def get_model_info(self) -> Dict:
        """Get information about the fitted model."""
        info = {
            'model_type': self.model_type,
            'n_states': self.n_states,
            'state_labels': self.state_labels,
            'feature_names': self.feature_names
        }

        if self.model_type == "HMM" and self.model is not None:
            info.update({
                'transition_matrix': self.model.transmat_,
                'means': self.model.means_,
                'covariances': self.model.covars_
            })

        return info

    def score(self, df_features: pd.DataFrame,
             company_col: str = 'company_name',
             time_col: str = 'financial_year',
             feature_cols: Optional[List[str]] = None) -> float:
        """
        Calculate log-likelihood score for the fitted model.

        Args:
            df_features: DataFrame with features
            company_col: Company column name
            time_col: Time column name
            feature_cols: Feature column names

        Returns:
            Log-likelihood score
        """
        if self.model_type != "HMM" or self.model is None:
            logger.warning("Scoring only available for HMM models")
            return 0.0

        if feature_cols is None:
            exclude_cols = {company_col, time_col}
            feature_cols = [col for col in df_features.columns
                          if col not in exclude_cols and df_features[col].dtype in ['float64', 'int64']]

        # Prepare data in HMM format
        companies_multi = df_features.groupby(company_col)[time_col].nunique()
        time_series_companies = companies_multi[companies_multi > 1].index.tolist()

        if not time_series_companies:
            return 0.0

        observations_list = []
        lengths = []

        for company in sorted(time_series_companies):
            company_data = df_features[df_features[company_col] == company].sort_values(time_col)
            obs = company_data[feature_cols].values
            observations_list.append(obs)
            lengths.append(len(obs))

        X_all = np.vstack(observations_list)
        return self.model.score(X_all, lengths)
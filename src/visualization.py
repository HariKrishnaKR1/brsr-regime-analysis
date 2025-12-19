"""
Visualization Module for Sustainability Regime Analysis

This module provides plotting and visualization utilities for
sustainability regime analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


class SustainabilityVisualizer:
    """
    Visualization utilities for sustainability regime analysis.

    Provides methods for plotting regime distributions, temporal trends,
    feature importance, and model diagnostics.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.

        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']

    def plot_regime_distribution(self, df_regimes: pd.DataFrame,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the distribution of inferred regimes.

        Args:
            df_regimes: DataFrame with regime predictions
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        regime_counts = df_regimes['regime_label'].value_counts()

        bars = ax.bar(range(len(regime_counts)), regime_counts.values,
                     color=self.colors[:len(regime_counts)])

        ax.set_title('Sustainability Regime Distribution', fontweight='bold', fontsize=14)
        ax.set_xlabel('Regime', fontsize=12)
        ax.set_ylabel('Number of Observations', fontsize=12)
        ax.set_xticks(range(len(regime_counts)))
        ax.set_xticklabels(regime_counts.index, rotation=45, ha='right')

        # Add value labels on bars
        for bar, count in zip(bars, regime_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved regime distribution plot to {save_path}")

        return fig

    def plot_regime_temporal_trends(self, df_regimes: pd.DataFrame,
                                   time_col: str = 'financial_year',
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot regime distribution over time.

        Args:
            df_regimes: DataFrame with regime predictions and time data
            time_col: Column name for time periods
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        if time_col not in df_regimes.columns:
            logger.warning(f"Time column '{time_col}' not found. Skipping temporal plot.")
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        # Group by time period and regime
        regime_time = df_regimes.groupby([time_col, 'regime_label']).size().unstack(fill_value=0)

        # Sort by time period if possible
        try:
            regime_time = regime_time.sort_index()
        except:
            pass

        regime_time.plot(kind='bar', ax=ax, color=self.colors[:len(regime_time.columns)])

        ax.set_title('Regime Distribution by Time Period', fontweight='bold', fontsize=14)
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Number of Observations', fontsize=12)
        ax.legend(title='Regime', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved temporal trends plot to {save_path}")

        return fig

    def plot_feature_importance(self, feature_importance: pd.DataFrame,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance from ablation analysis.

        Args:
            feature_importance: DataFrame with feature importance scores
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        if 'feature' not in feature_importance.columns or 'importance_%' not in feature_importance.columns:
            logger.error("Feature importance DataFrame must have 'feature' and 'importance_%' columns")
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        df_sorted = feature_importance.sort_values('importance_%')

        bars = ax.barh(df_sorted['feature'], df_sorted['importance_%'],
                      color=self.colors[0])

        ax.set_title('Feature Importance (Ablation Analysis)', fontweight='bold', fontsize=14)
        ax.set_xlabel('Importance (%)', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)

        # Add value labels
        for bar, importance in zip(bars, df_sorted['importance_%']):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{importance:.1f}%', ha='left', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")

        return fig

    def plot_emission_distributions(self, model_info: Dict,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot emission distributions for HMM states.

        Args:
            model_info: Dictionary with model information from HMM
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        if 'means' not in model_info or 'covariances' not in model_info:
            logger.warning("Model info does not contain emission parameters")
            return None

        means = model_info['means']
        covars = model_info['covariances']
        feature_names = model_info.get('feature_names', [f'Feature_{i}' for i in range(len(means[0]))])

        n_states = len(means)
        n_features = len(feature_names)

        fig, axes = plt.subplots(n_features, 1, figsize=(self.figsize[0], self.figsize[1] * n_features / 3))
        if n_features == 1:
            axes = [axes]

        for i, feature in enumerate(feature_names):
            ax = axes[i]

            for state in range(n_states):
                mean = means[state][i]
                std = np.sqrt(covars[state][i])

                # Plot normal distribution
                x = np.linspace(mean - 3*std, mean + 3*std, 100)
                y = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

                ax.plot(x, y, label=f'State {state} ({model_info["state_labels"][state]})',
                       color=self.colors[state % len(self.colors)])
                ax.axvline(mean, linestyle='--', alpha=0.7, color=self.colors[state % len(self.colors)])

            ax.set_title(f'Emission Distribution: {feature}', fontsize=12)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved emission distributions plot to {save_path}")

        return fig

    def plot_transition_matrix(self, model_info: Dict,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the HMM transition matrix as a heatmap.

        Args:
            model_info: Dictionary with model information
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        if 'transition_matrix' not in model_info:
            logger.warning("Model info does not contain transition matrix")
            return None

        transmat = model_info['transition_matrix']
        state_labels = model_info.get('state_labels', [f'State_{i}' for i in range(len(transmat))])

        fig, ax = plt.subplots(figsize=self.figsize)

        sns.heatmap(transmat, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=state_labels, yticklabels=state_labels, ax=ax)

        ax.set_title('HMM Transition Matrix', fontweight='bold', fontsize=14)
        ax.set_xlabel('To State')
        ax.set_ylabel('From State')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved transition matrix plot to {save_path}")

        return fig

    def create_comprehensive_dashboard(self, df_regimes: pd.DataFrame,
                                    model_info: Dict,
                                    feature_importance: Optional[pd.DataFrame] = None,
                                    time_col: str = 'financial_year',
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple plots.

        Args:
            df_regimes: DataFrame with regime predictions
            model_info: Dictionary with model information
            feature_importance: Optional DataFrame with feature importance
            time_col: Column name for time periods
            save_path: Optional path to save the dashboard

        Returns:
            Matplotlib figure object
        """
        n_plots = 2  # Base plots: distribution and temporal
        if feature_importance is not None:
            n_plots += 1
        if 'transition_matrix' in model_info:
            n_plots += 1

        fig, axes = plt.subplots(n_plots, 1, figsize=(self.figsize[0], self.figsize[1] * n_plots))

        if n_plots == 1:
            axes = [axes]

        plot_idx = 0

        # Regime distribution
        regime_counts = df_regimes['regime_label'].value_counts()
        axes[plot_idx].bar(range(len(regime_counts)), regime_counts.values,
                          color=self.colors[:len(regime_counts)])
        axes[plot_idx].set_title('Regime Distribution', fontweight='bold')
        axes[plot_idx].set_xticks(range(len(regime_counts)))
        axes[plot_idx].set_xticklabels(regime_counts.index, rotation=45, ha='right')
        plot_idx += 1

        # Temporal trends
        if time_col in df_regimes.columns:
            regime_time = df_regimes.groupby([time_col, 'regime_label']).size().unstack(fill_value=0)
            try:
                regime_time = regime_time.sort_index()
            except:
                pass

            regime_time.plot(kind='bar', ax=axes[plot_idx],
                           color=self.colors[:len(regime_time.columns)])
            axes[plot_idx].set_title('Regime Distribution by Time Period', fontweight='bold')
            axes[plot_idx].tick_params(axis='x', rotation=45)
            plot_idx += 1

        # Feature importance
        if feature_importance is not None:
            df_sorted = feature_importance.sort_values('importance_%')
            axes[plot_idx].barh(df_sorted['feature'], df_sorted['importance_%'],
                              color=self.colors[0])
            axes[plot_idx].set_title('Feature Importance', fontweight='bold')
            plot_idx += 1

        # Transition matrix
        if 'transition_matrix' in model_info:
            transmat = model_info['transition_matrix']
            state_labels = model_info.get('state_labels', [f'State_{i}' for i in range(len(transmat))])
            sns.heatmap(transmat, annot=True, fmt='.3f', cmap='Blues',
                       xticklabels=state_labels, yticklabels=state_labels,
                       ax=axes[plot_idx])
            axes[plot_idx].set_title('Transition Matrix', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comprehensive dashboard to {save_path}")

        return fig
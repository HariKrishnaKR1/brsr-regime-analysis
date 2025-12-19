#!/usr/bin/env python3
"""
Main Script for BRSR Sustainability Regime Analysis

This script orchestrates the complete analysis pipeline:
1. Data scraping from NSE India
2. Feature extraction from BRSR PDFs
3. HMM-based regime inference
4. Results visualization and analysis
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data_scraper import BRSRScraper
from src.feature_extractor import FeatureExtractor
from src.hmm_model import SustainabilityHMM
from src.visualization import SustainabilityVisualizer
from src.utils import (
    process_pdf_batch, normalize_features_within_firms,
    perform_ablation_analysis, save_results_to_csv,
    setup_logging, validate_dataframe
)

logger = logging.getLogger(__name__)


def run_data_collection(html_file_path: str, output_dir: str = 'data/processed') -> pd.DataFrame:
    """
    Run the data collection phase.

    Args:
        html_file_path: Path to HTML file with BRSR data
        output_dir: Directory to save processed data

    Returns:
        DataFrame with BRSR metadata
    """
    logger.info("Starting data collection phase...")

    # Initialize scraper
    scraper = BRSRScraper(html_file_path)

    # Scrape BRSR data
    df_brsr = scraper.scrape_brsr()

    if df_brsr.empty:
        logger.error("No BRSR data collected. Exiting.")
        return pd.DataFrame()

    # Validate data
    stats = scraper.validate_data(df_brsr)
    logger.info(f"Data collection complete: {stats}")

    # Save raw metadata
    save_results_to_csv(df_brsr, 'brsr_metadata', output_dir)

    return df_brsr


def run_feature_extraction(df_brsr: pd.DataFrame, output_dir: str = 'data/processed') -> pd.DataFrame:
    """
    Run the feature extraction phase.

    Args:
        df_brsr: DataFrame with BRSR metadata
        output_dir: Directory to save processed data

    Returns:
        DataFrame with extracted features
    """
    logger.info("Starting feature extraction phase...")

    # Filter to PDFs with valid URLs
    valid_pdfs = df_brsr[df_brsr['pdf_url'].str.len() > 0].copy()

    if valid_pdfs.empty:
        logger.error("No valid PDF URLs found. Exiting.")
        return pd.DataFrame()

    # Process PDFs in batch
    pdf_results = process_pdf_batch(
        valid_pdfs['pdf_url'].tolist(),
        valid_pdfs['company_name'].tolist()
    )

    # Initialize feature extractor
    extractor = FeatureExtractor()

    # Extract features from texts
    texts = pdf_results['text'].tolist()
    features_array = extractor.extract_features_batch(texts)

    # Create feature DataFrame
    feature_cols = extractor.get_feature_names()
    df_features = pd.DataFrame(features_array, columns=feature_cols)

    # Combine with metadata
    df_combined = pd.concat([
        pdf_results[['company_name', 'pdf_url', 'status']],
        valid_pdfs[['financial_year', 'year_of_declaration']].reset_index(drop=True),
        df_features
    ], axis=1)

    # Filter successful extractions
    df_success = df_combined[df_combined['status'] == 'success'].copy()

    # Normalize features
    df_normalized = normalize_features_within_firms(
        df_success,
        feature_cols,
        company_col='company_name'
    )

    logger.info(f"Feature extraction complete: {len(df_normalized)} successful extractions")

    # Save results
    save_results_to_csv(df_normalized, 'extracted_features', output_dir)

    return df_normalized


def run_regime_inference(df_features: pd.DataFrame, output_dir: str = 'results') -> Tuple[pd.DataFrame, Dict]:
    """
    Run the regime inference phase.

    Args:
        df_features: DataFrame with normalized features
        output_dir: Directory to save results

    Returns:
        Tuple of (regime predictions DataFrame, model info dict)
    """
    logger.info("Starting regime inference phase...")

    # Initialize HMM model
    hmm_model = SustainabilityHMM(n_states=3)

    # Fit model
    hmm_model.fit(df_features)

    # Predict regimes
    df_regimes = hmm_model.predict(df_features)

    # Get model information
    model_info = hmm_model.get_model_info()

    logger.info(f"Regime inference complete: {model_info['model_type']} model with {model_info['n_states']} states")

    # Save results
    save_results_to_csv(df_regimes, 'regime_predictions', output_dir)

    return df_regimes, model_info


def run_analysis_and_visualization(df_regimes: pd.DataFrame, model_info: Dict,
                                 df_features: pd.DataFrame, output_dir: str = 'results'):
    """
    Run analysis and create visualizations.

    Args:
        df_regimes: DataFrame with regime predictions
        model_info: Dictionary with model information
        df_features: DataFrame with features
        output_dir: Directory to save results and plots
    """
    logger.info("Starting analysis and visualization phase...")

    # Initialize visualizer
    viz = SustainabilityVisualizer()

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Plot regime distribution
    fig_dist = viz.plot_regime_distribution(df_regimes)
    if fig_dist:
        fig_dist.savefig(Path(output_dir) / 'regime_distribution.png', dpi=300, bbox_inches='tight')

    # Plot temporal trends
    fig_temporal = viz.plot_regime_temporal_trends(df_regimes)
    if fig_temporal:
        fig_temporal.savefig(Path(output_dir) / 'regime_temporal_trends.png', dpi=300, bbox_inches='tight')

    # Plot emission distributions (if HMM)
    if model_info['model_type'] == 'HMM':
        fig_emissions = viz.plot_emission_distributions(model_info)
        if fig_emissions:
            fig_emissions.savefig(Path(output_dir) / 'emission_distributions.png', dpi=300, bbox_inches='tight')

        # Plot transition matrix
        fig_transitions = viz.plot_transition_matrix(model_info)
        if fig_transitions:
            fig_transitions.savefig(Path(output_dir) / 'transition_matrix.png', dpi=300, bbox_inches='tight')

    # Perform ablation analysis if possible
    if model_info['model_type'] == 'HMM' and len(df_features) > 0:
        try:
            # Prepare data for ablation
            companies_multi = df_features.groupby('company_name')['financial_year'].nunique()
            time_series_companies = companies_multi[companies_multi > 1].index.tolist()

            if time_series_companies:
                observations_list = []
                lengths = []

                for company in sorted(time_series_companies):
                    company_data = df_features[df_features['company_name'] == company].sort_values('financial_year')
                    obs = company_data[['Commitment', 'Metric', 'Governance', 'Capital', 'Enforcement', 'Supply']].values
                    observations_list.append(obs)
                    lengths.append(len(obs))

                X_all = np.vstack(observations_list)

                # Perform ablation
                df_ablation = perform_ablation_analysis(
                    model_info.get('model'), X_all, lengths,
                    ['Commitment', 'Metric', 'Governance', 'Capital', 'Enforcement', 'Supply']
                )

                # Save ablation results
                save_results_to_csv(df_ablation, 'feature_importance', output_dir)

                # Plot feature importance
                fig_importance = viz.plot_feature_importance(df_ablation)
                if fig_importance:
                    fig_importance.savefig(Path(output_dir) / 'feature_importance.png', dpi=300, bbox_inches='tight')

        except Exception as e:
            logger.warning(f"Could not perform ablation analysis: {e}")

    # Create comprehensive dashboard
    fig_dashboard = viz.create_comprehensive_dashboard(df_regimes, model_info)
    if fig_dashboard:
        fig_dashboard.savefig(Path(output_dir) / 'analysis_dashboard.png', dpi=300, bbox_inches='tight')

    logger.info("Analysis and visualization complete")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='BRSR Sustainability Regime Analysis')
    parser.add_argument('--html-file', type=str, required=True,
                       help='Path to HTML file containing BRSR data')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--skip-data-collection', action='store_true',
                       help='Skip data collection phase')
    parser.add_argument('--skip-feature-extraction', action='store_true',
                       help='Skip feature extraction phase')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    logger.info("Starting BRSR Sustainability Regime Analysis")
    logger.info(f"HTML file: {args.html_file}")
    logger.info(f"Output directory: {args.output_dir}")

    try:
        # Phase 1: Data Collection
        if not args.skip_data_collection:
            df_brsr = run_data_collection(args.html_file, 'data/processed')
            if df_brsr.empty:
                return 1
        else:
            # Load existing data
            data_path = Path('data/processed/brsr_metadata.csv')
            if data_path.exists():
                df_brsr = pd.read_csv(data_path)
                logger.info(f"Loaded existing BRSR data: {len(df_brsr)} records")
            else:
                logger.error("No existing BRSR data found. Run without --skip-data-collection")
                return 1

        # Phase 2: Feature Extraction
        if not args.skip_feature_extraction:
            df_features = run_feature_extraction(df_brsr, 'data/processed')
            if df_features.empty:
                return 1
        else:
            # Load existing features
            features_path = Path('data/processed/extracted_features.csv')
            if features_path.exists():
                df_features = pd.read_csv(features_path)
                logger.info(f"Loaded existing features: {len(df_features)} records")
            else:
                logger.error("No existing features found. Run without --skip-feature-extraction")
                return 1

        # Phase 3: Regime Inference
        df_regimes, model_info = run_regime_inference(df_features, args.output_dir)

        # Phase 4: Analysis and Visualization
        run_analysis_and_visualization(df_regimes, model_info, df_features, args.output_dir)

        logger.info("Analysis pipeline completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")

        return 0

    except Exception as e:
        logger.error(f"Analysis pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
"""
Utility Functions for BRSR Sustainability Analysis

This module contains helper functions for data processing, normalization,
and analysis utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

logger = logging.getLogger(__name__)


def extract_text_from_pdf_memory(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes using PyMuPDF.

    Args:
        pdf_bytes: PDF content as bytes

    Returns:
        Extracted text content
    """
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in pdf)
        pdf.close()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""


def normalize_features_within_firms(df: pd.DataFrame,
                                  feature_cols: List[str],
                                  company_col: str = 'company_name') -> pd.DataFrame:
    """
    Normalize features within firms (z-score) while preserving temporal dynamics.

    Args:
        df: DataFrame with features and company identifiers
        feature_cols: List of feature column names to normalize
        company_col: Column name for company identifier

    Returns:
        DataFrame with normalized features
    """
    df_normalized = df.copy()

    # Get companies with multiple time periods
    companies_multi = df.groupby(company_col)['financial_year'].nunique()
    time_series_companies = companies_multi[companies_multi > 1].index.tolist()

    logger.info(f"Normalizing features for {len(time_series_companies)} multi-period companies")

    # Within-firm z-score normalization
    for company in time_series_companies:
        mask = df_normalized[company_col] == company
        for col in feature_cols:
            if col in df_normalized.columns:
                values = df_normalized.loc[mask, col]
                if values.std() > 1e-8:  # Avoid division by zero
                    df_normalized.loc[mask, col] = (values - values.mean()) / values.std()

    # Global normalization for remaining features
    for col in feature_cols:
        if col in df_normalized.columns:
            values = df_normalized[col]
            if values.std() > 1e-8:
                df_normalized[col] = (values - values.mean()) / values.std()

    return df_normalized


def process_pdf_batch(pdf_urls: List[str], company_names: List[str],
                     max_workers: int = 4) -> pd.DataFrame:
    """
    Process a batch of PDFs in parallel.

    Args:
        pdf_urls: List of PDF URLs to download and process
        company_names: Corresponding list of company names
        max_workers: Maximum number of parallel workers

    Returns:
        DataFrame with processed results
    """
    results = []

    def process_single_pdf(idx: int, url: str, company: str) -> Dict:
        """Process a single PDF."""
        try:
            session = requests.Session()
            session.headers.update({"User-Agent": "Mozilla/5.0"})

            resp = session.get(url, timeout=30, allow_redirects=True)
            resp.raise_for_status()

            text = extract_text_from_pdf_memory(resp.content)

            return {
                'company_name': company,
                'pdf_url': url,
                'text': text,
                'status': 'success' if text else 'no_text_extracted'
            }

        except Exception as e:
            return {
                'company_name': company,
                'pdf_url': url,
                'text': '',
                'status': f'error: {str(e)[:50]}'
            }

    logger.info(f"Processing {len(pdf_urls)} PDFs with {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_pdf, i, url, company): i
                  for i, (url, company) in enumerate(zip(pdf_urls, company_names))}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            status_icon = "✓" if result.get('status') == 'success' else "✗"
            logger.info(f"{status_icon} {result['company_name']} - {result.get('status', 'unknown')}")

    return pd.DataFrame(results)


def calculate_model_metrics(model, X: np.ndarray, lengths: List[int]) -> Dict[str, float]:
    """
    Calculate model evaluation metrics.

    Args:
        model: Fitted HMM model
        X: Observation sequences
        lengths: Lengths of sequences

    Returns:
        Dictionary with evaluation metrics
    """
    try:
        log_likelihood = model.score(X, lengths)
        n_params = model.n_components * model.n_components + model.n_components * X.shape[1]
        n_samples = len(X)

        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(n_samples)

        return {
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'n_parameters': n_params,
            'n_samples': n_samples
        }
    except Exception as e:
        logger.error(f"Error calculating model metrics: {e}")
        return {}


def perform_ablation_analysis(model, X: np.ndarray, lengths: List[int],
                            feature_names: List[str]) -> pd.DataFrame:
    """
    Perform feature ablation analysis to determine importance.

    Args:
        model: Fitted baseline HMM model
        X: Full observation sequences
        lengths: Sequence lengths
        feature_names: Names of features

    Returns:
        DataFrame with ablation results
    """
    baseline_ll = model.score(X, lengths)
    ablation_results = []

    logger.info("Performing feature ablation analysis...")

    for i, feat in enumerate(feature_names):
        X_abl = np.delete(X, i, axis=1)

        try:
            m_abl = model.__class__(
                n_components=model.n_components,
                covariance_type=model.covariance_type,
                n_iter=500,
                random_state=42
            )
            m_abl.fit(X_abl, lengths)
            ll_abl = m_abl.score(X_abl, lengths)
            ll_drop = baseline_ll - ll_abl
            importance_pct = (ll_drop / abs(baseline_ll) * 100) if baseline_ll != 0 else 0

            ablation_results.append({
                'feature': feat,
                'importance_%': importance_pct,
                'll_drop': ll_drop
            })

            logger.info(f"✓ {feat:15s}: importance={importance_pct:5.2f}%")

        except Exception as e:
            logger.error(f"Error in ablation for {feat}: {e}")
            ablation_results.append({
                'feature': feat,
                'importance_%': 0.0,
                'll_drop': 0.0
            })

    return pd.DataFrame(ablation_results).sort_values('importance_%', ascending=False)


def validate_dataframe(df: pd.DataFrame, required_cols: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_cols: List of required column names

    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False, missing_cols

    return True, []


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def save_results_to_csv(df: pd.DataFrame, filename: str, output_dir: str = 'results') -> str:
    """
    Save DataFrame results to CSV file.

    Args:
        df: DataFrame to save
        filename: Filename without extension
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    output_path = Path(output_dir) / f"{filename}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")

    return str(output_path)
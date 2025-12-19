import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import fitz
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

logger = logging.getLogger(__name__)


def extract_text_from_pdf_memory(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in pdf)
        pdf.close()
        return text
    except Exception as e:
        return ""


def normalize_features_within_firms(df: pd.DataFrame,
                                  feature_cols: List[str],
                                  company_col: str = 'company_name') -> pd.DataFrame:
    """Normalize features within firms."""
    df_normalized = df.copy()

    companies_multi = df.groupby(company_col)['financial_year'].nunique()
    time_series_companies = companies_multi[companies_multi > 1].index.tolist()

    for company in time_series_companies:
        mask = df_normalized[company_col] == company
        for col in feature_cols:
            if col in df_normalized.columns:
                values = df_normalized.loc[mask, col]
                if values.std() > 1e-8:
                    df_normalized.loc[mask, col] = (values - values.mean()) / values.std()

    for col in feature_cols:
        if col in df_normalized.columns:
            values = df_normalized[col]
            if values.std() > 1e-8:
                df_normalized[col] = (values - values.mean()) / values.std()

    return df_normalized


def process_pdf_batch(pdf_urls: List[str], company_names: List[str],
                     max_workers: int = 4) -> pd.DataFrame:
    """Process batch of PDFs in parallel."""
    results = []

    def process_single_pdf(idx: int, url: str, company: str) -> Dict:
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

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_pdf, i, url, company): i
                  for i, (url, company) in enumerate(zip(pdf_urls, company_names))}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    return pd.DataFrame(results)


def calculate_model_metrics(model, X: np.ndarray, lengths: List[int]) -> Dict[str, float]:
    """Calculate model evaluation metrics."""
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
        return {}


def perform_ablation_analysis(model, X: np.ndarray, lengths: List[int],
                            feature_names: List[str]) -> pd.DataFrame:
    """Perform feature ablation analysis."""
    baseline_ll = model.score(X, lengths)
    ablation_results = []

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

        except Exception as e:
            ablation_results.append({
                'feature': feat,
                'importance_%': 0.0,
                'll_drop': 0.0
            })

    return pd.DataFrame(ablation_results).sort_values('importance_%', ascending=False)


def validate_dataframe(df: pd.DataFrame, required_cols: List[str]) -> Tuple[bool, List[str]]:
    """Validate DataFrame has required columns."""
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        return False, missing_cols

    return True, []


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """Set up logging configuration."""
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
    """Save DataFrame to CSV."""
    output_path = Path(output_dir) / f"{filename}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    return str(output_path)
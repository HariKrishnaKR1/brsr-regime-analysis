"""
BRSR Data Scraper Module

Handles web scraping of BRSR data from NSE India portal.
"""

import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BRSRScraper:
    """Scraper for BRSR data from NSE India portal."""

    def __init__(self, html_file_path: Optional[str] = None):
        self.html_file_path = html_file_path

    def scrape_brsr_data(self) -> pd.DataFrame:
        """Scrape BRSR data from NSE portal or local HTML file."""
        if self.html_file_path and Path(self.html_file_path).exists():
            return self._parse_local_html(self.html_file_path)
        else:
            logger.warning("No local HTML file provided")
            return pd.DataFrame()

    def _parse_local_html(self, file_path: str) -> pd.DataFrame:
        """Parse BRSR data from local HTML file."""
        logger.info(f"Parsing BRSR data from {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        data = []

        # Find all BRSR tables (they have specific ID pattern)
        tables = soup.find_all('table', id=lambda x: x and 'CFBussinessSustainabilitiyTable' in x)

        for table in tables:
            tbody = table.find('tbody')
            if not tbody:
                continue

            for row in tbody.find_all('tr'):
                # Skip rows without proper data cells
                tds = row.find_all('td')
                if len(tds) < 7:
                    continue

                try:
                    # Extract company name (more robust parsing)
                    company_cell = row.find('td', headers='companyName')
                    if not company_cell:
                        # Fallback to positional extraction
                        company_cell = tds[0] if len(tds) > 0 else None

                    if not company_cell:
                        continue

                    company_name = company_cell.get_text(strip=True)
                    if not company_name:
                        continue

                    # Extract financial years
                    fy_from_cell = row.find('td', headers='fyFrom') or tds[1] if len(tds) > 1 else None
                    fy_to_cell = row.find('td', headers='fyTo') or tds[2] if len(tds) > 2 else None

                    if not all([fy_from_cell, fy_to_cell]):
                        continue

                    from_year = fy_from_cell.get_text(strip=True)
                    to_year = fy_to_cell.get_text(strip=True)

                    # Extract submission date
                    submission_cell = row.find('td', headers='submissionDate') or tds[5] if len(tds) > 5 else None
                    submission_date = submission_cell.get_text(strip=True) if submission_cell else ''

                    # Extract year for sorting (last part of date)
                    year_of_declaration = (submission_date.split('-')[-1]
                                         if '-' in submission_date else submission_date)

                    # Extract URLs with better error handling
                    pdf_url = xbrl_url = ''

                    # PDF attachment
                    attachment_cell = row.find('td', headers='attachmentFile') or tds[3] if len(tds) > 3 else None
                    if attachment_cell and attachment_cell.find('a'):
                        pdf_url = attachment_cell.find('a').get('href', '')

                    # XBRL file
                    xbrl_cell = row.find('td', headers='xbrlFile') or tds[4] if len(tds) > 4 else None
                    if xbrl_cell and xbrl_cell.find('a'):
                        xbrl_url = xbrl_cell.find('a').get('href', '')

                    # Extract file sizes if available
                    pdf_size = xbrl_size = ''
                    if attachment_cell:
                        size_elem = attachment_cell.find('p', class_='mt-1')
                        if size_elem:
                            pdf_size = size_elem.get_text(strip=True).strip('()')

                    if xbrl_cell:
                        size_elem = xbrl_cell.find('p', class_='mt-1')
                        if size_elem:
                            xbrl_size = size_elem.get_text(strip=True).strip('()')

                    data.append({
                        'company_name': company_name,
                        'fy_from': from_year,
                        'fy_to': to_year,
                        'financial_year': f"{from_year}-{to_year}",
                        'year_of_declaration': year_of_declaration,
                        'submission_date': submission_date,
                        'pdf_url': pdf_url,
                        'xbrl_url': xbrl_url,
                        'pdf_size': pdf_size,
                        'xbrl_size': xbrl_size
                    })

                except Exception as e:
                    logger.warning(f"Error parsing row: {e}")
                    continue

        df = pd.DataFrame(data)

        if not df.empty:
            # Clean and deduplicate
            df = df[df['company_name'].str.len() > 0].drop_duplicates()

            # Sort by company and financial year
            df = df.sort_values(['company_name', 'fy_from'])

            logger.info(f"Parsed {len(df)} BRSR records for {df['company_name'].nunique()} companies")
        else:
            logger.warning("No valid BRSR data found in HTML file")

        return df

    def validate_data(self, df: pd.DataFrame) -> Dict[str, int]:
        """Validate scraped BRSR data."""
        if df.empty:
            return {'total_records': 0, 'error': 'Empty dataframe'}

        return {
            'total_records': len(df),
            'unique_companies': df['company_name'].nunique(),
            'records_with_pdf': df['pdf_url'].str.len().gt(0).sum(),
            'records_with_xbrl': df['xbrl_url'].str.len().gt(0).sum(),
            'financial_years': df['financial_year'].nunique(),
            'companies_with_multiple_years': (df.groupby('company_name')['financial_year'].nunique() > 1).sum(),
            'avg_pdf_size_mb': df['pdf_size'].str.extract(r'(\d+\.?\d*)').astype(float).mean() if 'pdf_size' in df.columns and df['pdf_size'].str.len().gt(0).any() else 0,
            'date_range': f"{df['year_of_declaration'].min()} - {df['year_of_declaration'].max()}" if 'year_of_declaration' in df.columns else 'Unknown'
        }
"""
BRSR Data Scraper Module

This module handles web scraping of BRSR (Business Responsibility and Sustainability Reporting)
data from the NSE India corporate filings portal.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BRSRScraper:
    """
    Scraper for BRSR data from NSE India portal.

    This class handles the extraction of BRSR PDF links and metadata
    from the NSE corporate filings portal.
    """

    def __init__(self, html_file_path: Optional[str] = None):
        """
        Initialize the BRSR scraper.

        Args:
            html_file_path: Path to local HTML file containing BRSR data.
                          If None, will attempt web scraping.
        """
        self.html_file_path = html_file_path
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def scrape_brsr_data(self) -> pd.DataFrame:
        """
        Scrape BRSR data from NSE portal or local HTML file.

        Returns:
            DataFrame containing BRSR metadata with columns:
            - company_name: Name of the company
            - fy_from: Financial year from
            - fy_to: Financial year to
            - financial_year: Combined financial year string
            - year_of_declaration: Year of declaration
            - submission_date: Full submission date
            - pdf_url: URL to BRSR PDF
            - xbrl_url: URL to XBRL file
        """
        if self.html_file_path and Path(self.html_file_path).exists():
            return self._parse_local_html(self.html_file_path)
        else:
            logger.warning("No local HTML file provided. Web scraping not implemented.")
            return pd.DataFrame()

    def _parse_local_html(self, file_path: str) -> pd.DataFrame:
        """
        Parse BRSR data from local HTML file.

        Args:
            file_path: Path to HTML file containing BRSR table data

        Returns:
            DataFrame with parsed BRSR data
        """
        logger.info(f"Parsing BRSR data from {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')

        data = []

        for table in tables:
            tbody = table.find('tbody')
            if tbody:
                rows = tbody.find_all('tr')
                for row in rows:
                    tds = row.find_all('td')
                    if len(tds) >= 7:
                        # Extract company name
                        company_td = row.find('td', headers='companyName')
                        company = company_td.get_text(strip=True) if company_td else ''

                        # Extract financial years
                        fy_from_td = row.find('td', headers='fyFrom')
                        fy_to_td = row.find('td', headers='fyTo')
                        from_year = fy_from_td.get_text(strip=True) if fy_from_td else ''
                        to_year = fy_to_td.get_text(strip=True) if fy_to_td else ''

                        # Extract submission date
                        submission_td = row.find('td', headers='submissionDate')
                        submission_date = submission_td.get_text(strip=True) if submission_td else ''

                        # Extract year of declaration
                        year_of_declaration = (submission_date.split('-')[-1]
                                             if submission_date and '-' in submission_date
                                             else submission_date)

                        # Extract PDF URL
                        pdf_url = ''
                        attachment_td = row.find('td', headers='attachmentFile')
                        if attachment_td:
                            a_tag = attachment_td.find('a')
                            if a_tag and 'href' in a_tag.attrs:
                                pdf_url = a_tag['href']

                        # Extract XBRL URL
                        xbrl_url = ''
                        xbrl_td = row.find('td', headers='xbrlFile')
                        if xbrl_td:
                            a_tag = xbrl_td.find('a')
                            if a_tag and 'href' in a_tag.attrs:
                                xbrl_url = a_tag['href']

                        data.append({
                            'company_name': company,
                            'fy_from': from_year,
                            'fy_to': to_year,
                            'financial_year': f"{from_year}-{to_year}",
                            'year_of_declaration': year_of_declaration,
                            'submission_date': submission_date,
                            'pdf_url': pdf_url,
                            'xbrl_url': xbrl_url
                        })

        df = pd.DataFrame(data)

        # Remove duplicates and empty entries
        df = df[df['company_name'].str.len() > 0].drop_duplicates()

        logger.info(f"Parsed {len(df)} BRSR records for {df['company_name'].nunique()} companies")

        return df

    def validate_data(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Validate the scraped BRSR data.

        Args:
            df: DataFrame with BRSR data

        Returns:
            Dictionary with validation statistics
        """
        stats = {
            'total_records': len(df),
            'unique_companies': df['company_name'].nunique(),
            'records_with_pdf': df['pdf_url'].str.len().gt(0).sum(),
            'records_with_xbrl': df['xbrl_url'].str.len().gt(0).sum(),
            'financial_years': df['financial_year'].nunique(),
        }

        return stats
"""
Tests for Data Scraper Module
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
from src.data_scraper import BRSRScraper


class TestBRSRScraper:
    """Test cases for BRSR data scraper."""

    def test_scraper_initialization(self):
        """Test scraper initialization."""
        scraper = BRSRScraper()
        assert scraper.html_file_path is None

        scraper_with_file = BRSRScraper("test.html")
        assert scraper_with_file.html_file_path == "test.html"

    def test_scrape_without_html_file(self):
        """Test scraping without HTML file returns empty DataFrame."""
        scraper = BRSRScraper()
        df = scraper.scrape_brsr_data()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_validate_data_empty(self):
        """Test validation of empty DataFrame."""
        scraper = BRSRScraper()
        df = pd.DataFrame()
        stats = scraper.validate_data(df)

        expected_keys = ['total_records', 'unique_companies', 'records_with_pdf', 'records_with_xbrl', 'financial_years']
        assert all(key in stats for key in expected_keys)
        assert all(stats[key] == 0 for key in expected_keys)

    def test_validate_data_with_content(self):
        """Test validation of DataFrame with content."""
        scraper = BRSRScraper()

        # Create test data
        test_data = {
            'company_name': ['Company A', 'Company B', 'Company A'],
            'financial_year': ['2022-23', '2022-23', '2023-24'],
            'pdf_url': ['http://example.com/pdf1.pdf', 'http://example.com/pdf2.pdf', ''],
            'xbrl_url': ['http://example.com/xbrl1.xml', '', 'http://example.com/xbrl3.xml']
        }
        df = pd.DataFrame(test_data)

        stats = scraper.validate_data(df)

        assert stats['total_records'] == 3
        assert stats['unique_companies'] == 2
        assert stats['records_with_pdf'] == 2
        assert stats['records_with_xbrl'] == 2
        assert stats['financial_years'] == 2

    def test_parse_local_html_file_not_found(self):
        """Test parsing non-existent HTML file."""
        scraper = BRSRScraper("nonexistent.html")
        df = scraper._parse_local_html("nonexistent.html")
        assert df.empty

    @pytest.mark.parametrize("html_content,expected_rows", [
        ("<table><tbody><tr><td>Company A</td></tr></tbody></table>", 0),  # No proper structure
        ("<table><tbody><tr><td headers='companyName'>Company A</td><td headers='fyFrom'>2022</td></tr></tbody></table>", 0),  # Missing required columns
    ])
    def test_parse_minimal_html_content(self, html_content, expected_rows):
        """Test parsing minimal HTML content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            temp_path = f.name

        try:
            scraper = BRSRScraper()
            df = scraper._parse_local_html(temp_path)
            assert len(df) == expected_rows
        finally:
            Path(temp_path).unlink()
#!/usr/bin/env python3
"""
Run Script for BRSR Sustainability Analysis

This script provides a simple interface to run the BRSR analysis pipeline.
"""

import argparse
import sys
from pathlib import Path

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='BRSR Sustainability Regime Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py --html-file path/to/brsr_data.html
  python run_analysis.py --html-file data/brsr_data.html --output-dir my_results
  python run_analysis.py --html-file data/brsr_data.html --log-level DEBUG
        """
    )

    parser.add_argument(
        '--html-file',
        type=str,
        default='txt.txt',
        help='Path to HTML file containing BRSR data from NSE India (default: txt.txt)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--skip-data-collection',
        action='store_true',
        help='Skip data collection phase (use existing processed data)'
    )

    parser.add_argument(
        '--skip-feature-extraction',
        action='store_true',
        help='Skip feature extraction phase (use existing features)'
    )

    args = parser.parse_args()

    # Import and run the main analysis
    try:
        from src.main import main as run_analysis

        # Modify sys.argv to pass arguments to the main function
        original_argv = sys.argv
        sys.argv = [
            'src/main.py',
            '--html-file', args.html_file,
            '--output-dir', args.output_dir,
            '--log-level', args.log_level
        ]

        if args.skip_data_collection:
            sys.argv.append('--skip-data-collection')

        if args.skip_feature_extraction:
            sys.argv.append('--skip-feature-extraction')

        exit_code = run_analysis()
        sys.argv = original_argv

        return exit_code

    except ImportError as e:
        print(f"Error importing analysis modules: {e}")
        print("Make sure you're running this from the project root directory.")
        return 1

    except Exception as e:
        print(f"Error running analysis: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
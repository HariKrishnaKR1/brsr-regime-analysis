# BRSR Sustainability Regime Analysis

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![NLP](https://img.shields.io/badge/NLP-HMM-orange.svg)](#)
[![Finance](https://img.shields.io/badge/Domain-Sustainability-blue.svg)](#)

## Overview

This project analyzes Business Responsibility and Sustainability Reporting (BRSR) disclosures from Indian companies to infer latent corporate sustainability regimes using Hidden Markov Models (HMMs) and Natural Language Processing (NLP) techniques.

## 🎯 Motivation

BRSR is mandated by SEBI for large listed companies to enhance transparency in sustainability practices. However, there's often a disconnect between disclosure breadth and actual operational integration. This study addresses whether we can distinguish genuine commitment from symbolic compliance by analyzing temporal disclosure patterns using probabilistic frameworks.

## 🔍 Key Features

- **Automated Data Extraction**: Web scraping of BRSR PDFs from NSE India portal
- **NLP Feature Engineering**: Six interpretable dimensions of sustainability disclosure
- **Hidden Markov Models**: Inference of latent sustainability regimes
- **Temporal Analysis**: Longitudinal firm-level trajectory tracking
- **Regime Classification**: Three distinct corporate postures identification

## 📊 Sustainability Regimes

The model identifies three latent regimes based on empirical disclosure patterns:

1. **Minimal Compliance**: Checklist-driven reporting with low quantification
2. **Lobbyist-Influenced Dilution**: Procedural language suggesting regulatory resistance
3. **Enforcement-Oriented**: Explicit targets and capital commitment indicating genuine integration

## 🏗️ Project Structure

```
brsr-regime-analysis/
├── src/
│   ├── __init__.py
│   ├── data_scraper.py          # NSE web scraping functionality
│   ├── feature_extractor.py     # NLP feature engineering
│   ├── hmm_model.py            # Hidden Markov Model implementation
│   ├── visualization.py        # Plotting and analysis utilities
│   └── utils.py                # Helper functions
├── notebooks/
│   └── Assignment 3.ipynb      # Original comprehensive notebook
├── data/
│   ├── raw/                    # Raw PDF downloads
│   └── processed/              # Processed datasets
├── tests/
│   ├── test_data_scraper.py
│   ├── test_feature_extractor.py
│   └── test_hmm_model.py
├── docs/
│   └── technical_details.md
├── requirements.txt
├── setup.py
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Chrome browser (for web scraping)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/brsr-regime-analysis.git
cd brsr-regime-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Data Collection
```python
from src.data_scraper import BRSRScraper

scraper = BRSRScraper()
data = scraper.scrape_brsr_data()
```

#### Feature Extraction
```python
from src.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_features(pdf_text)
```

#### HMM Modeling
```python
from src.hmm_model import SustainabilityHMM

model = SustainabilityHMM(n_states=3)
regimes = model.fit_predict(features)
```

#### Complete Pipeline
```python
from src.main import run_analysis

results = run_analysis()
```

## 📈 Methodology

### Feature Engineering

Six theoretically motivated disclosure dimensions:

1. **Commitment Language**: Targets, pledges, aspirational statements
2. **Metric Intensity**: Quantified indicators, percentages, year-on-year comparisons
3. **Governance Language**: Board committees, accountability mechanisms
4. **Capital Allocation**: Investment and expenditure signaling
5. **Enforcement Language**: Audit, verification, penalty references
6. **Supply Chain Responsibility**: Scope-3 due diligence, supplier standards

### HMM Framework

- **Baum-Welch Algorithm**: Parameter estimation via expectation-maximization
- **Viterbi Algorithm**: Most probable state sequence identification
- **Gaussian Emissions**: Multivariate normal distributions for feature vectors
- **Z-score Normalization**: Within-firm standardization preserving temporal dynamics

## 📊 Results & Evaluation

### Model Validation

- **Log-likelihood**: Model fit assessment
- **Information Criteria**: AIC/BIC for model comparison
- **Viterbi Accuracy**: Against simulated ground truth
- **Linguistic Interpretability**: Emission parameters analysis

### Key Findings

- Clear separation of three distinct disclosure regimes
- Temporal stability within firms with occasional transitions
- Strong correlation between regime assignments and qualitative assessments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NSE India for BRSR data accessibility
- SEBI for sustainability reporting framework
- Academic literature on corporate sustainability disclosure

## 📚 References

- Securities and Exchange Board of India (SEBI) BRSR Framework
- Hidden Markov Models for sequential data analysis
- NLP applications in financial disclosure analysis

## 📞 Contact

**K R Hari Krishna**
- Roll Number: M2022BSASS020
- Date: 15-12-2025

---

*This research contributes to evidence-based assessment of corporate sustainability disclosure quality and regulatory effectiveness in emerging markets.*
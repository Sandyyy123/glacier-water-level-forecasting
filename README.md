> **⚠️ Proprietary — All Rights Reserved.** © 2026 Sandeep Grover. This repository is licensed to Sandeep Grover and may **not** be used, run, copied, modified, distributed, or used to train models without prior written permission. Public visibility does not grant a license. See [LICENSE](LICENSE).

---

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Environmental ML](https://img.shields.io/badge/Environmental-ML-green) ![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)

# Vernagtferner Glacier Water Level Forecasting

Time-series regression forecasting glacial meltwater runoff from meteorological inputs using LSTM and gradient boosting.

---

## Task

**Environmental Time-series Forecasting**

---

## Architecture

```
Met Station Inputs → Lag Features → LightGBM / LSTM → Daily Runoff Forecast → Walk-forward CV
```

---

## Key Features

- Daily meltwater runoff forecast from temperature, precipitation, snow depth
- LSTM on multivariate meteorological sequence inputs
- LightGBM baseline on lag + calendar features
- Walk-forward validation respecting temporal causality
- Climate-change framing: accelerating melt trend modelling

---

## Dataset

[Vernagtferner Glacier — BAdW Hydrological Station](https://www.bayerische-gletscher.de/vernagtferner/abfluss)

---

## Project Structure

```
├── src/
│   ├── model_baseline.py      # Baseline model
│   └── model_advanced.py      # Advanced model
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory analysis
├── manuscripts/
│   └── manuscript.md          # IMRaD writeup
├── reports/
│   └── references.md          # Verified references
├── deliverables/
│   └── presentation.html      # Self-contained HTML
├── data/
│   └── README.md              # Dataset download instructions
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Sandyyy123/glacier-water-level-forecasting.git
cd glacier-water-level-forecasting
pip install -r requirements.txt

# See data/README.md for dataset download
python src/model_baseline.py
python src/model_advanced.py
```

---

## Tech Stack

`PyTorch · LightGBM · pandas · scikit-learn`

---

## Author

**Dr. Sandeep Grover** — PhD Data Science, independent ML researcher, Germany.

---

## License

MIT

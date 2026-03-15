# SGD-VQC-EWS: Predicting Sharp Economic Growth Deceleration Using a Variational Quantum Classifier

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Qiskit 1.x](https://img.shields.io/badge/Qiskit-1.x-6929c4.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Paper:** "Predicting Sharp Economic Growth Deceleration in Developing Countries: A Variational Quantum Classifier Approach"
> **Journal:** *Expert Systems with Applications* (ESWA), Elsevier — ISSN: 0957-4174

---

## Overview

This repository contains the complete Python pipeline for replicating all results reported in the paper. The pipeline implements:

- **SGD target construction**: Sharp Growth Deceleration label (three threshold variants: 40%, 50%, 60%)
- **Post-split SMOTE** (k=4) oversampling to handle class imbalance
- **GridSearchCV** hyperparameter tuning for all classical baselines
- **Variational Quantum Classifier (VQC)** using Qiskit v1.x — ablation-validated configuration: ZZFeatureMap(reps=1) + EfficientSU2(reps=2) + COBYLA(maxiter=500)
- **Dual-threshold evaluation** (default τ=0.50 and optimal τ=0.45)
- **Bootstrap confidence intervals** (n=1,000, seed=42)
- **McNemar statistical testing**
- **Circuit depth ablation** across 9 VQC configurations
- **Asymmetric loss analysis** and **calibration analysis**
- **All figure-generation scripts** (publication-ready, ESWA style)
- **Per-country breakdown** and **robustness checks** (SGD threshold sensitivity)

All random seeds are fixed (`seed=42` throughout). Results are fully reproducible using the provided `environment.yml`.

---

## Repository Structure

```
SGD-VQC-ESWA/
├── sgd_vqc_pipeline.py      # Main pipeline (all steps, v3.1)
├── environment.yml           # Conda environment with pinned versions
├── requirements.txt          # pip-based alternative
├── .gitignore
├── data/
│   └── README.md             # Data download instructions (World Bank + IMF)
└── sgd_results_v31/          # Auto-created on run — all outputs saved here
    ├── *.png                  # Figures (ROC, PR curves, confusion matrices, etc.)
    ├── sgd_full_results_v3.xlsx
    ├── results_summary.json
    └── sgd_events.csv
```

---

## Data

**No proprietary data are used.** All macroeconomic data are sourced from publicly available repositories:

| Source | URL |
|--------|-----|
| World Bank World Development Indicators (WDI) | https://databank.worldbank.org/source/world-development-indicators |
| IMF International Financial Statistics (IFS) | https://data.imf.org/ |

See [`data/README.md`](data/README.md) for detailed download and formatting instructions.

The pipeline expects a CSV file named `economic_data_collected.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `country` | ISO country code or name |
| `year` | Year (integer) |
| `gdp_growth` | GDP growth rate (annual %, WDI: NY.GDP.MKTP.KD.ZG) |
| `inflation` | CPI inflation (annual %, WDI: FP.CPI.TOTL.ZG) |
| `unemployment` | Unemployment rate (%, WDI: SL.UEM.TOTL.ZS) |
| `reserves_import_cover` | Foreign reserves in months of imports (IMF IFS) |
| `trade_openness_pct_gdp` | Trade (% of GDP, WDI: NE.TRD.GNFS.ZS) |
| `government_expenditure_pct_gdp` | Gov. expenditure (% of GDP, WDI: NE.CON.GOVT.ZS) |

---

## Installation

### Option A — Conda (recommended)

```bash
git clone https://github.com/canhtq-stack/SGD-VQC-ESWA.git
cd SGD-VQC-ESWA
conda env create -f environment.yml
conda activate sgd-vqc
```

### Option B — pip

```bash
git clone https://github.com/canhtq-stack/SGD-VQC-ESWA.git
cd SGD-VQC-ESWA
pip install -r requirements.txt
```

> **Hardware note:** VQC runs on Qiskit's statevector simulator (CPU). Full pipeline (with ablation) takes approximately 60–90 minutes on a standard laptop. Use `--skip-ablation` to reduce runtime to ~5–10 minutes.

---

## Usage

```bash
# Full pipeline (recommended for replication)
python sgd_vqc_pipeline.py

# With a custom data path
python sgd_vqc_pipeline.py --data path/to/economic_data_collected.csv

# Skip VQC (classical baselines only, ~5 min)
python sgd_vqc_pipeline.py --skip-vqc

# Skip circuit depth ablation (~60 min saved)
python sgd_vqc_pipeline.py --skip-ablation

# Faster bootstrap CI (n=500 instead of 1000)
python sgd_vqc_pipeline.py --bootstrap-n 500

# Minimal run (classical only, no ablation)
python sgd_vqc_pipeline.py --skip-vqc --skip-ablation
```

All outputs are saved to `./sgd_results_v31/`.

---

## VQC Configuration (Ablation-Validated)

The primary VQC configuration reported in the paper was validated through a 9-configuration ablation study (Table in paper):

| Component | Setting | Rationale |
|-----------|---------|-----------|
| Feature map | ZZFeatureMap, reps=1 | AUC=0.750, best in ablation grid |
| Ansatz | EfficientSU2, reps=2 | 24 trainable parameters |
| Optimizer | COBYLA, maxiter=500 | Gradient-free, robust on small datasets |
| Input features | 4 PCA components | 85%+ variance retained |
| Reporting threshold τ | 0.45 (optimal) / 0.50 (default) | F1=0.529, Recall=0.90 at τ=0.45 |

---

## Reproducibility

- **All random seeds fixed:** `numpy.random.seed(42)`, `random_state=42` for all estimators and CV splits
- **Time-based split:** Train ≤ 2018, Test ≥ 2019 (no data leakage)
- **SMOTE applied post-split** on training set only
- **PCA fitted on augmented training set**, applied to raw test set
- **Pinned package versions** in `environment.yml`

To verify results match the paper, check `sgd_results_v31/results_summary.json` after running the pipeline.

---

## Key Results (Table 3 in paper)

| Model | AUC | F1 (τ=0.50) | F1 (τ_opt) | Recall (τ_opt) |
|-------|-----|-------------|------------|----------------|
| VQC ★ | 0.750 | — | 0.529 | 0.90 |
| Best Classical | — | — | — | — |

*Full results including bootstrap CIs and McNemar p-values are in `sgd_results_v31/sgd_full_results_v3.xlsx`.*

---

## Citation

If you use this code, please cite:

```bibtex
@article{author2025sgdvqc,
  title   = {Predicting Sharp Economic Growth Deceleration in Developing Countries:
             A Variational Quantum Classifier Approach},
  journal = {Expert Systems with Applications},
  year    = {2025},
  issn    = {0957-4174},
  doi     = {10.1016/j.eswa.XXXX.XXXXXX}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

For questions about the code or data, please open an issue on this repository.

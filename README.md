# Ambiguity Detection in Software Requirements — Replication Package

This repository contains the replication package for the manuscript:

"Ambiguity Detection in Software Requirements: Cross-Dataset Evaluation of Classical and Transformer-Based Models"

This package includes ONLY the following experiment scripts:

- baseline_ml_experiments.py
- distilbert_best_only_cpu.py
- eda.py

It also includes:
- requirements.txt
- REPLICATION.md
- LICENSE
- data/pure_labeled.csv
- data/userstories_labeled.csv

------------------------------------------------------------
Repository Structure
------------------------------------------------------------

.
├── data/
│   ├── pure_labeled.csv
│   └── userstories_labeled.csv
├── baseline_ml_experiments.py
├── distilbert_best_only_cpu.py
├── eda.py
├── requirements.txt
├── REPLICATION.md
├── LICENSE
└── README.md

The scripts automatically create:
- results/
- figures/

------------------------------------------------------------
Environment Setup
------------------------------------------------------------

Create a virtual environment:

python -m venv .venv
source .venv/bin/activate        (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install -r requirements.txt

If torch installation fails, install CPU build first:

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

------------------------------------------------------------
Reproducing Experiments
------------------------------------------------------------

1) Classical Machine Learning Baselines

python baseline_ml_experiments.py

Runs:
- Intra-dataset (PURE, UserStories)
- Cross-dataset (PURE→UserStories and reverse)

Outputs:
- results/baseline_results.csv
- results/report_*.txt
- figures/confusion_*.png

------------------------------------------------------------

2) Transformer Model (DistilBERT, CPU)

python distilbert_best_only_cpu.py

Runs:
- PURE train/val/test split
- Evaluation on PURE test (intra)
- Cross-dataset evaluation (PURE→UserStories)

Outputs:
- results/bert_cpu_best_results.csv
- results/bert_cpu_report_*.txt
- figures/bert_cpu_confusion_*.png

------------------------------------------------------------

3) Exploratory Data Analysis (EDA)

python eda.py

Generates:
- Class distribution plots
- Word/character length plots
- Ambiguity trigger plots
- eda_summary.csv

Outputs saved in:
- figures/
- eda_summary.csv

------------------------------------------------------------
Reproducibility Notes
------------------------------------------------------------

- Random seed fixed to 42.
- Classical models use stratified splits with random_state=42.
- Transformer experiment calls set_seed(42).
- All experiments are configured for CPU execution.
- Exact tested environment is documented in REPLICATION.md.

------------------------------------------------------------

For replication details, see REPLICATION.md.

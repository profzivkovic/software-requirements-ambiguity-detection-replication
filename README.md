# Ambiguity Detection in Software Requirements — Replication Package

This repository contains the replication package for the paper:

**"Ambiguity Detection in Software Requirements: Cross-Dataset Evaluation of Classical and Transformer-Based Models"**

---

## 📌 Overview

This project investigates automated ambiguity detection in software requirements, with a focus on **cross-dataset generalization** between:

- **PURE** (formal software requirements specifications)
- **User Stories** (agile requirements)

The repository provides full experimental code for:

- Classical machine learning models (SVM, Random Forest, XGBoost, LightGBM)
- Transformer-based models (BERT, RoBERTa, ALBERT, DistilBERT)
- T5 (text-to-text formulation)
- Cross-dataset evaluation (both directions)
- Statistical significance testing

---

## ⚙️ Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📂 Data

Datasets are **included in this repository** under:

```
data/
```

Files:

- `pure_labeled.csv`
- `userstories_labeled.csv`

Each dataset contains:

- `requirement_text`
- `label`

---

## 🚀 Running Experiments

### Classical models
```bash
python run_baselines_multiseed.py
```

### Transformer models
```bash
python run_transformers.py
```

### T5 model
```bash
python run_t5.py
```

---

## 📊 Analysis

### Aggregate results
```bash
python analyze_results.py
```

### Statistical significance testing
```bash
python run_transformer_significance.py
```

---

## 📁 Outputs

Generated outputs (not included in repository):

- results/
- figures/

These will be created automatically when running experiments.

---

## 🔁 Reproducibility

The repository is designed to enable full reproduction of experiments described in the paper:

- Fixed seeds
- Defined evaluation protocol
- Explicit model configurations

---

## 📄 License

This project is licensed under the MIT License.

---

## 📖 Citation

If you use this code, please cite the associated paper.

# Replication and Reproducibility Information

This document provides technical details necessary to reproduce the experimental
results reported in the manuscript:

"Ambiguity Detection in Software Requirements: Cross-Dataset Evaluation of Classical and Transformer-Based Models"

---------------------------------------------------------------------
1. Tested Environment
---------------------------------------------------------------------

The experiments were tested under the following environment:

Operating System:
- Windows 11 Version 25H2

Hardware:
- CPU: Intel Core i5-11400H
- RAM: 16 GB
- GPU: Not used (all experiments executed on CPU)

Python:
- Python 3.10.12

Key Libraries:
- torch 2.3.1 (CPU build)
- transformers 4.41.2
- scikit-learn 1.4.2
- lightgbm 4.3.0
- xgboost 2.0.3
- pandas 2.2.2
- numpy 1.26.4
- matplotlib 3.8.4

Exact versions are listed in requirements.txt.

---------------------------------------------------------------------
2. Determinism and Random Seeds
---------------------------------------------------------------------

To ensure reproducibility:

- Random seed fixed to 42 in all experiments.
- Classical ML models use:
      train_test_split(..., random_state=42)
- Transformer experiments use:
      transformers.set_seed(42)

Minor numerical variations may occur across different platforms
due to low-level linear algebra implementations.

---------------------------------------------------------------------
3. Dataset Splitting Protocol
---------------------------------------------------------------------

Classical ML experiments:
- Stratified 80/20 train/test split per dataset.

Transformer experiments:
- PURE dataset split into:
    70% training
    10% validation
    20% test
- Best model retrained on train+validation for final evaluation.

Cross-dataset evaluation:
- Model trained on full source dataset.
- Evaluated on full target dataset.

---------------------------------------------------------------------
4. Expected Outputs
---------------------------------------------------------------------

Running the provided scripts generates:

results/
- baseline_results.csv
- bert_cpu_best_results.csv
- classification report text files

figures/
- Confusion matrix PNG files for each setting

These outputs correspond to the tables and figures reported in the manuscript.

---------------------------------------------------------------------
5. Notes on Execution Time
---------------------------------------------------------------------

Approximate runtimes (CPU):
- Classical ML baselines: ~2–5 minutes
- Best-only DistilBERT run: ~15–25 minutes

---------------------------------------------------------------------
6. Replication Verification
---------------------------------------------------------------------

To verify the results:

1. Create a clean virtual environment.
2. Install dependencies via:
      pip install -r requirements.txt
3. Place datasets in the data/ directory.
4. Run the experiment scripts.

The generated F1, precision, and recall values should match those
reported in the manuscript within negligible numerical tolerance.

---------------------------------------------------------------------

For questions regarding replication, please contact the corresponding author.

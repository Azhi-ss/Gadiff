# Polymer GA Optimization Toolkit

A comprehensive toolkit for calculating polymer properties, executing genetic algorithm (GA) evolutionary optimizations, and generating related scientific visualizations.

## 🚀 Overview

This repository integrates tools and models for molecular design:
- **`src/models/polyga`**: The core Polymer Genetic Algorithm library.
- **`src/models/MMPolymer`**: Pre-trained machine learning models for property prediction (e.g., Dielectric Constant (DC) and Glass Transition Temperature (Tg)).
- **`src/optimization`**: Pipeline scripts to run GA experiments, perform cross-over, mutations, and trace the evolutionary trajectory.
- **`scripts/`**: Utilities for integrating the models, cleaning training datasets, and processing large data outputs.

## 📂 Repository Structure

- **`src/`**: Core logic including models (`models/polyga`, `models/MMPolymer`) and optimization workflows.
- **`scripts/`**: Standalone automation, data processing, and integration scripts.
- **`data/`**: Core curated datasets (vocabulary, seed databases of fragments/SMILES).
- **`examples/`**: Demonstration structural representations (e.g., POSCAR configs) and baseline model outputs.

## 📦 Installation & Requirements

Ensure you are using an environment optimized for chemoinformatics and deep learning (e.g. Conda). The key dependencies typically include:

- Python 3.8+
- PyTorch
- RDKit
- NumPy, pandas, SciPy, scikit-learn
- SQLAlchemy
- Matplotlib, Seaborn

See `requirements.txt` for specific version pinning.

## 💡 Quick Start

Before running the GA optimisation, verify the integration components.
Then execute evolutionary visualisation tracking (from initial random generation to sweet-spot convergence):
```bash
python src/optimization/run_optimization.py
```

## 📝 License / Citation
*(Please refer to the LICENSE files found in `src/models` subdirectories for respective licensing info.)*
# MM-GenePoly

# PolyDiff-3D: Polymer GA Optimization with Diffusion-Guided Evolution

A comprehensive toolkit for polymer property prediction and genetic algorithm (GA) evolutionary optimization, enhanced with diffusion-model-guided genetic operators.

## Overview

- **`src/models/polyga`** — Core Polymer Genetic Algorithm library with EGD noise-space mutation and DEMO three-population co-evolution
- **`src/models/MMPolymer`** — Transformer-based property prediction for linear polymers (Dielectric Constant, Glass Transition Temperature)
- **`src/optimization`** — GA experiment pipelines, crossover, mutation, and evolutionary trajectory visualization
- **`scripts/`** — Integration utilities, data processing, and fragment pregeneration

### New: Diffusion-Guided Genetic Operators

| Module | Description |
|--------|-------------|
| `diffusion_mutator.py` | EGD noise-space mutation/crossover — mutates fragments in 3D continuous space rather than random ID replacement |
| `saes_selector.py` | Structure-Aware Environmental Selection — maintains diversity via composite fingerprint + shape distance |
| `demo_polyga.py` | Three-population co-evolution (Explorer / Refiner / Elite Archive) with adaptive noise scheduling |
| `validate_linear_polymer()` | Linear polymer constraint — MMPolymer requires exactly 2 `[*]` endpoints |

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: Python 3.8+, PyTorch, RDKit, NumPy, pandas, SciPy, SQLAlchemy, Uni-Core.

## Quick Start

```bash
# Run GA optimization
python src/optimization/run_optimization.py

# Pregenerate fragments for DNA pool expansion (requires EvoDiffMol checkpoint)
python scripts/pregenerate_fragments.py --checkpoint /path/to/model.pt --n-molecules 500 --output data/enriched_dna.csv
```

## Formal PolyGen

The formal PolyGen workflow uses EvoDiffMol-enriched DNA fragments, MMPolymer
property prediction, and a balanced normalized fitness objective:

```text
fitness = (tg_norm + dc_norm + sa_norm) / 3
```

Where higher `Tg` is better, lower `DC` is better, and lower `SA_score` is
better. The default inputs are:

- DNA: `data/enriched_dna.csv`
- MMPolymer weights: `/internfs/Zy/polygen_assets/mm_polymer/finetune_data`
- Tokenizer files: `data/tokenizer/vocab.json` and `data/tokenizer/merges.txt`
- Uni-Core dictionary: `src/models/MMPolymer/dict.txt`

Validate setup before a formal run:

```bash
python scripts/validate_polygen_setup.py --repo-root . --weight-dir /internfs/Zy/polygen_assets/mm_polymer/finetune_data
```

Run the formal optimization entrypoint:

```bash
PYTHONPATH=src/models:. python src/optimization/run_optimization.py
```

## Testing

```bash
# All tests are mock-based — no GPU or model weights required
pytest tests/ -v
```

## Repository Structure

```
├── src/
│   ├── models/
│   │   ├── polyga/          # GA core + EGD/SAES/DEMO modules
│   │   └── MMPolymer/       # Transformer property predictor
│   └── optimization/        # Experiment pipelines & visualization
├── scripts/
│   ├── integration/         # PolyGA-MMPolymer integration
│   └── pregenerate_fragments.py  # Diffusion fragment generation
├── tests/                   # 145 mock tests
├── data/tokenizer/          # Polymer SMILES tokenizer
└── requirements.txt
```

## License

Please refer to LICENSE files in `src/models` subdirectories for respective licensing.

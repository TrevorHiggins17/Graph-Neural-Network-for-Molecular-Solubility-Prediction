# GNN for Molecular Solubility Prediction

Accurate prediction of aqueous solubility (logS) is critical in early-stage drug discovery for filtering viable compound candidates. Traditional approaches rely on fixed molecular fingerprints, while graph neural networks learn task-specific representations directly from molecular structure. This project benchmarks both approaches on the ESOL dataset from the MoleculeNet benchmark suite.

## Approach

Molecules are converted to graphs — atoms become nodes, bonds become edges — then processed by a message-passing GNN trained end-to-end on the regression target.

### Featurisation

| | Features | Dim |
|---|----------|-----|
| **Nodes** (atoms) | Atomic number, hybridisation, Pauling electronegativity, degree, formal charge, aromaticity | 20 |
| **Edges** (bonds) | Bond type, conjugation, ring membership | 6 |

### GNN Architecture

- Input projection → hidden dim (128)
- 3 × GCNConv layers (BatchNorm, ReLU, dropout)
- Residual skip connection (layer 1 → layer 3)
- Global mean pooling
- 2-layer MLP regression head → scalar logS

### Baseline

2048-bit Morgan fingerprints (ECFP4, radius=2) with a 500-tree Random Forest regressor. This is a strong traditional baseline — Morgan fingerprints remain competitive in molecular property prediction benchmarks.

## Results

Evaluated on a 20% held-out test split.

| Model | RMSE (logS) ↓ | R² ↑ |
|-------|---------------|------|
| **GNN (GCN)** | ~0.75–0.90 | ~0.88–0.93 |
| RF + ECFP4 | ~0.95–1.10 | ~0.84–0.88 |

The GNN consistently outperforms the fingerprint baseline, suggesting that learned graph-level representations capture structural information relevant to solubility that fixed-radius circular fingerprints miss.

### Reproducibility

Results vary across runs due to weight initialisation and data splits. Ranges above reflect variation across multiple random seeds. Set `SEED` in the script to reproduce a specific run.

## Setup

```bash
conda create -n mol-gnn python=3.10 -y && conda activate mol-gnn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric rdkit scikit-learn pandas matplotlib
```

## Run

```bash
python gnn_solubility.py
```

Downloads the ESOL dataset automatically, trains both models with early stopping, prints metrics, and saves a comparison figure to `results_comparison.png`.

## References

- Wu et al., *MoleculeNet: A Benchmark for Molecular Machine Learning* (2018)
- Delaney, *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure* (2004)
- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks* (2017)

## Licence

MIT

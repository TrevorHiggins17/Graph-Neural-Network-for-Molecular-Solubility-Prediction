"""
GNN for molecular solubility prediction (ESOL dataset).
Compares a 3-layer GCN against a Morgan fingerprint + Random Forest baseline.

PyTorch Geometric / RDKit / sklearn
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---- make CPU runs less painful ----
# (prevents torch + numpy from fighting for every core and stalling)
torch.set_num_threads(min(4, os.cpu_count() or 4))

# -- atom/bond vocab for one-hot encoding --
ATOM_NUMS = [6, 7, 8, 9, 15, 16, 17, 35, 53]
HYBRIDISATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
# pauling electronegativities
EN = {
    1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66
}

def one_hot(val, allowed):
    """One-hot with an extra 'other' bin."""
    enc = [0] * (len(allowed) + 1)
    if val in allowed:
        enc[allowed.index(val)] = 1
    else:
        enc[-1] = 1
    return enc


def atom_features(atom):
    return (
        one_hot(atom.GetAtomicNum(), ATOM_NUMS)              # 10
        + one_hot(atom.GetHybridization(), HYBRIDISATIONS)   # 6
        + [EN.get(atom.GetAtomicNum(), 2.20) / 3.98]         # 1 normalised EN
        + [atom.GetDegree()]                                 # 1
        + [atom.GetFormalCharge()]                           # 1
        + [float(atom.GetIsAromatic())]                      # 1  -> 20 total
    )


def bond_features(bond):
    bt = bond.GetBondType()
    return (
        [int(bt == t) for t in BOND_TYPES]        # 4
        + [float(bond.GetIsConjugated())]         # 1
        + [float(bond.IsInRing())]                # 1  -> 6 total
    )


def smiles_to_graph(smi):
    """SMILES -> PyG Data object. Returns None if RDKit can't parse it."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_idx, edge_attr = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        edge_idx += [[i, j], [j, i]]
        edge_attr += [bf, bf]

    if edge_idx:
        edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def morgan_fps(smiles_list, radius=2, nbits=2048):
    """Morgan fingerprints (ECFP4) using the non-deprecated generator API."""
    gen = GetMorganGenerator(radius=radius, fpSize=nbits)
    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = gen.GetFingerprint(mol)
            out.append(np.array(fp))
        else:
            out.append(np.zeros(nbits, dtype=int))
    return np.array(out)

class SolubilityGNN(nn.Module):
    """
    3-layer GCN with batch norm, residual skip, global mean pooling,
    and a 2-layer MLP readout head.
    """

    def __init__(self, in_dim, hidden=64, drop=0.1):
        super().__init__()
        # hidden reduced from 128 -> 64 for CPU speed
        self.proj = nn.Linear(in_dim, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden // 2, 1),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.proj(x))

        h = self.drop(F.relu(self.bn1(self.conv1(x, ei))))
        h = self.drop(F.relu(self.bn2(self.conv2(h, ei))))
        h = h + x  # residual
        h = self.drop(F.relu(self.bn3(self.conv3(h, ei))))

        h = global_mean_pool(h, batch)
        return self.head(h).squeeze(-1)


def train_gnn(model, train_loader, test_loader, epochs=120, lr=1e-3, patience=20):
    """
    CPU-friendly defaults:
      - epochs reduced (300 -> 120)
      - patience reduced (30 -> 20)
      - hidden reduced in model (128 -> 64)
    """
    model = model.to(DEVICE)
    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = ReduceLROnPlateau(opt, factor=0.5, patience=10)
    loss_fn = nn.MSELoss()

    best_loss, best_state, wait = float("inf"), None, 0
    train_hist, test_hist = [], []

    for ep in range(1, epochs + 1):
        model.train()
        t_loss = 0.0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad(set_to_none=True)

            pred = model(batch)
            loss = loss_fn(pred, batch.y.view(-1))  # ensure shape [B]
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            t_loss += loss.item() * batch.num_graphs

        t_loss /= len(train_loader.dataset)

        # eval
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                pred = model(batch)
                v_loss += loss_fn(pred, batch.y.view(-1)).item() * batch.num_graphs
        v_loss /= len(test_loader.dataset)

        sched.step(v_loss)
        train_hist.append(t_loss)
        test_hist.append(v_loss)

        if v_loss < best_loss:
            best_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if ep % 20 == 0 or ep == 1:
            print(f"  ep {ep:3d}  train {t_loss:.4f}  test {v_loss:.4f}  lr {opt.param_groups[0]['lr']:.1e}")

        if wait >= patience:
            print(f"  early stop at epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.to(DEVICE)
    return train_hist, test_hist


@torch.no_grad()
def predict_gnn(model, loader):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        preds.append(model(batch).cpu().numpy())
        targets.append(batch.y.view(-1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def plot_results(results, path="results_comparison.png"):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Aqueous Solubility Prediction — GNN vs Random Forest Baseline",
                 fontsize=13, fontweight="bold", y=1.02)

    colours = {"GNN (3-layer GCN)": "#2196F3", "Random Forest + ECFP4": "#4CAF50"}

    # scatter plots (first two panels)
    for idx, (name, r) in enumerate(results.items()):
        if idx >= 2:
            break
        ax = axes[idx]
        ax.scatter(r["true"], r["pred"], alpha=0.5, s=20, c=colours.get(name, "#888"),
                   edgecolors="white", linewidth=0.3)
        lo = min(r["true"].min(), r["pred"].min()) - 0.5
        hi = max(r["true"].max(), r["pred"].max()) + 0.5
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual logS"); ax.set_ylabel("Predicted logS")
        ax.set_title(f"{name}\nRMSE={r['rmse']:.3f}   R²={r['r2']:.3f}", fontsize=11)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.25)

    # bar chart (third panel)
    ax = axes[2]
    names = list(results.keys())
    rmses = [results[n]["rmse"] for n in names]
    r2s = [results[n]["r2"] for n in names]
    x = np.arange(len(names))
    w = 0.3
    ax.bar(x - w/2, rmses, w, label="RMSE ↓", color="#EF5350", alpha=0.85)
    ax2 = ax.twinx()
    ax2.bar(x + w/2, r2s, w, label="R² ↑", color="#42A5F5", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9, rotation=12, ha="right")
    ax.set_ylabel("RMSE", color="#EF5350"); ax2.set_ylabel("R²", color="#42A5F5")
    ax.set_title("Metrics Comparison", fontsize=11)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"figure saved -> {path}")
    plt.show()

def main():
    print(f"device: {DEVICE}\n")

    # load data
    csv = "delaney-processed.csv"
    if not os.path.exists(csv):
        url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
        print("downloading ESOL dataset...")
        df = pd.read_csv(url)
    else:
        df = pd.read_csv(csv)

    target_col = "measured log solubility in mols per litre"
    df = df[["smiles", target_col]].rename(columns={target_col: "y"})
    print(f"loaded {len(df)} molecules  |  logS range [{df.y.min():.1f}, {df.y.max():.1f}]\n")

    # build graphs
    graphs, ys, kept_smiles = [], [], []
    bad = 0
    for _, row in df.iterrows():
        smi = row.smiles
        g = smiles_to_graph(smi)
        if g is None:
            bad += 1
            continue
        g.y = torch.tensor([row.y], dtype=torch.float)
        graphs.append(g)
        ys.append(row.y)
        kept_smiles.append(smi)

    ys = np.array(ys)
    fps = morgan_fps(kept_smiles)

    print(f"featurised {len(graphs)} molecules (dropped {bad})"
          f"  |  node feats: {graphs[0].x.shape[1]}  edge feats: {graphs[0].edge_attr.shape[1]}\n")

    #train/test split
    idx = np.arange(len(graphs))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED)

    train_graphs = [graphs[i] for i in tr]
    test_graphs = [graphs[i] for i in te]
    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=64)

    X_tr, X_te = fps[tr], fps[te]
    y_tr, y_te = ys[tr], ys[te]
    print(f"split: {len(tr)} train / {len(te)} test\n")

    #GNN
    print("training GNN...")
    gnn = SolubilityGNN(in_dim=graphs[0].x.shape[1], hidden=64)
    n_params = sum(p.numel() for p in gnn.parameters())
    print(f"  {n_params:,} parameters")
    train_gnn(gnn, train_loader, test_loader)

    gnn_pred, gnn_true = predict_gnn(gnn, test_loader)
    gnn_rmse = np.sqrt(mean_squared_error(gnn_true, gnn_pred))
    gnn_r2 = r2_score(gnn_true, gnn_pred)
    print(f"  -> RMSE {gnn_rmse:.4f}  R² {gnn_r2:.4f}\n")

    #Random Fores
    print("training random forest...")
    rf = RandomForestRegressor(
        n_estimators=400,          
        min_samples_split=5,
        n_jobs=-1,
        random_state=SEED
    )
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)
    rf_rmse = np.sqrt(mean_squared_error(y_te, rf_pred))
    rf_r2 = r2_score(y_te, rf_pred)
    print(f"  -> RMSE {rf_rmse:.4f}  R² {rf_r2:.4f}\n")

    # results
    print("=" * 45)
    print(f"  GNN (GCN)         RMSE {gnn_rmse:.4f}   R² {gnn_r2:.4f}")
    print(f"  RF + ECFP4        RMSE {rf_rmse:.4f}   R² {rf_r2:.4f}")
    print("=" * 45)

    results = {
        "GNN (3-layer GCN)": {"true": gnn_true, "pred": gnn_pred, "rmse": gnn_rmse, "r2": gnn_r2},
        "Random Forest + ECFP4": {"true": y_te, "pred": rf_pred, "rmse": rf_rmse, "r2": rf_r2},
    }
    plot_results(results)


if __name__ == "__main__":
    main()
import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader
from dataset import MyDataset, collate_fn, KIBADataset
from model import Model
from config.cfg import get_cfg_defaults
from solver import Solver
import matplotlib.pyplot as plt
import numpy as np

# ---- config ----
CKPT_PATH = "saved/model_-3459988036641379738_epoch_39.pt"  # replace XX with the epoch you want
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ---- load checkpoint ----
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
cfg = get_cfg_defaults()

model = Model(cfg=cfg)
model.load_state_dict(ckpt['model_state_dict'])
model.to(DEVICE)

solver = Solver(model, cfg, device=DEVICE, optim=torch.optim.Adam, loss_fn=cfg.SOLVER.LOSS_FN, eval=None)

test_ds = MyDataset(cfg.DATA.TEST_CSV_PATH, cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=False)

set = next(iter(test_dl))
labels = set['label'].to(DEVICE)
protein_emb = set['protein_emb'].to(DEVICE)
drug_emb = set['drug_emb'].to(DEVICE)
protein_mask = set['protein_mask'].to(DEVICE)
drug_mask = set['drug_mask'].to(DEVICE)
drug_id = set['drug_id']
protein_id = set['uniprot_id']
print("Drug ID:", drug_id)
print("Protein ID:", protein_id)

drug_rep = torch.load(f"drug/embeddings_atomic/{drug_id[0]}_unimol.pt", weights_only=False)
print(np.array(drug_rep["atomic_reprs"]))  # (N_atoms, 512)
print(np.array(drug_rep["atomic_reprs"]).shape)  # (N_atoms, 512)
print(np.array(drug_rep["atomic_symbol"]))  # (N_atoms, 512)
print(np.array(drug_rep["atomic_symbol"]).shape)  # (N_atoms, 512)

model.eval()
with torch.no_grad():
    predictions, attentionp, attentiond = model(
        protein_emb,
        drug_emb,
        protein_mask=protein_mask,
        drug_mask=drug_mask,
        return_attention=True,
    )

print("Predictions:", predictions, predictions.shape)
print("Protein Attention:", attentionp, attentionp.shape)
print("Drug Attention:", attentiond, attentiond.shape)

# ---- attention → numpy (batch 0) ----
# For nn.MultiheadAttention with batch_first=True and default average_attn_weights=True,
# attention shapes are (B, L_target, L_source) with no explicit head dimension.
# So we just take the first batch element directly.
# attentionp: (B, Lp, Ld)  protein queries → drug keys
# attentiond: (B, Ld, Lp)  drug queries → protein keys
attn_p = attentionp[0].detach().cpu().numpy()  # (Lp, Ld)
attn_d = attentiond[0].detach().cpu().numpy()  # (Ld, Lp)

# ---- build atom labels for drug dimension ----
# drug_rep['atomic_symbol'] is expected to contain one symbol per atom (possibly with a SEP at the end)
atom_labels = np.array(drug_rep["atomic_symbol"]).reshape(-1)

# align labels with number of drug tokens (Ld)
Ld = attn_p.shape[1]
if atom_labels.shape[0] == Ld + 1:
    # drop SEP or extra token
    atom_labels = atom_labels[:-1]
elif atom_labels.shape[0] > Ld:
    atom_labels = atom_labels[:Ld]
elif atom_labels.shape[0] < Ld:
    # pad with generic labels if needed
    pad_len = Ld - atom_labels.shape[0]
    atom_labels = np.concatenate([atom_labels, np.array([f"X{i}" for i in range(pad_len)])])

# choose an atom index to highlight (0-based). Change this depending on which atom you care about.
highlight_atom_idx = 0
highlight_atom_idx = max(0, min(highlight_atom_idx, Ld - 1))

# ---- symmetric interaction map (residue ↔ atom) ----
# attn_p: (Lp, Ld) protein queries → drug tokens
# attn_d: (Ld, Lp) drug queries → protein residues
# We define a symmetric score I(i,j) combining both directions.
# Use arithmetic mean here; you can switch to geometric mean if desired.
inter_map = np.sqrt(attn_p * attn_d.T + 1e-12)  # (Lp, Ld)

# ---- heatmap: protein residues (rows) vs drug atoms (cols) ----
plt.figure(figsize=(8, 6))
plt.imshow(attn_p, aspect="auto", interpolation="nearest")
plt.colorbar(label="Attention weight")
plt.xlabel("Drug atoms")
plt.ylabel("Protein residues")
plt.title(f"{protein_id} Protein queries → drug atoms {drug_id}")

# x-axis: label each column by atom symbol
x_positions = np.arange(attn_p.shape[1])
plt.xticks(x_positions, atom_labels, rotation=90, fontsize=6)

# highlight chosen atom column
plt.axvline(highlight_atom_idx, linestyle="--")

plt.tight_layout()

# ---- heatmap: drug atoms (rows) vs protein residues (cols) ----
plt.figure(figsize=(8, 6))
plt.imshow(attn_d, aspect="auto", interpolation="nearest")
plt.colorbar(label="Attention weight")
plt.xlabel("Protein residues")
plt.ylabel("Drug atoms")
plt.title(f"{drug_id} Drug queries → protein residues {protein_id}")

# y-axis: label each row by atom symbol
y_positions = np.arange(attn_d.shape[0])
plt.yticks(y_positions, atom_labels, fontsize=6)

# highlight chosen atom row
plt.axhline(highlight_atom_idx, linestyle="--")

plt.tight_layout()

# ---- heatmap: symmetric interaction map (residue ↔ atom) ----
plt.figure(figsize=(8, 6))
plt.imshow(inter_map, aspect="auto", interpolation="nearest")
plt.colorbar(label="Symmetric attention score")
plt.xlabel("Drug atoms")
plt.ylabel("Protein residues")
plt.title("Symmetric residue–atom interaction map")

x_positions = np.arange(inter_map.shape[1])
plt.xticks(x_positions, atom_labels, rotation=90, fontsize=6)

# highlight chosen atom column
plt.axvline(highlight_atom_idx, linestyle="--")

plt.tight_layout()

plt.show()

#TODO: pre extract protein chains so i can use them for this purpose
import os
import re
import json
import glob
from pathlib import Path

import torch
from esm import pretrained
from esm.inverse_folding.util import load_coords, get_encoder_output

# -------------------- Config --------------------
STRUCT_ROOT = Path("structures")             # root folder containing your CIF/PDB files
EMB_ROOT = Path("embeddings")                # where to write outputs
SAVE_PER_FILE = True                          # save per-structure residue embeddings
AGGREGATE = True                              # write aggregated global embeddings
DEVICE = torch.device("cpu")                 # keep CPU for stability
POOL_STD = True                               # include std pooling (for 1024-D)

# ------------------------------------------------
EMB_ROOT.mkdir(parents=True, exist_ok=True)

# Load model + alphabet
model, alphabet = pretrained.esm_if1_gvp4_t16_142M_UR50()  # ESM-IF1
model = model.to(DEVICE).eval()

# Utility: derive an id from a structure path
# Expected names like: structures/O00141/2R5T_A.cif -> id: O00141_2R5T_A
ID_RE = re.compile(r"(?P<pdb>[0-9A-Za-z]{4})_(?P<chain>[A-Za-z0-9])\.(?:cif|pdb)$")

def derive_id(struct_path: Path) -> str:
    parts = struct_path.parts
    # try UniProt / PDB_Chain from typical structure layout
    uniprot = None
    if len(parts) >= 2 and parts[-2] and len(parts[-2]) >= 5:
        uniprot = parts[-2]
    m = ID_RE.search(struct_path.name)
    if m:
        base = f"{m.group('pdb').upper()}_{m.group('chain')}"
    else:
        base = struct_path.stem
    return f"{uniprot}_{base}" if uniprot else base

# Pooling helpers
def pool_mean(emb: torch.Tensor) -> torch.Tensor:
    return emb.mean(dim=0)

def pool_mean_std(emb: torch.Tensor) -> torch.Tensor:
    mean = emb.mean(dim=0)
    std = emb.std(dim=0, unbiased=False)
    return torch.cat([mean, std], dim=0)

# Enumerate structure files
files = sorted(
    list(STRUCT_ROOT.rglob("*.cif")) + list(STRUCT_ROOT.rglob("*.pdb"))
)
if not files:
    print(f"No structures found under {STRUCT_ROOT}/. Nothing to do.")
    raise SystemExit(0)

index = []            # metadata rows
globals_512 = []      # list of 512-D tensors
globals_1024 = []     # list of 1024-D tensors (mean+std)

for i, fpath in enumerate(files, 1):
    try:
        # Load coords + sequence (let util infer chain tokens from filename when possible)
        # For multi-chain files named like 2R5T_A.cif this is correct; if not, you can
        # force a chain by adding chain_id="A" to load_coords.
        coords, seq = load_coords(str(fpath), chain="A")
        # Extract residue embeddings [L, 512] using built-in helper
        rep = get_encoder_output(model, alphabet, coords)
        # Ensure CPU tensor
        rep = rep.detach().cpu()
        L = rep.shape[0]

        # Save per-structure embedding mirroring folder structure
        if SAVE_PER_FILE:
            rel = fpath.relative_to(STRUCT_ROOT)
            out_pt = EMB_ROOT.joinpath(rel).with_suffix(rel.suffix + ".pt")
            out_pt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(rep, out_pt)
        else:
            out_pt = None

        # Aggregate
        if AGGREGATE:
            g512 = pool_mean(rep)  # [512]
            globals_512.append(g512.unsqueeze(0))
            if POOL_STD:
                g1024 = pool_mean_std(rep)  # [1024]
                globals_1024.append(g1024.unsqueeze(0))

        # Metadata row
        index.append({
            "id": derive_id(fpath),
            "path": str(fpath),
            "L": int(L),
            "out_pt": str(out_pt) if out_pt is not None else None,
        })
        if i % 20 == 0 or i == len(files):
            print(f"Processed {i}/{len(files)}")

    except Exception as e:
        print(f"[WARN] Skipping {fpath}: {e}")
        continue

# Write aggregated tensors
if AGGREGATE and globals_512:
    G512 = torch.cat(globals_512, dim=0)  # [N, 512]
    torch.save(G512, EMB_ROOT / "all_global_512.pt")
    if POOL_STD and globals_1024:
        G1024 = torch.cat(globals_1024, dim=0)  # [N, 1024]
        torch.save(G1024, EMB_ROOT / "all_global_1024.pt")
    print(f"Wrote aggregated tensors to {EMB_ROOT}/")

# Write index metadata
with open(EMB_ROOT / "index.json", "w") as f:
    json.dump(index, f, indent=2)
print(f"Wrote index with {len(index)} entries to {EMB_ROOT/'index.json'}")

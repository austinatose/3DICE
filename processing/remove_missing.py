# remove_missing.py
# Remove rows whose drug or protein embeddings do not exist on disk

import os
import pandas as pd

INPUT_CSV = "lists/mf_val.csv"
OUTPUT_CSV = "lists/mf_val_fixed.csv"
DROPPED_CSV = "lists/mf_val_dropped.csv"

DRUG_EMBEDDING_DIR = "drug/embeddings_atomic/"
PROTEIN_EMBEDDING_DIR = "embeddings/"

DRUG_ID_COL = "drug_id"
PROTEIN_ID_COL = "uniprot_id"


# -----------------------
# Helper checks
# -----------------------

def drug_embedding_exists(drug_id: str) -> bool:
    """
    Check whether a drug embedding file exists.
    Accepts:
      - embeddings_atomic/DBxxxx
      - embeddings_atomic/DBxxxx.pt
      - embeddings_atomic/DBxxxx.npy
    """
    base = os.path.join(DRUG_EMBEDDING_DIR, drug_id)

    if os.path.isfile(base):
        return True

    for ext in [".pt", ".npy", ".npz"]:
        if os.path.isfile(base + "_unimol" + ext):
            return True

    return False


def protein_embedding_exists(protein_id: str) -> bool:
    """
    Check whether a protein embedding folder exists AND is not empty.
    """
    folder = os.path.join(PROTEIN_EMBEDDING_DIR, protein_id)

    if not os.path.isdir(folder):
        return False

    # Folder exists — check non-empty
    try:
        return len(os.listdir(folder)) > 0
    except Exception:
        return False


# -----------------------
# Main
# -----------------------

def main():
    df = pd.read_csv(INPUT_CSV)

    # Sanity check
    required = {DRUG_ID_COL, PROTEIN_ID_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n0 = len(df)

    drug_ok = df[DRUG_ID_COL].astype(str).apply(drug_embedding_exists)
    protein_ok = df[PROTEIN_ID_COL].astype(str).apply(protein_embedding_exists)

    keep_mask = drug_ok & protein_ok

    df_good = df[keep_mask].copy()
    df_bad = df[~keep_mask].copy()

    # Report
    print(f"Loaded rows: {n0}")
    print(f"Kept rows:  {len(df_good)}")
    print(f"Dropped rows: {len(df_bad)}")
    print()
    print("Drop reasons:")
    print(f"  Missing drug embedding: {(~drug_ok).sum()}")
    print(f"  Missing protein embedding: {(~protein_ok).sum()}")
    print(f"  Missing both: {((~drug_ok) & (~protein_ok)).sum()}")

    # Save
    df_good.to_csv(OUTPUT_CSV, index=False)
    df_bad.to_csv(DROPPED_CSV, index=False)

    print()
    print(f"Saved cleaned CSV → {OUTPUT_CSV}")
    print(f"Saved dropped rows → {DROPPED_CSV}")


if __name__ == "__main__":
    main()
# from the positive-interaction file, remove all very small compounds or inorganic compounds

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from pathlib import Path
INPUT_CSV = "lists/pairs_valid.csv"
OUTPUT_CSV = "lists/pairs_valid_nocomplex.csv"

# Assumes there is a separate CSV containing DrugBank IDs and SMILES strings.
# Adjust STRUCTURES_CSV and column names if your file is different.
STRUCTURES_CSV = "lists/pairs_raw.csv"  # must contain columns: drugbank_id, smiles

def load_data(interactions_path: str, structures_path: str):
    """Load positive interaction pairs and drug structures."""
    interactions_path = Path(interactions_path)
    structures_path = Path(structures_path)

    if not interactions_path.is_file():
        raise FileNotFoundError(f"Input interactions file not found: {interactions_path}")

    if not structures_path.is_file():
        raise FileNotFoundError(f"Drug structures file not found: {structures_path}")

    interactions = pd.read_csv(interactions_path)
    structures = pd.read_csv(structures_path)

    required_inter_cols = {"drugbank_id", "uniprot_id"}
    if not required_inter_cols.issubset(interactions.columns):
        missing = required_inter_cols - set(interactions.columns)
        raise KeyError(f"Missing required columns in interactions CSV: {missing}")

    required_struct_cols = {"drug_id", "smiles"}
    if not required_struct_cols.issubset(structures.columns):
        missing = required_struct_cols - set(structures.columns)
        raise KeyError(f"Missing required columns in structures CSV: {missing}")

    return interactions, structures


from rdkit.Chem import Descriptors

METALS = {3,4,11,12,19,20,21,22,23,24,25,26,27,28,29,30,47,50,53,56}

def is_inorganic_or_too_small(mol):
    if mol is None:
        return True

    # Remove metal-containing species
    if any(atom.GetAtomicNum() in METALS for atom in mol.GetAtoms()):
        return True

    # Remove very small molecules
    if Descriptors.HeavyAtomCount(mol) < 6:
        return True

    # Remove based on molecular weight
    if Descriptors.MolWt(mol) < 100:
        return True

    return False

def build_drug_filter(structures: pd.DataFrame):
    """Build a mapping of drugbank_id -> keep_flag based on SMILES heuristics."""
    drug_smiles = structures[["drug_id", "smiles"]].drop_duplicates()

    keep_flags = {}
    n_total = len(drug_smiles)
    n_invalid = 0
    n_filtered_inorganic_small = 0

    for _, row in drug_smiles.iterrows():
        dbid = row["drug_id"]
        smi = row["smiles"]

        if pd.isna(smi) or not isinstance(smi, str) or smi.strip() == "":
            # Missing or empty SMILES -> cannot use
            keep_flags[dbid] = False
            n_invalid += 1
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # RDKit could not recognize the SMILES
            keep_flags[dbid] = False
            n_invalid += 1
            continue

        if is_inorganic_or_too_small(mol):
            keep_flags[dbid] = False
            n_filtered_inorganic_small += 1
        else:
            keep_flags[dbid] = True

    print(f"Total unique drugs with SMILES: {n_total}")
    print(f"  - Invalid/unparseable SMILES: {n_invalid}")
    print(f"  - Inorganic or very small compounds filtered: {n_filtered_inorganic_small}")
    print(f"  - Kept: {sum(keep_flags.values())}")
    return keep_flags


def prune_interactions(interactions: pd.DataFrame, keep_flags: dict) -> pd.DataFrame:
    """Remove interactions involving filtered drugs.

    - Drops any rows where the drugbank_id is not in keep_flags or is marked False.
    """
    before = len(interactions)
    mask_keep = interactions["drugbank_id"].map(keep_flags).fillna(False)
    pruned = interactions[mask_keep].reset_index(drop=True)
    after = len(pruned)
    print(f"Interactions before pruning: {before}")
    print(f"Interactions after pruning:  {after}")
    print(f"Dropped {before - after} interactions involving unwanted drugs.")
    return pruned


def main():
    interactions, structures = load_data(INPUT_CSV, STRUCTURES_CSV)
    keep_flags = build_drug_filter(structures)
    pruned = prune_interactions(interactions, keep_flags)

    out_path = Path(OUTPUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pruned.to_csv(out_path, index=False)
    print(f"Pruned interactions written to: {out_path}")


if __name__ == "__main__":
    main()

"""moltrans_gen.py

Convert DrugBank pair CSVs of the form:

    uniprot_id,drug_id,interaction

into MolTrans-style CSVs with columns:

    SMILES,Target Sequence,label

"""

import argparse
import os
import time
import requests
from typing import Tuple
from tqdm import tqdm
import pandas as pd

def fetch_uniprot_sequence(uniprot_id: str, sleep: float = 0.1):
    """Fetch a protein sequence from UniProt by UniProt ID using the REST API.
    Cached locally to avoid repeated API calls.
    Returns the amino-acid sequence string or None on failure.
    """
    # Cache directory
    cache_dir = os.path.join(os.path.dirname(__file__), "uniprot_fasta")

    cache_path = os.path.join(cache_dir, f"{uniprot_id}.fasta")

    # 1. Use cache if present
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                lines = f.read().splitlines()
            seq_lines = [ln.strip() for ln in lines if ln and not ln.startswith(">")]
            seq = "".join(seq_lines)
            if seq:
                print(f"[moltrans_gen] Using cached sequence for {uniprot_id}")
                return seq
        except Exception:
            pass  # fall back to API fetch

    # 2. Fetch from UniProt API
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        r = requests.get(url, timeout=10)
    except Exception:
        return None
    if r.status_code != 200:
        return None

    lines = r.text.splitlines()
    seq_lines = [ln.strip() for ln in lines if ln and not ln.startswith(">")] 
    seq = "".join(seq_lines)

    # Save to cache
    try:
        with open(cache_path, "w") as f:
            f.write(r.text)
    except Exception:
        pass

    if sleep > 0:
        time.sleep(sleep)

    return seq or None


def fetch_proteins_from_uniprot(uniprot_ids):
    cache_csv = os.path.join(os.path.dirname(__file__), "cache", "uniprot_sequences.csv")
    os.makedirs(os.path.dirname(cache_csv), exist_ok=True)

    # Load cache if exists
    cache_df = None
    if os.path.exists(cache_csv):
        try:
            cache_df = pd.read_csv(cache_csv)
        except Exception:
            cache_df = None

    cache_map = {}
    if cache_df is not None:
        for _, row in cache_df.iterrows():
            cache_map[str(row["uniprot_id"])] = str(row["sequence"])

    records = []
    updated_cache = []
    seen = set()

    for uid in tqdm(uniprot_ids):
        uid = str(uid)
        if uid in seen:
            continue
        seen.add(uid)

        # Use CSV cache if present
        if uid in cache_map:
            records.append({"uniprot_id": uid, "sequence": cache_map[uid]})
            updated_cache.append({"uniprot_id": uid, "sequence": cache_map[uid]})
            continue

        # Otherwise fetch
        seq = fetch_uniprot_sequence(uid)
        if seq is None:
            print(f"[moltrans_gen] WARN: could not fetch sequence for {uid}")
            continue

        records.append({"uniprot_id": uid, "sequence": seq})
        updated_cache.append({"uniprot_id": uid, "sequence": seq})

    # Write updated cache
    try:
        pd.DataFrame(updated_cache).drop_duplicates(subset=["uniprot_id"]).to_csv(cache_csv, index=False)
    except Exception:
        pass

    if not records:
        raise RuntimeError("No protein sequences could be fetched from UniProt.")

    return pd.DataFrame(records)


def load_pairs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"uniprot_id", "drug_id", "interaction"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    # keep at most one row per (uniprot_id, drug_id, interaction)
    df = df.drop_duplicates(subset=["uniprot_id", "drug_id", "interaction"])
    return df


def load_drugs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"drug_id", "smiles"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    # one smiles per drug_id
    df = df.drop_duplicates(subset=["drug_id"])
    return df[["drug_id", "smiles"]]


def load_proteins_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"uniprot_id", "sequence"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df = df.drop_duplicates(subset=["uniprot_id"])
    return df[["uniprot_id", "sequence"]]


def to_moltrans_format(
    pairs_csv: str,
    drug_csv: str,
    prot_csv: str,
    out_csv: str,
) -> Tuple[int, int]:
    """Convert one pairs CSV into MolTrans-style CSV.

    Returns
    -------
    kept : int
        Number of rows with both SMILES and sequence present.
    dropped : int
        Number of rows dropped due to missing SMILES or sequence.
    """

    print(f"[moltrans_gen] Reading pairs from {pairs_csv}")
    pairs = load_pairs(pairs_csv)

    print(f"[moltrans_gen] Reading drugs from {drug_csv}")
    drugs = load_drugs(drug_csv)

    if prot_csv:
        print(f"[moltrans_gen] Reading proteins from {prot_csv}")
        prots = load_proteins_from_csv(prot_csv)
    else:
        uniq_uniprot = sorted(pairs["uniprot_id"].astype(str).unique().tolist())
        print(f"[moltrans_gen] Fetching {len(uniq_uniprot)} unique UniProt sequences from API")
        prots = fetch_proteins_from_uniprot(uniq_uniprot)

    # Merge pairs with SMILES and sequences
    df = pairs.merge(drugs, on="drug_id", how="left")
    df = df.merge(prots, on="uniprot_id", how="left")

    before = len(df)
    df = df.dropna(subset=["smiles", "sequence"])
    after = len(df)
    dropped = before - after

    if dropped > 0:
        print(f"[moltrans_gen] Dropped {dropped} rows with missing SMILES or sequence (kept {after}).")
    else:
        print(f"[moltrans_gen] Kept all {after} rows.")

    df = df.drop_duplicates(subset=["drug_id", "uniprot_id", "interaction"])
    
    # Construct MolTrans-style columns
    out_df = pd.DataFrame({
        "SMILES": df["smiles"].astype(str),
        "Target Sequence": df["sequence"].astype(str),
        "Label": df["interaction"].astype(int),
    })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[moltrans_gen] Wrote MolTrans CSV with {len(out_df)} rows to {out_csv}")

    return len(out_df), dropped


def main():
    ap = argparse.ArgumentParser(description="Convert DrugBank pairs into MolTrans-style CSV.")
    ap.add_argument("--pairs_csv", required=True, help="CSV with columns: uniprot_id,drug_id,interaction")
    ap.add_argument("--drug_csv", required=True, help="CSV with columns: drug_id,smiles")
    ap.add_argument("--prot_csv", default="", help="Optional CSV with columns: uniprot_id,sequence. If omitted, sequences are fetched from UniProt API.")
    ap.add_argument("--out_csv", required=True, help="Output MolTrans-style CSV: SMILES,Target Sequence,label")
    args = ap.parse_args()

    to_moltrans_format(
        pairs_csv=args.pairs_csv,
        drug_csv=args.drug_csv,
        prot_csv=args.prot_csv,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()
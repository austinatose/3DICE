
import argparse
import os
import sys
import time
import json
from typing import Dict, List, Optional

import requests
import pandas as pd
from tqdm import tqdm

PDBe_SIFTS_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{}"
PDBe_SUMMARY_URL = "https://www.ebi.ac.uk/pdbe/api/pdb/entry/summary/{}"   # lowercase pdb id ok
RCSB_DOWNLOAD_MMCIF = "https://files.rcsb.org/download/{}.cif"
RCSB_DOWNLOAD_PDB   = "https://files.rcsb.org/download/{}.pdb"
AFDB_PDB_URL        = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb"

HEADERS = {"User-Agent": "structure-fetcher/1.0 (DTI pipeline)"}

def safe_get_json(url: str, retries: int = 3, backoff: float = 0.8):
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(backoff * (2 ** i))
    return None

def fetch_pdbe_sifts(uniprot: str):
    """Return mapping dict from PDBe SIFTS for a UniProt accession or None."""
    return safe_get_json(PDBe_SIFTS_URL.format(uniprot))

def fetch_pdbe_summary(pdb_id: str):
    """Return simple metadata (resolution, method) for a PDB entry from PDBe."""
    js = safe_get_json(PDBe_SUMMARY_URL.format(pdb_id.lower()))
    # Expected format: {"1abc": [{"resolution": 1.8, "experimental_method": "X-ray diffraction", ...}]}
    if not js:
        return None
    key = list(js.keys())[0]
    arr = js.get(key, [])
    return arr[0] if arr else None

def parse_sifts_coverage(uniprot: str, sifts_js: dict) -> List[dict]:
    """
    Extract a flat list of {pdb_id, chain_id, unp_start, unp_end, pdb_start, pdb_end, coverage}
    where coverage is (unp_end - unp_start + 1) / UniProt length if available (approx).
    """
    out = []
    if not sifts_js or uniprot not in sifts_js:
        return out
    root = sifts_js[uniprot]
    # UniProt sequence length (if present)
    unp_len = None
    try:
        unp_len = int(root.get("UniProt", {}).get(uniprot, {}).get("length", None))
    except Exception:
        unp_len = None
    pdb_map = root.get("PDB", {})
    for pdb_id, entry in pdb_map.items():
        # PDBe may expose 'mappings' (current) or legacy 'chains'
        chains = entry.get("mappings", []) or entry.get("chains", [])
        for ch in chains:
            chain_id = ch.get("chain_id") or ch.get("struct_asym_id")
            unp_start = ch.get("unp_start")
            unp_end = ch.get("unp_end")
            pdb_start = ch.get("start")
            pdb_end = ch.get("end")
            if unp_start is None or unp_end is None:
                continue
            cov = None
            if unp_len and unp_len > 0:
                cov = (int(unp_end) - int(unp_start) + 1) / float(unp_len)
            out.append({
                "pdb_id": pdb_id.upper(),
                "chain_id": chain_id,
                "unp_start": unp_start,
                "unp_end": unp_end,
                "pdb_start": pdb_start,
                "pdb_end": pdb_end,
                "coverage": cov
            })
    return out

def choose_best_mapping(rows: List[dict], min_coverage: float = 0.2) -> Optional[dict]:
    """
    Choose a single (pdb_id, chain) mapping with preference:
      1) coverage >= min_coverage
      2) method preference: X-ray or EM over NMR/model
      3) lowest resolution value preferred (for X-ray); any EM okay
      4) otherwise, highest coverage
    """
    if not rows:
        return None
    # Deduplicate by pdb_id, chain_id taking max coverage
    keyd = {}
    for r in rows:
        k = (r["pdb_id"], r["chain_id"])
        if k not in keyd or (r.get("coverage") or 0) > (keyd[k].get("coverage") or 0):
            keyd[k] = r
    cand = list(keyd.values())
    # Pull PDBe summary for each unique PDB id once
    summary_cache = {}
    for r in cand:
        pid = r["pdb_id"]
        if pid not in summary_cache:
            meta = fetch_pdbe_summary(pid) or {}
            summary_cache[pid] = {
                "method": (meta.get("experimental_method") or "").lower(),
                "resolution": meta.get("resolution", None)
            }
        r["method"] = summary_cache[pid]["method"]
        r["resolution"] = summary_cache[pid]["resolution"]
    # Filter by coverage
    filt = [r for r in cand if (r.get("coverage") or 0) >= min_coverage]
    if not filt:
        filt = cand  # fall back to any
    # Sort: prefer x-ray/cryo-em; then lower resolution; then higher coverage
    def score(r):
        method = r.get("method","")
        res = r.get("resolution", 1e9) or 1e9
        cov = r.get("coverage") or 0.0
        if "x-ray" in method:
            mscore = 2
        elif "electron" in method or "cryo" in method:
            mscore = 1
        else:
            mscore = 0
        return (-mscore, res, -cov)
    filt.sort(key=score)
    return filt[0] if filt else None

def download_coords(pdb_id: str, fmt: str, out_path: str) -> bool:
    url = RCSB_DOWNLOAD_MMCIF.format(pdb_id.upper()) if fmt == "mmcif" else RCSB_DOWNLOAD_PDB.format(pdb_id.upper())
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        if r.status_code == 200 and len(r.content) > 100:
            with open(out_path, "wb") as f:
                f.write(r.content)
            return True
    except requests.RequestException:
        pass
    return False

def download_alphafold(uniprot: str, out_path: str) -> bool:
    url = AFDB_PDB_URL.format(uniprot)
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        if r.status_code == 200 and len(r.content) > 100:
            with open(out_path, "wb") as f:
                f.write(r.content)
            return True
    except requests.RequestException:
        pass
    return False

def process_uniprot(uniprot: str, out_dir: str, fmt: str, min_cov: float, af_fallback: bool) -> dict:
    os.makedirs(os.path.join(out_dir, uniprot), exist_ok=True)
    record = {
        "uniprot": uniprot,
        "source": None,
        "pdb_id": None,
        "chain_id": None,
        "method": None,
        "resolution": None,
        "coverage": None,
        "file_path": None,
        "note": None
    }
    sifts = fetch_pdbe_sifts(uniprot)
    rows = parse_sifts_coverage(uniprot, sifts) if sifts else []
    choice = choose_best_mapping(rows, min_cov) if rows else None
    if choice:
        pdb_id = choice["pdb_id"]
        chain_id = choice.get("chain_id")
        ext = "cif" if fmt == "mmcif" else "pdb"
        fpath = os.path.join(out_dir, uniprot, f"{pdb_id}_{chain_id}.{ext}")
        ok = download_coords(pdb_id, fmt, fpath)
        if ok:
            record.update({
                "source": "pdb",
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "method": choice.get("method"),
                "resolution": choice.get("resolution"),
                "coverage": choice.get("coverage"),
                "file_path": fpath
            })
            return record
        else:
            record["note"] = f"Failed to download {fmt.upper()} for {pdb_id}"
    # Fallback to AlphaFold
    if af_fallback:
        fpath = os.path.join(out_dir, uniprot, f"AF-{uniprot}-F1-model_v4.pdb")
        ok = download_alphafold(uniprot, fpath)
        if ok:
            record.update({
                "source": "alphafold",
                "file_path": fpath
            })
            return record
        else:
            record["note"] = (record.get("note") + "; " if record.get("note") else "") + "AlphaFold download failed"
    return record

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="CSV with a column of UniProt accessions")
    ap.add_argument("--uniprot_col", default="uniprot", help="Column name holding UniProt IDs")
    ap.add_argument("--out_dir", required=True, help="Folder to store structures and manifest")
    ap.add_argument("--format", choices=["mmcif","pdb"], default="mmcif", help="Download format for PDB entries")
    ap.add_argument("--min_coverage", type=float, default=0.2, help="Minimum UniProt coverage to accept a PDB chain")
    ap.add_argument("--alphafold_fallback", type=int, default=1, help="Use AlphaFold if no PDB meets criteria (1=yes,0=no)")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.uniprot_col not in df.columns:
        for alt in ["UniProt", "uniprot_id", "uniprot_acc", "uniprot_accession"]:
            if alt in df.columns:
                args.uniprot_col = alt
                break
    if args.uniprot_col not in df.columns:
        print(f"Could not find UniProt column. Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.out_dir, exist_ok=True)

    records = []
    for acc in tqdm(df[args.uniprot_col].astype(str).tolist(), desc="Fetching structures"):
        acc = acc.strip()
        if not acc or acc.lower() == "nan":
            continue
        rec = process_uniprot(acc, args.out_dir, args.format, args.min_coverage, bool(args.alphafold_fallback))
        records.append(rec)

    manifest = pd.DataFrame(records)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(records, f, indent=2)

    n_pdb = sum(1 for r in records if r.get("source") == "pdb")
    n_af = sum(1 for r in records if r.get("source") == "alphafold")
    n_err = sum(1 for r in records if not r.get("file_path"))
    print(f"Done. PDB: {n_pdb} | AlphaFold: {n_af} | Failed: {n_err}")
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    main()

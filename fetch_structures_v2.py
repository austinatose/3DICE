
import argparse
import os
import sys
import time
import json
from typing import Dict, List, Optional

import requests
import pandas as pd
from tqdm import tqdm

# --- Endpoints (new + legacy + fallbacks) ---
PDBe_V2_SIFTS = "https://www.ebi.ac.uk/pdbe/api/v2/mappings/uniprot/{}"      # NEW (2025)
PDBe_LEGACY_SIFTS = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{}"      # Legacy
PDBe_ENTRY_SUMMARY = "https://www.ebi.ac.uk/pdbe/api/pdb/entry/summary/{}"    # entry metadata

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTRY_CORE = "https://data.rcsb.org/rest/v1/core/entry/{}"               # resolution/method
RCSB_DOWNLOAD_MMCIF = "https://files.rcsb.org/download/{}.cif"
RCSB_DOWNLOAD_PDB   = "https://files.rcsb.org/download/{}.pdb"

AFDB_FILE_TMPLS = [
    "https://alphafold.ebi.ac.uk/files/AF-{}-{}-model_v4.pdb",  # F1 v4
    "https://alphafold.ebi.ac.uk/files/AF-{}-{}-model_v3.pdb",  # F1 v3
]

HEADERS = {"User-Agent": "evidti-structure-fetcher/1.1"}

def _get_json(url: str, retries: int = 2, timeout: int = 20):
    err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception as e:
            err = e
        time.sleep(0.6 * (2 ** i))
    return None

def _post_json(url: str, payload: dict, retries: int = 2, timeout: int = 25):
    err = None
    for i in range(retries):
        try:
            r = requests.post(url, json=payload, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            err = e
        time.sleep(0.6 * (2 ** i))
    return None

# ---------- PDBe SIFTS parsing ----------
def fetch_pdbe_mappings(uniprot: str) -> Optional[dict]:
    """Try PDBe v2 first, then legacy SIFTS."""
    js = _get_json(PDBe_V2_SIFTS.format(uniprot))
    if js is None:
        js = _get_json(PDBe_LEGACY_SIFTS.format(uniprot))
    return js

def parse_pdbe_rows(uniprot: str, sifts_js: dict) -> List[dict]:
    """Flatten mappings into rows with coverage."""
    rows = []
    if not sifts_js:
        return rows
    # v2 packs under top-level 'data' OR by uniprot accession; legacy under uniprot key
    root = sifts_js.get(uniprot) or sifts_js.get("data", {}).get(uniprot) or sifts_js
    if not root:
        return rows
    # uniprot length if provided
    unp_len = None
    try:
        unp_len = int((root.get("UniProt", {}) or {}).get(uniprot, {}).get("length", None))
    except Exception:
        unp_len = None

    pdb_map = root.get("PDB", {}) or {}
    for pdb_id, entry in pdb_map.items():
        chains = entry.get("mappings", []) or entry.get("chains", []) or []
        for ch in chains:
            chain_id = ch.get("chain_id") or ch.get("struct_asym_id")
            unp_start = ch.get("unp_start"); unp_end = ch.get("unp_end")
            pdb_start = ch.get("start");     pdb_end = ch.get("end")
            if unp_start is None or unp_end is None:
                continue
            cov = None
            if unp_len and unp_len > 0:
                cov = (int(unp_end) - int(unp_start) + 1) / float(unp_len)
            rows.append({
                "pdb_id": pdb_id.upper(),
                "chain_id": chain_id,
                "unp_start": unp_start, "unp_end": unp_end,
                "pdb_start": pdb_start, "pdb_end": pdb_end,
                "coverage": cov
            })
    return rows

# ---------- RCSB fallback search ----------
def rcsb_polymer_entities_for_uniprot(uniprot: str) -> List[dict]:
    """Return list of polymer entities (entry_id + entity_id) mapped to a UniProt accession."""
    payload = {
      "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
          {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
              "operator": "exact_match",
              "value": uniprot
            }
          },
          {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
              "operator": "exact_match",
              "value": "UniProt"
            }
          }
        ]
      },
      "return_type": "polymer_entity",
      "request_options": {"results_content_type": ["experimental"]}  # exclude computed by default
    }
    js = _post_json(RCSB_SEARCH, payload)
    out = []
    if not js:
        return out
    for it in js.get("result_set", []):
        identifier = it.get("identifier")  # format: "XXXX_entity_id" e.g., "4HHB_1"
        if not identifier or "_" not in identifier:
            continue
        pdb_id, entity_id = identifier.split("_", 1)
        out.append({"pdb_id": pdb_id.upper(), "entity_id": entity_id})
    return out

def rcsb_entry_meta(pdb_id: str) -> dict:
    js = _get_json(RCSB_ENTRY_CORE.format(pdb_id))
    if not js:
        return {}
    # Extract method and resolution if present
    method = (js.get("exptl", [{}])[0].get("method") or js.get("rcsb_entry_info", {}).get("experimental_method")) or ""
    res = js.get("rcsb_entry_info", {}).get("resolution_combined", None)
    if isinstance(res, list) and res:
        res = res[0]
    return {"method": str(method).lower(), "resolution": res}

# ---------- Selection & download ----------
def choose_best(rows: List[dict], min_cov: float = 0.2) -> Optional[dict]:
    if not rows:
        return None
    # Dedup per (pdb, chain)
    uniq = {}
    for r in rows:
        k = (r["pdb_id"], r.get("chain_id"))
        if k not in uniq or (r.get("coverage") or 0) > (uniq[k].get("coverage") or 0):
            uniq[k] = r
    cands = list(uniq.values())
    # enrich with method/resolution
    meta_cache = {}
    for r in cands:
        pid = r["pdb_id"]
        if pid not in meta_cache:
            meta_cache[pid] = rcsb_entry_meta(pid)
        r.update(meta_cache[pid])
    # filter by coverage
    filt = [r for r in cands if (r.get("coverage") or 0) >= min_cov] or cands
    def key(r):
        method = r.get("method","")
        res = r.get("resolution", 1e9) or 1e9
        cov = r.get("coverage") or 0.0
        mscore = 2 if "x-ray" in method else (1 if "electron" in method or "cryo" in method else 0)
        return (-mscore, res, -cov)
    return sorted(filt, key=key)[0]

def download_coords(pdb_id: str, fmt: str, out_path: str) -> bool:
    url = RCSB_DOWNLOAD_MMCIF.format(pdb_id) if fmt == "mmcif" else RCSB_DOWNLOAD_PDB.format(pdb_id)
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        if r.status_code == 200 and len(r.content) > 200:
            with open(out_path, "wb") as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    return False

def download_alphafold(uniprot: str, isoform: str, out_path: str) -> bool:
    idx = isoform or "F1"
    for tmpl in AFDB_FILE_TMPLS:
        url = tmpl.format(uniprot, idx)
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 200 and len(r.content) > 200:
                with open(out_path, "wb") as f:
                    f.write(r.content)
                return True
        except Exception:
            pass
    return False

# ---------- Orchestration ----------
def process_one(uniprot: str, out_dir: str, fmt: str, min_cov: float, af_fallback: bool, af_isoform: str) -> dict:
    os.makedirs(os.path.join(out_dir, uniprot), exist_ok=True)
    rec = {
        "uniprot": uniprot, "source": None, "pdb_id": None, "chain_id": None,
        "method": None, "resolution": None, "coverage": None, "file_path": None, "note": None
    }
    # PDBe route
    sifts = fetch_pdbe_mappings(uniprot)
    rows = parse_pdbe_rows(uniprot, sifts) if sifts else []
    # If PDBe failed, try RCSB search to at least get PDB IDs
    if not rows:
        ents = rcsb_polymer_entities_for_uniprot(uniprot)
        rows = [{"pdb_id": e["pdb_id"], "chain_id": None, "coverage": None} for e in ents]
    choice = choose_best(rows, min_cov) if rows else None
    if choice:
        pid = choice["pdb_id"]; ch = choice.get("chain_id")
        ext = "cif" if fmt == "mmcif" else "pdb"
        fpath = os.path.join(out_dir, uniprot, f"{pid}{'_'+ch if ch else ''}.{ext}")
        if download_coords(pid, fmt, fpath):
            rec.update({
                "source": "pdb", "pdb_id": pid, "chain_id": ch,
                "method": choice.get("method"), "resolution": choice.get("resolution"),
                "coverage": choice.get("coverage"), "file_path": fpath
            })
            return rec
        rec["note"] = f"Failed to download {fmt} for {pid}"
    # AlphaFold fallback
    if af_fallback:
        fpath = os.path.join(out_dir, uniprot, f"AF-{uniprot}-{af_isoform or 'F1'}-model.pdb")
        if download_alphafold(uniprot, af_isoform, fpath):
            rec.update({"source": "alphafold", "file_path": fpath})
            return rec
        rec["note"] = (rec["note"] + "; " if rec["note"] else "") + "AlphaFold download failed"
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--uniprot_col", default="uniprot")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--format", choices=["mmcif","pdb"], default="mmcif")
    ap.add_argument("--min_coverage", type=float, default=0.2)
    ap.add_argument("--alphafold_fallback", type=int, default=1)
    ap.add_argument("--alphafold_isoform", default="F1", help="e.g., F1/F2 if you need a specific isoform")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.uniprot_col not in df.columns:
        for alt in ["UniProt","uniprot_id","uniprot_acc","uniprot_accession"]:
            if alt in df.columns:
                args.uniprot_col = alt
                break
    if args.uniprot_col not in df.columns:
        print(f"Column '{args.uniprot_col}' not found. Available: {list(df.columns)}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.out_dir, exist_ok=True)
    recs = []
    for acc in tqdm(df[args.uniprot_col].astype(str), desc="Fetching"):
        acc = acc.strip()
        if not acc or acc.lower() == "nan":
            continue
        r = process_one(acc, args.out_dir, args.format, args.min_coverage, bool(args.alphafold_fallback), args.alphafold_isoform)
        recs.append(r)

    man = pd.DataFrame(recs)
    man_path = os.path.join(args.out_dir, "manifest.csv")
    man.to_csv(man_path, index=False)
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(recs, f, indent=2)

    print(f"Saved manifest to {man_path}")
    print(f"PDB: {sum(1 for r in recs if r.get('source')=='pdb')}, AF: {sum(1 for r in recs if r.get('source')=='alphafold')}, Fail: {sum(1 for r in recs if not r.get('file_path'))}")

if __name__ == "__main__":
    main()



"""biotite_mapping.py

Map UniProt/FASTA indices to the *structure-derived* residue sequence used by ESM-IF1.

Key idea:
- `esm.inverse_folding.util.load_coords()` returns (coords, seq_struct)
  where seq_struct is the residue sequence *after* structure filtering
  (chain selection, amino-acids, and only residues with backbone coords).
- Your ESM-IF1 embeddings are indexed by this seq_struct.
- To relate embedding/coords indices to the original FASTA indices, align:
      seq_fasta  <->  seq_struct
  and build index maps.

Outputs per structure:
- JSON with arrays: fasta_to_struct (len=|FASTA|) and struct_to_fasta (len=|STRUCT|)
- CSV with row-wise mapping for convenience

Assumptions (edit as needed):
- Structure files live under `structures/` (same as your embed script)
- FASTA files live under `fastas/` with filename `{uniprot_id}.fasta` or `{uniprot_id}.fa`
- UniProt ID is inferred from the parent folder name (same as derive_id() in your embed script)

Run examples:
  python interpretation/biotite_mapping.py \
      --struct-root structures \
      --fasta-root fastas \
      --out-root interpretation/mappings

  # Optionally restrict to one UniProt ID:
  python interpretation/biotite_mapping.py --only-id O00141

"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from esm.inverse_folding.util import load_coords

import biotite.sequence as bseq
import biotite.sequence.align as balign
from biotite.sequence.align import SubstitutionMatrix


def derive_uniprot_id(struct_path: Path) -> str:
    """Return the UniProt ID (parent folder name) for indexing.

    Mirrors your embed code: structures/O00141/2R5T_A.cif -> O00141
    """
    parts = struct_path.parts
    return parts[-2] if len(parts) >= 2 else struct_path.stem


def infer_chain(struct_path: Path) -> str:
    """Infer chain exactly like your embedding script."""
    name = struct_path.name
    if struct_path.suffix == ".cif" and "AF-" in name:
        return "A"
    m = re.search(r"_(?P<chain>[A-Za-z0-9]+)\.(?:cif|pdb)$", name)
    return m.group("chain") if m else "A"


def read_fasta(path: Path) -> str:
    """Read a FASTA file and return the concatenated sequence string."""
    seq_lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_lines.append(line)
    seq = "".join(seq_lines).strip().upper()
    if not seq:
        raise ValueError(f"Empty FASTA sequence: {path}")
    return seq


def find_fasta_for_id(fasta_root: Path, uniprot_id: str) -> Optional[Path]:
    """Find a FASTA file for a UniProt ID under fasta_root."""
    candidates = [
        fasta_root / f"{uniprot_id}.fasta",
        fasta_root / f"{uniprot_id}.fa",
        fasta_root / f"{uniprot_id}.faa",
        fasta_root / f"{uniprot_id}.txt",
    ]
    for p in candidates:
        if p.exists():
            return p

    # fallback: search by stem match (case-insensitive)
    for p in fasta_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".fasta", ".fa", ".faa", ".txt"}:
            if p.stem.upper() == uniprot_id.upper():
                return p
    return None


def align_fasta_to_struct(seq_fasta: str, seq_struct: str) -> Tuple[List[Optional[int]], List[Optional[int]], str, str]:
    """Align seq_fasta to seq_struct and return index maps.

    Returns:
      fasta_to_struct: list length |FASTA|, mapping FASTA index -> struct index (or None)
      struct_to_fasta: list length |STRUCT|, mapping struct index -> FASTA index (or None)
      fasta_aln: gapped FASTA alignment string
      struct_aln: gapped STRUCT alignment string
    """
    # Biotite alignment objects
    s_fasta = bseq.ProteinSequence(seq_fasta)
    s_struct = bseq.ProteinSequence(seq_struct)

    # Standard protein substitution matrix
    matrix = SubstitutionMatrix.std_protein_matrix()

    # Structure-derived sequence is typically a subsequence of FASTA with gaps,
    # so we avoid penalizing terminal gaps too harshly.
    ali = balign.align_optimal(
        s_fasta,
        s_struct,
        matrix,
        gap_penalty=-10,
        terminal_penalty=False,
    )[0]

    # print(ali)

    fasta_aln, struct_aln = ali.get_gapped_sequences()

    fasta_to_struct: List[Optional[int]] = [None] * len(seq_fasta)
    struct_to_fasta: List[Optional[int]] = [None] * len(seq_struct)

    i_f = -1  # fasta index
    i_s = -1  # struct index
    for a, b in zip(fasta_aln, struct_aln):
        if a != "-":
            i_f += 1
        if b != "-":
            i_s += 1
        if a != "-" and b != "-":
            fasta_to_struct[i_f] = i_s
            struct_to_fasta[i_s] = i_f

    return fasta_to_struct, struct_to_fasta, fasta_aln, struct_aln


def write_csv(path: Path, seq_fasta: str, seq_struct: str, fasta_to_struct: List[Optional[int]], struct_to_fasta: List[Optional[int]]):
    """Write a row-wise CSV mapping for easy inspection."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)

    # Build inverse lookup (struct -> fasta already provided)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "struct_idx",
            "struct_aa",
            "fasta_idx",
            "fasta_aa",
        ])
        for s_idx, f_idx in enumerate(struct_to_fasta):
            s_aa = seq_struct[s_idx]
            if f_idx is None:
                w.writerow([s_idx, s_aa, "", ""])
            else:
                f_aa = seq_fasta[f_idx]
                w.writerow([s_idx, s_aa, f_idx, f_aa])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct-root", type=Path, default=Path("structures"), help="Root folder containing CIF/PDB files")
    ap.add_argument("--fasta-root", type=Path, default=Path("other/uniprot_fasta"), help="Folder containing FASTA files named by UniProt ID")
    ap.add_argument("--out-root", type=Path, default=Path("interpretation/mappings"), help="Output folder for mapping files")
    ap.add_argument("--only-id", type=str, default=None, help="Only process this UniProt ID")
    ap.add_argument("--max-files", type=int, default=None, help="Optional cap for debugging")
    args = ap.parse_args()

    struct_root: Path = args.struct_root
    fasta_root: Path = args.fasta_root
    out_root: Path = args.out_root

    files = sorted(list(struct_root.rglob("*.cif")) + list(struct_root.rglob("*.pdb")))
    if args.only_id:
        files = [p for p in files if derive_uniprot_id(p).upper() == args.only_id.upper()]
    if args.max_files is not None:
        files = files[: args.max_files]

    if not files:
        raise SystemExit(f"No structure files found under {struct_root}/ (after filters).")

    out_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "struct_root": str(struct_root),
        "fasta_root": str(fasta_root),
        "out_root": str(out_root),
        "processed": 0,
        "skipped_no_fasta": 0,
        "skipped_error": 0,
        "items": [],
    }

    for fpath in tqdm(files, desc="Mapping FASTA↔STRUCT"):
        uid = derive_uniprot_id(fpath)
        chain = infer_chain(fpath)

        fasta_path = find_fasta_for_id(fasta_root, uid)
        if fasta_path is None:
            summary["skipped_no_fasta"] += 1
            summary["items"].append({
                "id": uid,
                "path": str(fpath),
                "chain": chain,
                "status": "no_fasta",
            })
            continue

        try:
            # Structure-derived sequence used by ESM-IF1 embeddings
            coords, seq_struct = load_coords(str(fpath), chain=chain)
            # coords is (L, 3, 3); seq_struct length should be L
            L = int(coords.shape[0])
            if len(seq_struct) != L:
                # This should not happen, but keep it explicit
                raise ValueError(f"len(seq_struct)={len(seq_struct)} != coords L={L}")

            seq_fasta = read_fasta(fasta_path)

            fasta_to_struct, struct_to_fasta, fasta_aln, struct_aln = align_fasta_to_struct(seq_fasta, seq_struct)

            # Write outputs mirroring structure path
            rel = fpath.relative_to(struct_root)
            base = out_root.joinpath(rel).with_suffix("")  # drop .cif/.pdb

            out_json = base.with_suffix(".map.json")
            out_csv = base.with_suffix(".map.csv")

            out_json.parent.mkdir(parents=True, exist_ok=True)

            payload = {
                "id": uid,
                "chain": chain,
                "structure_path": str(fpath),
                "fasta_path": str(fasta_path),
                "L_struct": len(seq_struct),
                "L_fasta": len(seq_fasta),
                "seq_struct": seq_struct,
                # For large proteins, you may want to omit seq_fasta to keep JSON small
                "seq_fasta": seq_fasta,
                "fasta_to_struct": fasta_to_struct,
                "struct_to_fasta": struct_to_fasta,
                "fasta_aln": fasta_aln,
                "struct_aln": struct_aln,
            }
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            write_csv(out_csv, seq_fasta, seq_struct, fasta_to_struct, struct_to_fasta)

            summary["processed"] += 1
            summary["items"].append({
                "id": uid,
                "path": str(fpath),
                "chain": chain,
                "status": "ok",
                "L_struct": len(seq_struct),
                "L_fasta": len(seq_fasta),
                "out_json": str(out_json),
                "out_csv": str(out_csv),
            })

        except Exception as e:
            summary["skipped_error"] += 1
            summary["items"].append({
                "id": uid,
                "path": str(fpath),
                "chain": chain,
                "status": "error",
                "error": str(e),
            })
            continue

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Done. processed={summary['processed']} skipped_no_fasta={summary['skipped_no_fasta']} skipped_error={summary['skipped_error']}\n"
        f"Wrote {out_root/'summary.json'}"
    )


if __name__ == "__main__":
    main()
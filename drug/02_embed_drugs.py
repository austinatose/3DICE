import numpy as np
from unimol_tools import UniMolRepr
import multiprocessing as mp

CSV_PATH = "drug/drug_targets.csv"
OUTPUT_DIR = "drug/embeddings"

def main():
    # single smiles unimol representation
    clf = UniMolRepr(data_type='molecule', remove_hs=False)
    smiles = 'CC[C@H](C)[C@H](NC(=O)[C@H](CCC(O)=O)NC(=O)[C@H](CCC(O)=O)NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)[C@H](CC(O)=O)NC(=O)CNC(=O)[C@H](CC(N)=O)NC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCNC(N)=N)NC(=O)[C@@H]1CCCN1C(=O)[C@H](N)CC1=CC=CC=C1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(O)=O)C(=O)N[C@@H](CCC(O)=O)C(=O)N[C@@H](CC1=CC=C(O)C=C1)C(=O)N[C@@H](CC(C)C)C(O)=O'
    smiles_list = [smiles]
    unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=True)
    # CLS token repr
    print(np.array(unimol_repr['cls_repr']).shape)
    # atomic level repr, align with rdkit mol.GetAtoms()
    print(np.array(unimol_repr['atomic_reprs']))

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

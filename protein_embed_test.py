import torch
from esm import pretrained
from esm.inverse_folding.util import load_coords, CoordBatchConverter, get_encoder_output

# --- Add mappings for common modified residues so biotite can convert 3-letter to 1-letter ---
try:
    from biotite.sequence.seqtypes import ProteinSequence
    _m = ProteinSequence._dict_3to1
    # Map frequent PTMs/backbone subs to their canonical parents
    for k, v in {
        "MSE": "M",   # Selenomethionine → Met
        "SEP": "S",   # Phosphoserine → Ser
        "TPO": "T",   # Phosphothreonine → Thr
        "PTR": "Y",   # Phosphotyrosine → Tyr
        "CSO": "C",   # Oxidized Cys → Cys
        "HIP": "H",   # Protonated His → His
        "HID": "H",   # His delta-protonated → His
        "HIE": "H",   # His epsilon-protonated → His
    }.items():
        if k not in _m:
            _m[k] = v
except Exception:
    pass
# -----------------------------------------------------------------------------

# Select device (MPS if available, else CPU)
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print(torch.ones(1, device=device))
# else:
#     print("MPS device not found.")
#     device = torch.device("cpu")

device = torch.device("cpu")

# Load model + alphabet
model, alphabet = pretrained.esm_if1_gvp4_t16_142M_UR50() ## ESM-IF1
model = model.to(device).eval()

coords, seq = load_coords("structures/O00418/8GM4_A.cif", chain="A")  # temp

rep = get_encoder_output(model, alphabet, coords)  # may allocate on MPS

# save embeddings
torch.save(rep, "test.pt")
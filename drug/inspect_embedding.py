import torch
import numpy as np

# this is data, i don't want to load as weights
reps = torch.load("drug/embeddings_atomic/DB00130_unimol.pt", weights_only=False)
print(reps.keys())
print(np.array(reps["atomic_reprs"]))  # (N_atoms, 512)
print(np.array(reps["atomic_reprs"]).shape)  # (N_atoms, 512)
print(np.array(reps["atomic_symbol"]))  # (N_atoms, 512)
print(np.array(reps["atomic_symbol"]).shape)  # (N_atoms, 512)
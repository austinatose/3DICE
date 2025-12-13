import os
import re
import json
import glob
from pathlib import Path
from tqdm import tqdm

import torch
from esm.inverse_folding.util import load_coords

coords, seq = load_coords(str("structures/Q02127/4OQV_A.cif"), chain="A")
print(coords, seq)
print(f"Sequence length: {len(seq)}")

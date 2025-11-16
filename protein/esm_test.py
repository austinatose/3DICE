import os
import re
import json
import glob
from pathlib import Path
from tqdm import tqdm

import torch
from esm.inverse_folding.util import load_coords

coords, seq = load_coords(str("structures/P46781/5VYC_J1.cif"), chain="J1")
print(coords, seq)
print(f"Sequence length: {len(seq)}")


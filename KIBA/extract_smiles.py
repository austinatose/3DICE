import re
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

input_path = "data/Davis-KIBA/KIBA_targets.txt"
output_path = "drug/KIBA_drug_targets.csv"


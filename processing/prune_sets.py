# remove pairs from CSV that are too long
# FASTA referenced from uniprot_sequences

import argparse
import os
import time
import requests
from typing import Tuple
from tqdm import tqdm
import pandas as pd

MAX_PROT_LEN = 700
MAX_DRUG_LEN = 200

PRUNE_INPUT_PATH = "lists/train.csv"
PRUNE_OUTPUT_PATH = "lists/train_pruned.csv"
DRUG_CHAIN_PATH = "lists/pairs_raw.csv"
PROT_CHAIN_PATH = "other/cache/uniprot_sequences.csv"

input_df = pd.read_csv(PRUNE_INPUT_PATH)
prot_df = pd.read_csv(PROT_CHAIN_PATH)
drug_df = pd.read_csv(DRUG_CHAIN_PATH)

def get_prot_chain(uniprot_id: str, prot_df: pd.DataFrame) -> str:
    row = prot_df[prot_df["uniprot_id"] == uniprot_id]
    if row.empty:
        return ' '*(MAX_PROT_LEN) # cook it
    return row.iloc[0]["sequence"]

def get_drug_chain(drug_id: str, drug_df: pd.DataFrame) -> str:
    row = drug_df[drug_df["drug_id"] == drug_id]
    if row.empty:
        return ' '*(MAX_DRUG_LEN) # cook it
    return row.iloc[0]["smiles"]

for index, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):
    uniprot_id = row["uniprot_id"]
    drug_id = row["drug_id"]

    prot_chain = get_prot_chain(uniprot_id, prot_df)
    drug_chain = get_drug_chain(drug_id, drug_df)

    if len(prot_chain) > MAX_PROT_LEN or len(drug_chain) > MAX_DRUG_LEN:
        input_df.drop(index, inplace=True)

input_df.reset_index(drop=True, inplace=True)
input_df.to_csv(PRUNE_OUTPUT_PATH, index=False)

import pandas as pd

INPUT_CSV = "lists/pairs_filtered_new.csv"

# check for duplicate drug-target pairs
df = pd.read_csv(INPUT_CSV)
duplicates = df.duplicated(subset=['drugbank_id', 'target_uniprot'], keep=False)
duplicate_rows = df[duplicates]

print(duplicate_rows)

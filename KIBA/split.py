import pandas as pd
from sklearn.model_selection import train_test_split

def print_class_balance(df, name="dataset"):
    """
    df: pandas DataFrame with an 'interaction' column (0/1)
    name: label to print (e.g. 'full', 'train', 'val', 'test')
    """
    counts = df["interaction"].value_counts().sort_index()
    total = len(df)

    pos = int(counts.get(1, 0))
    neg = int(counts.get(0, 0))

    print(f"\n=== Class balance for {name} ===")
    print(f"Total samples: {total}")
    print(f"Negatives (0): {neg} ({neg / total:.3f})")
    print(f"Positives (1): {pos} ({pos / total:.3f})")

# Path to your full pairs file
pairs_path = "lists/KIBA/KIBA_pairs.csv"   # adjust if needed

# Load
df = pd.read_csv(pairs_path)

# 80% train+val, 20% test (you can change these)
test_size = 0.1
val_size = 0.1  # fraction of the *total* dataset

# 1) Split off test set
train_val_df, test_df = train_test_split(
    df,
    test_size=test_size,
    random_state=42,
    stratify=df["interaction"]  # keep class balance
)

# 2) Split train vs val from the remaining
# val_size relative to train_val_df
val_relative_size = val_size / (1 - test_size)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=val_relative_size,
    random_state=42,
    stratify=train_val_df["interaction"]
)

print(f"Total: {len(df)}")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

print_class_balance(df, "full")
print_class_balance(train_df, "train")
print_class_balance(val_df, "val")
print_class_balance(test_df, "test")

# Save to CSVs
train_df.to_csv("lists/KIBA/KIBA_pairs_train.csv", index=False)
val_df.to_csv("lists/KIBA/KIBA_pairs_val.csv", index=False)
test_df.to_csv("lists/KIBA/KIBA_pairs_test.csv", index=False)
import pandas as pd
from pathlib import Path

# Paths to the pruned CSVs produced earlier in the pipeline
TRAIN_CSV = "lists/train_pruned.csv"
VAL_CSV = "lists/val_pruned.csv"
TEST_CSV = "lists/test_pruned.csv"

# Rebalance to an overall 8:1:1 split and make the
# positive/negative label ratios as balanced as possible in
# each of train/val/test.

LABEL_COL = "label"  # change here if your label column has a different name
RANDOM_STATE = 42


def load_all_sets(train_path: str, val_path: str, test_path: str) -> pd.DataFrame:
    """Load train/val/test CSVs and concatenate into a single DataFrame.

    Any of the three files may be missing; only existing ones are used.
    """
    paths = [train_path, val_path, test_path]
    dfs = []
    for p in paths:
        p_obj = Path(p)
        if p_obj.is_file():
            dfs.append(pd.read_csv(p_obj))
        else:
            print(f"Warning: {p} not found, skipping.")

    if not dfs:
        raise FileNotFoundError("No input CSVs were found to rebalance.")

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df_all)} total samples from existing splits.")
    return df_all


def stratified_8_1_1_balanced(df: pd.DataFrame,
                               label_col: str = LABEL_COL,
                               random_state: int = RANDOM_STATE):
    """Create 8:1:1 train/val/test splits with (as) balanced labels.

    Strategy:
    - Assume a binary label column with values 0 and 1.
    - Downsample the majority class to match the minority class globally.
    - From this balanced pool, create 8:1:1 splits *separately* for each
      class, then merge so that each split is approximately 50/50.
    """
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in DataFrame.")

    # Shuffle once for reproducibility
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    pos = df[df[label_col] == 1]
    neg = df[df[label_col] == 0]

    n_pos, n_neg = len(pos), len(neg)
    print(f"Original counts -> pos: {n_pos}, neg: {n_neg}")

    if n_pos == 0 or n_neg == 0:
        # Nothing to balance; just do an 8:1:1 split on the whole set
        print("Only one class present; performing plain 8:1:1 split without class balancing.")
        return simple_8_1_1_split(df, random_state=random_state)

    # Downsample majority class to match minority
    n_use = min(n_pos, n_neg)
    pos_bal = pos.sample(n=n_use, random_state=random_state)
    neg_bal = neg.sample(n=n_use, random_state=random_state + 1)

    print(f"Using {n_use} samples from each class (total {2 * n_use}).")

    # Independently shuffle balanced subsets
    pos_bal = pos_bal.sample(frac=1.0, random_state=random_state + 2).reset_index(drop=True)
    neg_bal = neg_bal.sample(frac=1.0, random_state=random_state + 3).reset_index(drop=True)

    def split_one_class(df_class: pd.DataFrame):
        n = len(df_class)
        n_train = int(round(0.8 * n))
        n_val = int(round(0.1 * n))
        # Ensure we don't exceed n due to rounding
        if n_train + n_val > n:
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
        n_test = n - n_train - n_val

        train = df_class.iloc[:n_train]
        val = df_class.iloc[n_train:n_train + n_val]
        test = df_class.iloc[n_train + n_val:]
        return train, val, test

    train_pos, val_pos, test_pos = split_one_class(pos_bal)
    train_neg, val_neg, test_neg = split_one_class(neg_bal)

    train = pd.concat([train_pos, train_neg], ignore_index=True)
    val = pd.concat([val_pos, val_neg], ignore_index=True)
    test = pd.concat([test_pos, test_neg], ignore_index=True)

    # Final shuffles so that classes are mixed
    train = train.sample(frac=1.0, random_state=random_state + 4).reset_index(drop=True)
    val = val.sample(frac=1.0, random_state=random_state + 5).reset_index(drop=True)
    test = test.sample(frac=1.0, random_state=random_state + 6).reset_index(drop=True)

    def describe_split(name: str, df_split: pd.DataFrame):
        total = len(df_split)
        n_pos_split = (df_split[label_col] == 1).sum()
        n_neg_split = (df_split[label_col] == 0).sum()
        print(f"{name}: total={total}, pos={n_pos_split}, neg={n_neg_split}")

    describe_split("Train", train)
    describe_split("Val", val)
    describe_split("Test", test)

    return train, val, test


def simple_8_1_1_split(df: pd.DataFrame,
                        random_state: int = RANDOM_STATE):
    """Fallback: plain 8:1:1 split without class balancing."""
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n = len(df)
    n_train = int(round(0.8 * n))
    n_val = int(round(0.1 * n))
    if n_train + n_val > n:
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]

    print(f"Plain split sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


def main():
    df_all = load_all_sets(TRAIN_CSV, VAL_CSV, TEST_CSV)
    train_df, val_df, test_df = stratified_8_1_1_balanced(df_all)

    # Ensure output directory exists
    out_dir = Path(TRAIN_CSV).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print("Rebalanced splits written to:")
    print(f"  {TRAIN_CSV}")
    print(f"  {VAL_CSV}")
    print(f"  {TEST_CSV}")


if __name__ == "__main__":
    main()

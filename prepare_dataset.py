import pandas as pd
from pathlib import Path

DATASET_DIR = Path.home() / "AI-Network-Traffic-Analyzer" / "dataset"
OUTPUT_DIR = Path.home() / "AI-Network-Traffic-Analyzer" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "binary_network_traffic.csv"

RANDOM_STATE = 42
BENIGN_TARGET = 83_962
ATTACK_TARGET = 83_962
CHUNK_SIZE = 50_000

# We only need one pass through the files.
# Reservoir sampling keeps memory usage low.

benign_samples = []
attack_samples = []

benign_seen = 0
attack_seen = 0

def reservoir_add(samples, row, seen, target):
    if len(samples) < target:
        samples.append(row)
    else:
        import random
        j = random.randint(0, seen - 1)
        if j < target:
            samples[j] = row
    return seen + 1


import random
random.seed(RANDOM_STATE)

print("Starting memory-efficient dataset preparation...")

for file in sorted(DATASET_DIR.glob("Merged*.csv")):

    print(f"\nProcessing: {file.name}")

    for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE):

        # Fix the two missing-value columns
        chunk["Std"] = chunk["Std"].fillna(
            chunk["Std"].median()
        )

        chunk["Variance"] = chunk["Variance"].fillna(
            chunk["Variance"].median()
        )

        # Binary label
        chunk["binary_label"] = (
            chunk["Label"] != "BENIGN"
        ).astype("int8")

        # Process rows individually so we don't keep millions
        # of records in RAM.
        for row in chunk.itertuples(index=False, name=None):

            # Label column is column 39
            # binary_label is column 40
            label = row[39]

            if label == "BENIGN":
                benign_seen += 1

                if len(benign_samples) < BENIGN_TARGET:
                    benign_samples.append(row)
                else:
                    j = random.randint(0, benign_seen - 1)
                    if j < BENIGN_TARGET:
                        benign_samples[j] = row

            else:
                attack_seen += 1

                if len(attack_samples) < ATTACK_TARGET:
                    attack_samples.append(row)
                else:
                    j = random.randint(0, attack_seen - 1)
                    if j < ATTACK_TARGET:
                        attack_samples[j] = row

    print(
        f"Seen so far → "
        f"BENIGN: {benign_seen:,} | "
        f"ATTACK: {attack_seen:,}"
    )

# --------------------------------------------------
# Convert samples to DataFrames
# --------------------------------------------------

print("\nCreating balanced dataset...")

columns = list(pd.read_csv(
    sorted(DATASET_DIR.glob("Merged*.csv"))[0],
    nrows=0
).columns)

columns.append("binary_label")

benign_df = pd.DataFrame(
    benign_samples,
    columns=columns
)

attack_df = pd.DataFrame(
    attack_samples,
    columns=columns
)

balanced_df = pd.concat(
    [benign_df, attack_df],
    ignore_index=True
)

# Shuffle
balanced_df = balanced_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# --------------------------------------------------
# Save
# --------------------------------------------------

print("\nSaving dataset...")

balanced_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)

print(f"Output: {OUTPUT_FILE}")
print(f"Rows: {len(balanced_df):,}")
print(f"Columns: {len(balanced_df.columns)}")

print("\nBinary distribution:")
print(balanced_df["binary_label"].value_counts())

print("\nOriginal labels:")
print(
    balanced_df["Label"]
    .value_counts()
    .head(20)
)

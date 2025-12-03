from datasets import load_dataset
import pandas as pd
import os

# Load dataset
ds = load_dataset("amirhallaji/davis")

# Convert WITHOUT to_pandas()
data_dict = ds["train"][:]        # returns a dict of lists
df = pd.DataFrame(data_dict)      # convert to DataFrame

print("Original columns:", df.columns)

# Rename columns to match your pipeline
df = df.rename(columns={
    "Molecule Sequence": "smiles",
    "Protein Sequence": "protein_sequence",
    "Binding Affinity": "affinity"
})

# Keep only required columns
df = df[["smiles", "protein_sequence", "affinity"]]

# Ensure output folder exists
os.makedirs("data/raw", exist_ok=True)

out_path = "data/raw/davis_processed.csv"
df.to_csv(out_path, index=False)

print(f"\nSaved → {out_path}")
print(f"Rows: {len(df)}")

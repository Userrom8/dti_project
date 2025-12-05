import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def compute_embeddings(
    csv_path,
    esm_model="facebook/esm2_t6_8M_UR50D",
    output_path="data/processed/protein_embeddings.pt",
    device="cuda",
):
    print("Loading CSV:", csv_path)
    df = pd.read_csv(csv_path)

    # Unique protein sequences
    proteins = df["protein_sequence"].unique()
    print(f"Found {len(proteins)} unique protein sequences.")

    # Load ESM
    print("Loading ESM model...")
    tokenizer = AutoTokenizer.from_pretrained(esm_model, use_fast=False)
    model = AutoModel.from_pretrained(esm_model).to(device)
    model.eval()

    embeddings = {}

    with torch.no_grad():
        for seq in tqdm(proteins, desc="Embedding proteins"):
            # Tokenize
            inputs = tokenizer(
                seq, return_tensors="pt", truncation=True, max_length=1024
            ).to(device)

            # Forward
            outputs = model(**inputs)
            pooled = outputs.last_hidden_state[:, 0, :]  # CLS token

            embeddings[seq] = pooled.cpu()

    print("Saving embeddings to:", output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embeddings, output_path)

    print("Done! Saved:", output_path)
    return embeddings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--esm_model", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument(
        "--output", type=str, default="data/processed/protein_embeddings.pt"
    )

    args = parser.parse_args()

    compute_embeddings(
        csv_path=args.csv,
        esm_model=args.esm_model,
        output_path=args.output,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

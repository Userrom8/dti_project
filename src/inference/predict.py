# predict.py

import argparse
import torch

from src.inference.load_model import load_trained_model
from src.inference.preprocess import prepare_inputs
from src.inference.run_inference import run_inference


# src/inference/predict.py

import argparse
import torch

from src.inference.load_model import load_trained_model
from src.inference.preprocess import prepare_inputs
from src.inference.run_inference import run_inference


def main():
    parser = argparse.ArgumentParser(
        description="Predict binding affinity for a drug-target pair."
    )

    # 1. Make --smiles OPTIONAL
    parser.add_argument(
        "--smiles", type=str, default=None, help="SMILES string of the drug (e.g. CCO)"
    )

    # 2. Add --drug_name argument
    parser.add_argument(
        "--drug_name",
        type=str,
        default=None,
        help="Common name of the drug (e.g. Aspirin)",
    )

    parser.add_argument(
        "--protein", type=str, required=True, help="Target protein sequence"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="saved_models/best_checkpoint.pt",
        help="Path to model checkpoint",
    )

    args = parser.parse_args()

    # 3. Validate that at least one input is present
    if not args.smiles and not args.drug_name:
        parser.error("You must provide either --smiles OR --drug_name")

    print(f"Loading model from {args.ckpt}...")
    model = load_trained_model(args.ckpt)
    print("Model loaded successfully.\n")

    # 4. Pass both arguments to prepare_inputs
    # The logic inside prepare_inputs will handle the name->SMILES conversion if needed
    try:
        graph, seq = prepare_inputs(args.smiles, args.protein, args.drug_name)

        affinity = run_inference(model, graph, seq)
        print(f"\nPredicted binding affinity: {affinity:.4f}")

    except ValueError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected Error: {e}")


if __name__ == "__main__":
    main()

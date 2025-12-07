# predict.py

import argparse
import torch

from src.inference.load_model import load_trained_model
from src.inference.preprocess import prepare_inputs
from src.inference.run_inference import run_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", type=str, required=True)
    parser.add_argument("--protein", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="saved_models/best_checkpoint.pt")

    args = parser.parse_args()

    print(f"Loading model from {args.ckpt}...")
    model = load_trained_model(args.ckpt)
    print("Model loaded successfully.\n")

    graph, seq = prepare_inputs(args.smiles, args.protein)

    affinity = run_inference(model, graph, seq)

    print(f"\nPredicted binding affinity: {affinity:.4f}")


if __name__ == "__main__":
    main()

# backend/main.py

import torch
from fastapi import FastAPI
from pydantic import BaseModel

# Import your inference utilities
from src.inference.load_model import load_trained_model
from src.inference.preprocess import prepare_inputs
from src.inference.run_inference import run_inference

MODEL_PATH = "saved_models/best_checkpoint.pt"

app = FastAPI(title="DTI Prediction API", version="1.0.0")

print("🔄 Loading DTI model...")
model = load_trained_model(MODEL_PATH)
print("✅ Model loaded and ready for inference.")

# -------------------------------
# Request / Response Models
# -------------------------------


class PredictionRequest(BaseModel):
    smiles: str
    protein: str


class PredictionResponse(BaseModel):
    affinity: float


# -------------------------------
# Health Check
# -------------------------------
@app.get("/")
def root():
    return {"status": "online", "message": "DTI API running successfully!"}


# -------------------------------
# Predict Endpoint
# -------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    try:
        graph, seq = prepare_inputs(req.smiles, req.protein)
        affinity = run_inference(model, graph, seq)

        return PredictionResponse(affinity=float(affinity))

    except Exception as e:
        return {"error": str(e)}

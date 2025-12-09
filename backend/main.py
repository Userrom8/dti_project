# backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Import your inference utilities
from src.inference.load_model import load_trained_model
from src.inference.preprocess import prepare_inputs
from src.inference.run_inference import run_inference

MODEL_PATH = "saved_models/best_checkpoint.pt"

app = FastAPI(title="DTI Prediction API", version="1.0.0")

# -------------------------------
# CORS (Allow frontend calls)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Load model at startup
# -------------------------------
print("Loading DTI model...")
try:
    model = load_trained_model(MODEL_PATH)
    model.eval()
    print("Model loaded and ready for inference.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")


# -------------------------------
# Request / Response Models
# -------------------------------
class PredictionRequest(BaseModel):
    smiles: Optional[str] = None  # Must be Optional with default None
    drug_name: Optional[str] = None  # Must be Optional with default None
    protein: str


class PredictionResponse(BaseModel):
    affinity: float


# -------------------------------
# Health Check
# -------------------------------
@app.get("/")
def root():
    status = "online" if model else "degraded (model missing)"
    return {"status": status, "message": "DTI API running successfully!"}


# -------------------------------
# Predict Endpoint
# -------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    # 1. Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model failed to load. Check server logs."
        )

    try:
        # 2. Preprocess (Handles Name -> SMILES conversion)
        graph, seq = prepare_inputs(req.smiles, req.protein, req.drug_name)

        # 3. Run Inference
        affinity = run_inference(model, graph, seq)

        return PredictionResponse(affinity=float(affinity))

    except ValueError as ve:
        # 4. Handle invalid input (e.g. Drug Name not found)
        # RAISE exception, do NOT return a dict
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        # 5. Handle unexpected server errors
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

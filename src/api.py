from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import load_model, predict
from transformers import AutoTokenizer

app = FastAPI()

# Load model and tokenizer
MODEL_PATH = 'D:\project\classifier_model.pt'
NUM_CLASSES = 33  # Update this based on your actual number of categories
tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
model = load_model(MODEL_PATH, NUM_CLASSES)

# Map of category indices to category names
category_map = {
    0: 'POLITICS',
    1: 'ENTERTAINMENT',
    # ... add all your categories here
}

class PredictionRequest(BaseModel):
    text: str

@app.post("/api/predict")
async def predict_category(request: PredictionRequest):
    try:
        prediction_idx = predict(request.text, model, tokenizer)
        category = category_map.get(prediction_idx, "Unknown")
        return {"category": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))